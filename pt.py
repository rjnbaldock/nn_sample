import sampling
from pathos.multiprocessing import ProcessPool
import numpy as np
import json
import copy
from scipy.optimize import minimize
import os.path
import time

def build_pt(sampler_classobj, pe_method, force_method, numdim = 5, masses = np.ones(5), \
    nT = 10, nproc = 1, Tmin = 1.0, Tmax = 100.0, iteration = 0, max_iteration = 500, iters_to_swap = 1, \
    iters_to_waypoint = 100, iters_to_setdt = 10, iters_to_writestate = 100, run_token = 1, \
    dt = 1.0e-4, traj_len = 100, num_traj = 10, absxmax = 1.0e2, dt_max = None, \
    min_rate = 0.6, max_rate = 0.7, gaussianprior_std = None, initial_rand_bounds = 1.0e2):

    # CHECK FOR RESTART FILE AND DO RESTART IF PRESENT
    restrtfl = "restart_pt_"+str(run_token)+".txt"
    if os.path.isfile("./"+restrtfl): # read restart data from restart file
        didrestart = True

        print "Restarting from file ",restrtfl,time.ctime()
        nT, Tmin, Tmax, iteration, num_traj, samplers, walkers = \
            read_waypoint(restrtfl, sampler_classobj, pe_method, force_method)

    else:
        didrestart = False
        # a list of new walkers (which are class objects)
        samplers = build_samplers( sampler_classobj, pe_method, force_method, nT, Tmin, Tmax, dt, \
            traj_len, absxmax, dt_max, min_rate, max_rate, gaussianprior_std )

        print "Start initialise walkers ",time.ctime()
        walkers = np.asarray([])

        sampling.NewWalker.masses = masses
        sampling.NewWalker.numdim = numdim
        temp_pool = ProcessPool(nodes=nproc)

        # temporarily pass initial_random_bounds through samplers, since pathos multiprocessing is 
        # restrictive with arguments
        for sampler in samplers:
            sampler.random_init_bounds = initial_rand_bounds

        outs = sampling.apply_pool(temp_pool, initialise_walker, samplers)
        for i in xrange(len(outs)):
            walkers = np.append(walkers,outs[i][0])
            samplers[i] = outs[i][1]

        temp_pool.terminate() # close pool
        temp_pool.restart() # close pool
        print "Done initialise walkers ",time.ctime()

    coutfl = "ptconfsout_"+str(run_token)+".txt"
    ptoutfl = "ptout_"+str(run_token)+".txt"

    thispt = ParallelTempering(samplers, walkers, num_traj, nT, nproc, Tmin, Tmax, iteration, \
        max_iteration, iters_to_swap, iters_to_waypoint, iters_to_setdt, iters_to_writestate, run_token, coutfl,\
        ptoutfl, restrtfl )

    if (not didrestart):
        thispt.set_dt_all(thispt.pt_pool, step_fac = 0.1)

    return thispt

def build_samplers( sampler_classobj, pe_method, force_method, nT = 10, Tmin = 1.0, \
    Tmax = 100.0, dt = 1.0e-4, traj_len = 100, absxmax = 1.0e2, dt_max = None, min_rate = 0.6, \
    max_rate = 0.7, gaussianprior_std = None ):
    """Builds a list of samplers using ptparams. sampler_classobj should be an sampler class
    from module sampling. For example, "sampling.Hmc"."""

    samplers = []
    for i in xrange(nT):
        beta = 1.0/ith_temperature(Tmin, Tmax, nT, i)
        samplers.append( sampler_classobj( pe_method, force_method, dt, traj_len, absxmax,\
        dt_max, beta, min_rate, max_rate, gaussianprior_std ) )
    return samplers

class ParallelTempering:
    """This class implements a generic PT algorithm."""

    def __init__(self, samplers, walkers, num_traj = 10, nT = 10, nproc = 1, Tmin = 1.0, \
        Tmax = 100.0, iteration = 0, max_iteration = 500, iters_to_swap = 1, iters_to_waypoint = 100, \
        iters_to_setdt = 10, iters_to_writestate = 100, run_token = 1, coutfl = "ptconfsout_1.txt", \
        ptoutfl = "ptout_1.txt", restrtfl = "restart_pt_1.txt"):

        self.samplers = samplers
        self.walkers = walkers

        self.nT = nT
        self.nproc = nproc
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.iteration = iteration
        self.max_iteration = max_iteration
        self.iters_to_swap = iters_to_swap
        self.iters_to_waypoint = iters_to_waypoint
        self.iters_to_setdt = iters_to_setdt
        self.iters_to_writestate = iters_to_writestate
        self.run_token = run_token
        self.coutfl = coutfl
        self.ptoutfl = ptoutfl
        self.restrtfl = restrtfl

        self.pt_pool = ProcessPool(nodes=self.nproc) # pool - multiprocessing with pathos

        self.pt_trajs = [sampling.Traj(sampler, num_traj) for sampler in self.samplers]
        del self.samplers # from here on we deal only with self.pt_trajs

        # initialise a number of trajectory instances, corresponding to the samplers
        self.cout = open(self.coutfl,"a")
        self.ptout = open(self.ptoutfl,"a")

    def ptmain(self):
        """Main parallel tempering routine."""

        print "Start Parallel Tempering ", time.ctime()

        if (self.iteration == 0): 
            self.write_waypoint() # write a restart point at the start of the calculation, before setting dt etc

        for it in xrange(self.max_iteration):

            self.iteration += 1

            if (it%self.iters_to_setdt == 0):        # set stepsizes
                self.set_dt_all(self.pt_pool)

            if (it%self.iters_to_swap == 0):
                this_successes, this_trials = self.pt_swaps()    # do walker swaps

                print "Acceptance of T swaps: "
                for i in xrange(self.nT-1):
                    print i, i+1, "(betas =",self.pt_trajs[i].sampler.beta, \
                        self.pt_trajs[i+1].sampler.beta,"): ", this_successes[i]

            self.walkers = self.propagate_parallel() # propagate all walkers

            if ((it+1)%self.iters_to_waypoint == 0):             # write waypoint
                self.write_waypoint()

            if (it%self.iters_to_writestate == 0):
                # note that after restart, if iters_to_writestate<iters_to_waypoint, as is 
                # generally the case, the code will write additional lines of output for iterations 
                # since the last waypoint. This is intentional: it seems wasteful to automatically 
                # throw away the results of additional sampling - the user can decide whether to 
                # keep them or not.
                self.write_state()

    def pt_swaps(self):

        # build a list of adjacent integers, of even length, equally likely to start with 0 or 1
        start = int( np.random.uniform()<0.5 )
        swap_ids = range(start,len(self.walkers)-(len(self.walkers)-start)%2)
        it = iter(swap_ids)
        swap_ids = zip(it,it)

        successes = np.zeros(self.nT)
        attempts = np.zeros(self.nT)
        for pair in swap_ids:
            id1, id2 = pair
            beta1 = self.pt_trajs[id1].sampler.beta
            beta2 = self.pt_trajs[id2].sampler.beta
            te1 = self.walkers[id1].pe
            te2 = self.walkers[id2].pe
            logprob_accept = (beta2 - beta1)*(te2 - te1)

            if (np.log(np.random.uniform())<logprob_accept): # accept swap
                buf = sampling.copy_walker(self.walkers[id1], self.pt_trajs[id1].sampler.must_copy_p)
                self.walkers[id1] = sampling.copy_walker(self.walkers[id2], \
                    self.pt_trajs[id2].sampler.must_copy_p)
                self.walkers[id2] = sampling.copy_walker(buf, self.pt_trajs[id1].sampler.must_copy_p)
                successes[id1] += 1.0

            attempts[id1] += 1.0

        return successes, attempts

    def propagate_parallel(self):
        """Propagates a number of walkers with different samplers, in parallel."""

        def one_run(i):
            np.random.seed()    # reinitialise the random seed, otherwise the trajectories 
                                # become correlated
            this_traj = self.pt_trajs[i]
            this_walker = self.walkers[i]
            walker_out = this_traj.run_serial(this_walker)
            return walker_out

        walkers = np.asarray( sampling.apply_pool( self.pt_pool,one_run,range(len(self.pt_trajs)) ) )
        return walkers

    def set_dt_all(self, pool, step_fac = 0.9):
        """Sets timesteps for all samplers. These routines are already parallel, 
        so each dt is set in series.""" 

        def set_one_dt(i):
            np.random.seed()
            this_walker = self.walkers[i]
            this_sampler = self.pt_trajs[i].sampler
            this_step_setter = sampling.UpdateDt( this_sampler, nproc = 1, min_num_data_point = 10, \
                num_walkers = 1 )
            prfx = "iteration "+str(self.iteration)+" beta "+str(this_sampler.beta)
            this_step_setter.set([this_walker], prfx, adjust_step_factor = step_fac) 
                # must put walker into an array for .set()
            return this_sampler.dt

        dts = sampling.apply_pool( pool, set_one_dt, xrange(len(self.pt_trajs)) )

        # update dts
        for i in xrange(len(self.pt_trajs)):
            self.pt_trajs[i].sampler.dt = dts[i]

    def write_waypoint(self):
        """Writes the waypoint file"""

        ptparams = {}
        ptparams["nT"] = self.nT
        ptparams["nproc"] = self.nproc
        ptparams["Tmin"] = self.Tmin
        ptparams["Tmax"] = self.Tmax
        ptparams["iteration"] = self.iteration
        ptparams["max_iteration"] = self.max_iteration
        ptparams["iters_to_waypoint"] = self.iters_to_waypoint
        ptparams["iters_to_setdt"] = self.iters_to_setdt
        ptparams["iters_to_writestate"] = self.iters_to_writestate
        ptparams["run_token"] = self.run_token
        ptparams["masses"] = sampling.NewWalker.masses
        if type(ptparams["masses"]) is np.ndarray:
            ptparams["masses"] = ptparams["masses"].tolist()
        ptparams["numdim"] = sampling.NewWalker.numdim

        with open(self.restrtfl,"w") as rout:

            l = "ptparams "+json.dumps(ptparams)+"\n"
            rout.write(l)

            for this_traj in self.pt_trajs:
                sampler = this_traj.sampler

                int_params = {}
                int_params["dt"] = sampler.dt
                int_params["traj_len"] = sampler.traj_len
                int_params["absxmax"] = sampler.absxmax
                int_params["beta"] = sampler.beta
                int_params["dt_max"] = sampler.dt_max
                int_params["name"] = sampler.name
                int_params["min_rate"] = sampler.min_rate
                int_params["max_rate"] = sampler.max_rate
                int_params["num_traj"] = this_traj.num_traj
                int_params["gaussianprior_std"] = this_traj.sampler.gaussianprior_std

                if type(int_params["absxmax"]) is np.ndarray:
                    int_params["absxmax"] = int_params["absxmax"].tolist()

                l = "int_params "+json.dumps(int_params)+"\n"
                rout.write(l)

            for walker in self.walkers:
                sampling.write_walker(walker,rout,'config ')

    def write_state(self):
        """Writes current state: pe values go to self.ptout and configs to self.cout."""

        # self.ptout line contains: #iteration, then pe value for each walker in sequence
        l = str(self.iteration)+" "
        for i in xrange(self.nT):
            l += str(self.walkers[i].pe)+" "
        l += "\n"
        self.ptout.write(l)

        # self.cout line contains #iteration, #beta, walker (nT lines to report one state)
        for i in xrange(self.nT):
            spfx = str(self.iteration)+" "+str(self.pt_trajs[i].sampler.beta)+" "
            sampling.write_walker(self.walkers[i],self.cout,spfx)

def ith_temperature(Tmin,Tmax,nT,i, series_type = "geometric"):
    """Gives the ith temperature in a geometric or arithmetic progression from Tmin to Tmax,
    with i in [0,1,...,nT-1]. series_type must be either "geometric" or "arithmetic"."""

    if (nT<=1):
        T = Tmin
    else:
        if (series_type == "geometric"):
            T = Tmin * (float(Tmax)/Tmin)**(float(i)/(nT-1))
        elif (series_type == "arithmetic"):
            T = Tmin + float(Tmax-Tmin)*i/(nT-1) 
        else:
            sampling.exit_error("pt.ith_temperature requires series_type is 'geometric' or"+\
                " 'arithmetic'. Got "+str(series_type),13)

    return T

def read_waypoint(restrtfl, sampler_classobj, pe_method,force_method):
    """Reads the waypoint file"""
    walkers = np.asarray([])
    samplers = []
    with open(restrtfl) as rin:
        for line in rin:
            label = line.split()[0]
            if (label == "ptparams"):
                ptparams = json.loads( line[len(label):].strip() )

                nT = ptparams["nT"]
                Tmin = ptparams["Tmin"]
                Tmax = ptparams["Tmax"]
                iteration = ptparams["iteration"]
                sampling.NewWalker.masses = np.asarray(ptparams["masses"])
                sampling.NewWalker.numdim = ptparams["numdim"]

            elif (label == "int_params"):
                int_params = json.loads( line[len(label):].strip() )
                num_traj = int_params["num_traj"]
                del int_params["num_traj"]
                del int_params["name"]
                int_params["absxmax"] = np.asarray(int_params["absxmax"])
                sampler = sampler_classobj(pe_method,force_method,**int_params)
                samplers.append(sampler)

            elif (label == "config"):
                wd = json.loads( line[len(label):].strip() )
                wd["x"] = np.asarray(wd["x"])
                wd["p"] = np.asarray(wd["p"])
                walker = sampling.NewWalker(absxmax = None, sampler_name = None, **wd)
                walkers = np.append(walkers, walker)

    return nT, Tmin, Tmax, iteration, num_traj, samplers, walkers

def initialise_walker(sampler):
    """Robust function for initialising a walker at fixed temperature. 
    1. Generate random configuration. 
    2. Minimise potential energy function. 
    3. Slowly thermalise (reheat) that walker to target temperature.
    Exceptionally, this routine assumes that numdim and masses are part of sampler. This is for ease of passing
    to pathos."""

    max_num_trials = 10    # robust default value

    # generate a random configuration
    walker = sampling.NewWalker(sampler.random_init_bounds, sampler.name, \
        pe_method = sampler.pe_method, beta = sampler.beta )
    walker.p *= 0.0

    print "beta ",sampler.beta," START minimise walker. Initial pe: ",walker.pe,time.ctime()
    for i in xrange(max_num_trials):
        walker = rough_minimise(walker, sampler)
        if (max(np.abs(walker.x)-sampler.absxmax)<0.0):
            break
    if (max(np.abs(walker.x)-sampler.absxmax)>0.0):
        sampling.exit_error("Even after "+str(max_num_trials)+" attempts, minimisation failed. Is absxmax too small?",11)

    print "beta ",sampler.beta," END minimise walker. Final pe: ",walker.pe,time.ctime()

    # set rough dt and do 100 traj of burn-in
    print "beta ",sampler.beta," Setting approximate dt values and doing 20 trajectories of burn in "
    walker, sampler = thermalise_walker(walker,sampler,20)
    print "beta ",sampler.beta, \
        " re-setting approximate dt values and doing a further 80 trajectories of burn in "
    walker, sampler = thermalise_walker(walker,sampler,80)

    return walker, sampler

def rough_minimise(walker, sampler, nsteps = 1000):

    save_dt = sampler.dt

    walker.p = np.zeros(walker.numdim)
    walker.pe = sampler.pe_method(walker.x)
    for i in xrange(nsteps):
        walker_save = sampling.NewWalker(None,sampler.name,x=walker.x,p=walker.p, pe = walker.pe, \
            ke = 0.0)
        walker = sampler.propagate(walker)
        walker.pe = sampler.pe_method(walker.x)
        if (walker.pe>walker_save.pe):
            walker = sampling.NewWalker(None,sampler.name,x=walker_save.x,p=np.zeros(walker.numdim), \
                pe = walker_save.pe, ke = 0.0)
            sampler.dt *= 0.95
        else:
            sampler.dt += 0.05

    sampler.dt = save_dt # reset dt
    walker.p = np.zeros(walker.numdim)
    walker.ke = 0.0
    walker.pe = sampler.pe_method(walker.x)
    return walker

def thermalise_walker(walker,sampler, ntraj):
    """Set rough step length and do ntraj trajectories of burn in."""

    this_sampler = copy.deepcopy(sampler)

    therm_step_setter = sampling.UpdateDt( this_sampler , nproc = 1, \
        min_num_data_point = 10, num_walkers = 1 )
    therm_step_setter.set([walker], None, adjust_step_factor = 0.1)
    del therm_step_setter

    this_traj = sampling.Traj(this_sampler, ntraj)
    walker = this_traj.run_serial(walker)
    sampler.dt = this_sampler.dt # pass out final step size

    return walker, sampler
