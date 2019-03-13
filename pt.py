#!/usr/bin/env python
"""This module implements parallel tempering."""

import sampling
from pathos.multiprocessing import ProcessPool
import numpy as np
import json
import copy
from scipy.optimize import minimize
import os.path
import time

def build_pt(sampler_class, pe_method, force_method, numdim = 5, masses = 1.0, \
    nT = 10, nproc = 1, Tmin = 1.0, Tmax = 100.0, max_iteration = 500, iters_to_swap = 1, \
    iters_to_waypoint = 5, iters_to_setdt = 10, iters_to_writestate = 1, run_token = 1, \
    dt = 1.0e-4, traj_len = 100, num_traj = 10, absxmax = 1.0e2, initial_rand_bounds = 1.0e2, \
    dt_max = None, min_rate = 0.6, max_rate = 0.7, gaussianprior_std = None):
    """Builds an instance of ParallelTempering. Reads restart file if it exists, or initialises a 
    fresh run.
    
    Args:
        sampler_class : Sampler class from module sampling. Eg. sampling.Hmc .
        pe_method : A method for evaluating the potential energy.
        force_method : A method for evaluating the forces.
        numdim (int) :The number of dimensions of the configuration space ('parameter space'). 
            (Defualt: 5)
        masses (single float or numpy array of floats, with length 1 or length numdim): specifies the
            masses associated with each dimension of the configuration space ('parameter space'). 
            (Default: 1.0)
        nT (int) : Number of temperatures to use. (Default: 10)
        nproc (int) : Number of processors to use. (Default: 1)
        Tmin (float) : Lowest temperature in ladder of temperatures. (Default 1.0)
        Tmax (float) : Maximum temperature in ladder of temperatures. (Default 100.0).
        max_iteration (int) : Max number of iterations to run. (Default 500).
        iters_to_swap (int) : Configuration swaps between neighbouring temperatures are attempted
            every iters_to_swap iterations. (Default 1).
        iters_to_waypoint (int) : Restart information is written after every iters_to_waypoint 
            iterations. (Default 5). 
        iters_to_setdt (int) : The step sizes (or equivalently time steps) are updated after every 
            iters_to_setdt interations. (Default 10).
        iters_to_writestate (int) : The latest potential energy values and coordinates are written
            out after every iters_to_writestate iterations. (Default 1).
        run_token (int) : An integer for labelling the restart and output files for this calculation.
            (Default 1).
        dt (float) : Initial time step (or step size). This will be updated algorithmically, but a 
            good starting point saves time. (Default 1.0e-4).
        traj_len (int) : The number of time steps in a single trajectory. (Default 100).
        num_traj (int) : The number of trajectories run per iteration, per sampler. (Default 10).
        absxmax (single float or numpy array of floats, with length 1 or length numdim) : During the 
            main calculation, the sampler is restricted to a region x in [-absxmax,absxmax]. 
            (Default: 1.0e2).
        initial_rand_bounds : The same as absxmax, but applied only during random initialisation of the
            sampler's coordinate (parameters). This enables initialisation into a particular region, 
            which might for example, be most likely to contain the global minimum. (Default: 1.0e2).
        dt_max (float) : maximum step size (time step). (Default: median(absxmax), which is set in 
            module sampling.)
        min_rate (float) : minimum acceptance rate of trajectories. Used for setting step size (time 
            step). (Default: 0.6. The optimal acceptance rate for HMC on a multivariate Gaussian is 0.65
            http://www.mcmchandbook.net/HandbookChapter5.pdf, section 5.4.4.3).
        max_rate (float) : maximum acceptance rate of trajectories. Used for setting step size (time 
            step). (Default 0.7. The optimal acceptance rate for HMC on a multivariate Gaussian is 0.65
            http://www.mcmchandbook.net/HandbookChapter5.pdf, section 5.4.4.3).
        gaussianprior_std (single float or numpy array of floats, with length 1 or length numdim) : If 
            this is set to a real value then an additional term is applied to (H)MC acceptance/rejection 
            such that the target distribution is proportional to a multivariate Gaussian with this 
            standard deviation for each dimension. (Default: None.)

    Return:
        ParallelTempering class object

    """

    # CHECK FOR RESTART FILE AND DO RESTART IF PRESENT
    restrtfl = "restart_pt_"+str(run_token)+".txt"
    if os.path.isfile("./"+restrtfl): # read restart data from restart file
        didrestart = True

        print "Restarting from file ",restrtfl,time.ctime()
        nT, Tmin, Tmax, iteration, num_traj, samplers, walkers = \
            read_waypoint(restrtfl, sampler_class, pe_method, force_method)

    else:
        didrestart = False
        iteration = 0
        # a list of new walkers (which are class objects)
        samplers = build_samplers( sampler_class, pe_method, force_method, nT, Tmin, Tmax, dt, \
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

def build_samplers( sampler_class, pe_method, force_method, nT = 10, Tmin = 1.0, \
    Tmax = 100.0, dt = 1.0e-4, traj_len = 100, absxmax = 1.0e2, dt_max = None, min_rate = 0.6, \
    max_rate = 0.7, gaussianprior_std = None ):
    """Builds a list of nT samplers with temperatures evenly spaced on a log scale from Tmin to Tmax. 
    
    Args:
        sampler_class : Sampler class from module sampling. Eg. sampling.Hmc
        pe_method : A method for evaluating the potential energy.
        force_method : A method for evaluating the forces.
        nT (int) : Number of temperatures to use. (Default: 10)
        Tmin (float) : Lowest temperature in ladder of temperatures. (Default 1.0)
        Tmax (float) : Maximum temperature in ladder of temperatures. (Default 100.0).
        dt (float) : Initial time step (or step size). This will be updated algorithmically, but a 
            good starting point saves time. (Default 1.0e-4).
        traj_len (int) : The number of time steps in a single trajectory. (Default 100).
        absxmax (single float or numpy array of floats, with length 1 or length numdim) : During the 
            main calculation, the sampler is restricted to a region x in [-absxmax,absxmax]. 
            (Default: 1.0e2).
        dt_max (float) : maximum step size (time step). (Default: median(absxmax), which is set in 
            module sampling.)
        min_rate (float) : minimum acceptance rate of trajectories. Used for setting step size (time 
            step). (Default: 0.6. The optimal acceptance rate for HMC on a multivariate Gaussian is 0.65
            http://www.mcmchandbook.net/HandbookChapter5.pdf, section 5.4.4.3).
        max_rate (float) : maximum acceptance rate of trajectories. Used for setting step size (time 
            step). (Default 0.7. The optimal acceptance rate for HMC on a multivariate Gaussian is 0.65
            http://www.mcmchandbook.net/HandbookChapter5.pdf, section 5.4.4.3).
        gaussianprior_std (single float or numpy array of floats, with length 1 or length numdim) : If 
            this is set to a real value then an additional term is applied to (H)MC acceptance/rejection, 
            such that the target distribution is proportional to a multivariate Gaussian with this 
            standard deviation for each dimension. (Default: None.)

    Return:
        List of sampler_class objects, with Temperatures in accending order.

    """

    samplers = []
    for i in xrange(nT):
        beta = 1.0/ith_temperature(Tmin, Tmax, nT, i)
        samplers.append( sampler_class( pe_method, force_method, dt, traj_len, absxmax,\
        dt_max, beta, min_rate, max_rate, gaussianprior_std ) )
    return samplers

class ParallelTempering:
    """This class implements a generic PT algorithm.
    
    Args:
        samplers : list of samplers. Could for example be built with build_samplers
        walkers : numpy array of sampling.NewWalker objects
        num_traj (int) : The number of trajectories run per iteration, per sampler. (Default 10).
        nT (int) : Number of temperatures to use. (Default: 10)
        nproc (int) : number of processes to use in pathos multiprocessing
        Tmin (float) : Lowest temperature in ladder of temperatures. (Default 1.0)
        Tmax (float) : Maximum temperature in ladder of temperatures. (Default 100.0).
        iteration (int) : starting value of iteration counter (Default 0. May differ when restarting calc)
        max_iteration (int) : Max number of iterations to run. (Default 500).
        iters_to_swap (int) : Configuration swaps between neighbouring temperatures are attempted
            every iters_to_swap iterations. (Default 1).
        iters_to_waypoint (int) : Restart information is written after every iters_to_waypoint 
            iterations. (Default 5). 
        iters_to_setdt (int) : The step sizes (or equivalently time steps) are updated after every 
            iters_to_setdt interations. (Default 10).
        iters_to_writestate (int) : The latest potential energy values and coordinates are written
            out after every iters_to_writestate iterations. (Default 1).
        run_token (int) : An integer for labelling the restart and output files for this calculation.
            (Default 1).
        coutfl (str) : File name for writing out configurations (parameter values) every 
            iters_to_writestate iterations. (Default: 'ptconfsout_1.txt'.)
        ptoutfl (str) : File name for writing out all pe values every iters_to_writestate iterations. 
            (Default: 'ptout_1.txt'.)
        restrtfl (str) : File name for restart file. File is updated every iters_to_waypoint iterations.
            (Default: 'restart_pt_1.txt'.)

    Attributes:
        samplers : list of samplers. Could for example be built with build_samplers
        walkers : numpy array of sampling.NewWalker objects
        num_traj (int) : The number of trajectories run per iteration, per sampler. (Default 10).
        nT (int) : Number of temperatures to use. (Default: 10)
        nproc (int) : number of processes to use in pathos multiprocessing
        Tmin (float) : Lowest temperature in ladder of temperatures. (Default 1.0)
        Tmax (float) : Maximum temperature in ladder of temperatures. (Default 100.0).
        iteration (int) : starting value of iteration counter (Default 0. May differ when restarting calc)
        max_iteration (int) : Max number of iterations to run. (Default 500).
        iters_to_swap (int) : Configuration swaps between neighbouring temperatures are attempted
            every iters_to_swap iterations. (Default 1).
        iters_to_waypoint (int) : Restart information is written after every iters_to_waypoint 
            iterations. (Default 5). 
        iters_to_setdt (int) : The step sizes (or equivalently time steps) are updated after every 
            iters_to_setdt interations. (Default 10).
        iters_to_writestate (int) : The latest potential energy values and coordinates are written
            out after every iters_to_writestate iterations. (Default 1).
        run_token (int) : An integer for labelling the restart and output files for this calculation.
            (Default 1).
        coutfl (str) : File name for writing out configurations (parameter values) every 
            iters_to_writestate iterations. (Default: 'ptconfsout_1.txt'.)
        ptoutfl (str) : File name for writing out all pe values every iters_to_writestate iterations. 
            (Default: 'ptout_1.txt'.)
        restrtfl (str) : File name for restart file. File is updated every iters_to_waypoint iterations.
            (Default: 'restart_pt_1.txt'.)
        pt_pool : pathos ProcessPool with nproc workers
        pt_trajs : a list of sampling.Traj objects constructed from the input list samplers, and in the 
            same order.

    """

    def __init__(self, samplers, walkers, num_traj = 10, nT = 10, nproc = 1, Tmin = 1.0, \
        Tmax = 100.0, iteration = 0, max_iteration = 500, iters_to_swap = 1, iters_to_waypoint = 5, \
        iters_to_setdt = 10, iters_to_writestate = 1, run_token = 1, coutfl = "ptconfsout_1.txt", \
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
        """Implements Monte Carlo swaps of the coordinates between neighbouring temperatures, 
        such that the joint distribution is correctly sampled both after and before the swaps.

        Return:
            successes : 1-d numpy array of floats signaling successful swaps between temperature 
                pairs.
            attempts : 1-d numpy array of floats signaling attempted swaps between temperature pairs.

        """

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
        """Propagates all of self.walkers with self.pt_trajs, in parallel.
        
        Return:
            walkers : propagated walkers.
        """

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
        """Sets timesteps (dt) for all self.pt_trajs in parallel.""" 

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
        """Writes the waypoint file to self.restrtfl"""

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
    with i in [0,1,...,nT-1]. series_type must be either "geometric" or "arithmetic".
    
    Args:
        
        Tmin (float) : Lowest temperature in ladder of temperatures. (Default 1.0)
        Tmax (float) : Maximum temperature in ladder of temperatures. (Default 100.0).
        nT (int) : Number of temperatures to use. (Default: 10)
        i (int) : specifies the temperature as described above.
        series_type (str) : specifies the series of temperatures, as described above. Must be 
            'geometric' or 'arithmetic'. (Default: 'geometric'.)

    Return:
        T (float) : output temperature.
    """

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

def read_waypoint(restrtfl, sampler_class, pe_method, force_method):
    """Reads a waypoint from restrtfl
    
    Args:
        restrtfl (str) : file from which to read waypoint
        sampler_class : Sampler class from module sampling. Eg. sampling.Hmc .
        pe_method : A method for evaluating the potential energy.
        force_method : A method for evaluating the forces.

    Return:
        nT (int) : Number of temperatures to use. (Default: 10).
        Tmin (float) : Lowest temperature in ladder of temperatures. (Default 1.0)
        Tmax (float) : Maximum temperature in ladder of temperatures. (Default 100.0).
        iteration (int) : value of iteration counter when waypoint was written.
        num_traj (int) : The number of trajectories run per iteration, per sampler. (Default 10).
        samplers : list of sampling.Sampler objects.
        walkers : numpy array of sampling.NewWalker objects.
    """

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
                sampler = sampler_class(pe_method,force_method,**int_params)
                samplers.append(sampler)

            elif (label == "config"):
                wd = json.loads( line[len(label):].strip() )
                wd["x"] = np.asarray(wd["x"])
                wd["p"] = np.asarray(wd["p"])
                walker = sampling.NewWalker(absxmax = None, sampler_name = None, **wd)
                walker.pe = pe_method(walker.x)
                walkers = np.append(walkers, walker)

    return nT, Tmin, Tmax, iteration, num_traj, samplers, walkers

def initialise_walker(sampler):
    """Robust function for initialising a walker at fixed temperature. 
    1. Generate random configuration. 
    2. Minimise potential energy function. 
    3. Set stepsize and equilibrate that walker to target temperature.

    Args:
        sampler : sampling.Sampler object

    Return:
        walker, sampler : walker (sampling.NewWalker object) and sampler (with dt updated).

    """

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
    """Implements a rough minimisation strategy. Initially momenta are set to zero. Then the system's
    sampler is used to propagate the walker approximately down hill. At each step, if the
    potential energy decreases, the time step is increased by 0.05. However, if the potential
    energy increases during a timestep, then the walker moves backwards one timestep (to the
    lowest point on the trajectory so far), the momenta are reset to 0, and the timestep is
    multiplied by 0.95.

    Args:
        walker : sampling.NewWalker object
        sampler : sampling.Sampler object
        nsteps (int) : total number of timesteps to take (Default: 1000).

    Return:
        walker : sampling.NewWalker object after minimisation.
    """

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
    """Equilibrate walker at temperature 1.0/sampler.beta . Sets dt before and after equilibration.
    
    Args:
        walker : sampling.NewWalker object
        sampler : sampling.Sampler object
        ntraj (int) : number of trajectories to run during equilibration.

    Return:
        walker, sampler : walker and sampler (dt updated)
    """

    this_sampler = copy.deepcopy(sampler)

    therm_step_setter = sampling.UpdateDt( this_sampler , nproc = 1, \
        min_num_data_point = 10, num_walkers = 1 )
    therm_step_setter.set([walker], None, adjust_step_factor = 0.1)
    del therm_step_setter

    this_traj = sampling.Traj(this_sampler, ntraj)
    walker = this_traj.run_serial(walker)
    sampler.dt = this_sampler.dt # pass out final step size

    return walker, sampler
