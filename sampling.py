import numpy as np
import sys
import copy
from pathos.multiprocessing import ProcessPool
import time
from operator import itemgetter
import abc
import json

class NewWalker:
    """Either creates a new random walker with coordinates uniformly randomly chosen from 
    [-absxmax,+absxmax]. Handles missing data."""

    masses = None
    numdim = None

    def __init__(self, absxmax, sampler_name, pe_method = None, \
            x = None, p = None, pe = None, ke = None, beta = None):

        if (x is None):
            if (pe_method is None):
                exit_error('ERROR: NewWalker got x = None and pe_method = None.' + \
                    ' At least one must be specified.',15)

            np.random.seed()    # reinitialise the random seed so that processes called in 
                                # parallel generate different structures
            self.x = np.multiply(2.0*(np.random.uniform(size = self.numdim)-0.5),absxmax)
        else:
            self.x = x.copy()

        if (p is None):
            if (sampler_name=='hmc'):
                if (beta is None):
                    exit_error('Error, sampling.NewWalker: beta must be specified for hmc sampler',10)
                self.p = Hmc.mom_update(numdim = self.numdim, masses=self.masses, beta=beta)
            elif (sampler_name=='gmc'):
                self.p = Gmc.mom_update(numdim = self.numdim)
            else:
                exit_error('ERROR: initialise self. sampler_name must be "hmc" or "gmc". Got '+ \
                    sampler_name,10)
        else:
            self.p = p.copy()

        if (pe is None):
            self.pe = pe_method(self.x)
        else:
            self.pe = pe

        if (ke is None):
            self.ke = Traj.ke(self.p,self.masses)
        else:
            self.ke = ke

class Traj:
    """ This class implements a hmc (or gmc) trajectory.
    Note that GMC is a form of hmc."""

    def __init__(self, sampler, num_traj = 1):
        self.sampler = sampler          # a class with methods
                                        # 1. propagate : a method propagate that integrates 
                                        # through one time step
                                        # 2. pe_method : a method for calculating the potential 
                                        # energy
                                        # 3. force_method : a method for calculating the forces
                                        # 4. log_accept_prob : a method that gives the log 
                                        # probability of accepting or rejecting the trajectory 
                                        # 5. mom_update: a method that generates initial momenta
        self.num_traj = num_traj        # number of calls to self.run performed by run_serial


    @staticmethod
    def ke(momenta,masses):
        """A function for calculating the kinetic energy of the sampler."""

        ke = 0.5 * np.sum((momenta**2) / masses)    # ke = (p^2)/(2m)

        return ke

    def run(self,walker_in):
        """Run one H(G)MC trajectory 1. Stochastic momentum update. 2. Propagate dynamics.
        3. Do accept/reject. If trajectory rejected, flip momenta, so that the trajectory 
        ends as just before step 2, but with the momenta reversed.
        """
        # walker_in is a class containting the the initial starting 
        # configuration (x) and momenta (p), as well as the initial
        # potential and kinetic energies

        pe_initial = walker_in.pe

        walker_out = copy_walker(walker_in, self.sampler.must_copy_p)

        try: # not all samplers have attribute beta.
            this_beta = self.sampler.beta
        except AttributeError:
            this_beta = None

        walker_out.p = self.sampler.mom_update(walker_out.p,walker_out.numdim,walker_out.masses, \
            this_beta)
        if ( self.sampler.must_copy_p ):
            save_p = walker_out.p.deepcopy()
        else:
            save_p = np.asarray(1.0)
        ke_initial = self.ke(walker_out.p, walker_out.masses)

        inout_vals = {}
        if (self.sampler.gaussianprior_std is not None): # calculate x^2 for MC over a gaussian prior
            inout_vals["initial_xsq"] = np.inner(walker_in.x,walker_in.x)

        for step in xrange(self.sampler.traj_len):

            walker_out = self.sampler.propagate(walker_out)
            if (max(np.abs(walker_out.x)-10.0*self.sampler.absxmax)>0.0): 
            # check the walker hasn't gone so 
            # far out of bounds that it is unlikely to return. This is a way of catching bad integration 
            # before overflows occur.
                break

        # If pe and ke have not been evaluated in last call to propagate, they will be set to None
        if (walker_out.pe is None):
            walker_out.pe = self.sampler.pe_method(walker_out.x)
        if (walker_out.ke is None):
            walker_out.ke = self.ke(walker_out.p, walker_out.masses)

        inout_vals["pe_initial"] = walker_in.pe
        inout_vals["ke_initial"] = ke_initial
        inout_vals["pe_final"] = walker_out.pe
        inout_vals["ke_final"] = walker_out.ke

        if (self.sampler.gaussianprior_std is not None): # calculate x^2 for MC over a gaussian prior
            inout_vals["final_xsq"] = np.inner(walker_out.x,walker_out.x)

        logprob_accept = self.sampler.log_accept_prob(inout_vals)

        if ( (np.log(np.random.uniform()) < logprob_accept) and (max(np.abs(walker_out.x)-self.sampler.absxmax)<0.0) ): # accept final configuration in trajectory

            traj_accepted = True
        else: # we return the initial configuration with the momenta reversed
            walker_out = walker_in
            walker_out.p = - save_p
            traj_accepted = False

        return walker_out, traj_accepted

    def run_serial(self,walker):
        """This method does Traj.run on walker, num_traj times in a row."""

        buf = copy_walker(walker, self.sampler.must_copy_p)
        srate = 0
        for i in xrange(self.num_traj):
            buf, logical_spare = self.run(buf)
            srate += int(logical_spare)

        print "beta ",self.sampler.beta, " live success rate: ", float(srate)/self.num_traj

        return buf

class Sampler:
    """Super class of all samplers."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def propagate(self, walker):
        pass

    @abc.abstractmethod
    def log_accept_prob(self, inout_vals):
        pass

    @abc.abstractmethod
    def mom_update(p, numdim, masses, beta):
        pass

class Gmc(Sampler):
    """gmc sampler contains propagator, log_accept_prob and mom_update methods for gmc with total
    momentum randomisation."""

    def __init__(self, pe_method,force_method, dt = 0.1, traj_len = 100, absxmax = 1.0e2, dt_max = None, emax = sys.float_info.max, min_rate = 0.6, max_rate = 0.7):

        Sampler.__init__(self)
        self.name = 'gmc'
        self.pe_method = pe_method
        self.force_method = force_method
        self.emax = emax
        self.dt = dt
        self.absxmax = absxmax
        self.traj_len = traj_len
        self.min_rate = min_rate
        self.max_rate = max_rate
        if (dt_max is None): 
            self.dt_max = np.median(absxmax)
            # The sampler crosses a characteristic lengthscale of the space in two steps
        else:
            self.dt_max = dt_max
        self.must_copy_p = False
        self.gaussianprior_std = None

    def propagate(self,walker):
        """Does one dt of GMC propagation. A propagator updates x and p. If pe and/or ke have
        not been explicitly re-evaluated at the end of the trajectory, then they are set to None."""

        walker.pe, walker.ke = None, None

        walker.x += self.dt * walker.p / walker.masses
        walker.pe = self.pe_method(walker.x)
        if (walker.pe>=self.emax):
            v = walker.p/walker.masses
            n = -self.force_method(walker.x)
            n /= np.linalg.norm(n)
            v -= 2.0 * n * np.inner(n,v)
            walker.p = walker.masses*v

        return walker

    def log_accept_prob(self,inout_vals):
        """Returns 1.0 if inout_vals["pe_final"] < self.emax and 0.0 otherwise.
            ener_new should be the current potential energy."""

        if (inout_vals["pe_final"] < self.emax):
            logprob = 0.0
        else:
            logprob = - sys.float_info.max
        return logprob

    @staticmethod
    def mom_update(p = None, numdim = 5, masses = np.ones(5), beta = 1.0): # this routine can be
                                        # used stand alone function 
        """Generates a random sample from the surface of a unit hypershpere centered at the origin in 
        numdim dimensions."""
        pout = np.random.normal(size=numdim)
        pout /= np.linalg.norm(pout)
        return pout

class Hmc(Sampler):
    """Hmc sampler contains propagator, log_accept_prob and mom_update methods for hmc at fixed 
    temperature with total momentum randomisation."""

    def __init__(self,pe_method, force_method, dt = 0.1, traj_len = 100, \
        absxmax = 1.0e2, dt_max = None, beta = 1.0, min_rate = 0.6, max_rate = 0.7, \
        gaussianprior_std = None):

        Sampler.__init__(self)
        self.name = 'hmc'
        self.pe_method = pe_method
        self.force_method = force_method
        self.beta = beta
        self.dt = dt
        self.absxmax = absxmax
        self.traj_len = traj_len
        self.min_rate = min_rate
        self.max_rate = max_rate
        if (dt_max is None):
            self.dt_max = np.median(absxmax)
            # The sampler crosses a characteristic lengthscale of the space in two steps
        else:
            self.dt_max = dt_max
        self.gaussianprior_std = gaussianprior_std
        self.must_copy_p = False

    def propagate(self,walker):
        """Does one dt of HMC propagation. A propagator updates x and p. If pe and/or ke have
        not been explicitly re-evaluated at the end of the trajectory, then they are set to None."""

        walker.pe, walker.ke = None, None

        accel_t1 = self.force_method(walker.x) / walker.masses
        walker.x += self.dt * walker.p / walker.masses \
            + 0.5 * accel_t1 * (self.dt**2)
        accel_t2 = self.force_method(walker.x) / walker.masses

        walker.p += 0.5 * self.dt * (accel_t1+accel_t2) * walker.masses

        return walker

    def log_accept_prob(self,inout_vals):
        """Returns exp(-beta *[total_energy_new - total_energy_old ])."""

        logprob = - self.beta*( inout_vals["pe_final"] + inout_vals["ke_final"] \
            - inout_vals["pe_initial"] - inout_vals["ke_initial"] )

        if (self.gaussianprior_std is not None):
            logprob += -(0.5/self.gaussianprior_std**2) * (inout_vals["final_xsq"] - inout_vals["initial_xsq"])

        return logprob

    @staticmethod
    def mom_update(p = None, numdim = 5, masses = np.ones(5), beta = 1.0): 
        """Generates a random sample from a numdim-d Gaussian, at the correct temperature."""
        pout = np.random.normal(scale=np.sqrt(masses/beta),size=numdim)

        return pout

class UpdateDt(Traj):
    """This class implements stepsize updating. """

    def __init__(self, sampler, nproc = 1, min_num_data_point = 10, num_walkers = 1):
        Traj.__init__(self, sampler, 1)
        self.nproc = nproc # number of processes to use for parallel sampling
        self.min_num_data_point = min_num_data_point # minimum number of datapoints (trajectories) to
                                        # use when sampling acceptance rate data
        self.num_walkers = num_walkers  # number of independent samplers/particles/walkers we have
                                        # with the same sampler and same thermodynamic conditions

    def print_dt_change(self,old_dt,new_dt,message_prefix):
        """Prints a statement that step size has been updated from old_dt to new_dt. 
        Print nothing if message_prefix is none."""
        if (message_prefix is not None):
            print message_prefix +' '+self.sampler.name," stepsize adjusted from %.3E to %.3E" % \
                (old_dt, new_dt)
        pass

    def set(self,walkers,message_prefix, adjust_step_factor = 0.9):
        """Updates the stepsize to achieve a trajectory acceptance rate in or as close as possible
        to the range [self.sampler.min_rate, self.sampler.max_rate], with stepsize in the range
        [10^-50, self.sampler.dt_max]. adjust_step_factor: step size is updated by * or / by this value.
        message_prefix is printed before any output. No output will be printed if 
        message_prefix is None."""

        start_time = time.time()
        if (self.nproc > 1):
            set_pool = ProcessPool(nodes=self.nproc)
        else:
            set_pool = None

        steplength_store = self.sampler.dt
        steplength_in = self.sampler.dt
        # protects against possible future bugs that would be hard to detect

        walk_n_walkers = int(self.nproc * np.ceil(float(self.min_num_data_point)/self.nproc))
        # rounds up to next multiple of self.nproc for maximum usage of compute

        first_time = True # we will make at least two tries. Logical flag ensures this.

        # Step size calibration loop:
        while True:

            # collect statistics on trajectory acceptance rate
            run_outputs = apply_pool(set_pool, self.run, np.random.choice(walkers,size=walk_n_walkers))
            results = map(itemgetter(1), run_outputs)
            del run_outputs

            # The total number of accepted/rejected moves for this step size
            rate = float(np.sum(results))/walk_n_walkers

            if (rate>=self.sampler.min_rate and rate<=self.sampler.max_rate):
                # If the total acceptance rate is within the desired range, return this stepsize
                self.print_dt_change(steplength_in, self.sampler.dt, message_prefix)
                break
            else: # update the stepsize to get closer to the desired range
                if( not first_time ): # dodge this the first time round - no rate_store saved yet
                    # Check whether rate and rate_store are on different sides 
                    # of interval
                    if ((min(rate,rate_store) < self.sampler.min_rate) and (max(rate,rate_store) > self.sampler.max_rate)):
                        # We previously obtained an acceptance rate on one side of the desired range 
                        # and now find an acceptance rate on the other side. We return the step size 
                        # that gave an acceptance rate closest to the middle of the desired range.

                        target = 0.5*(self.sampler.min_rate+self.sampler.max_rate) # middle of range
                        if (abs(rate-target)<abs(rate_store-target)):
                            # take current step length
                            self.print_dt_change(steplength_in, self.sampler.dt, \
                                message_prefix)
                            break
                        else:
                            # take saved step length
                            self.sampler.dt = steplength_store
                            rate = rate_store
                            self.print_dt_change(steplength_in, self.sampler.dt, \
                                message_prefix)
                            break

                else: # this is the fist time - no rate_store saved yet
                    first_time = False

                # save current step length and acceptance rate
                steplength_store = self.sampler.dt
                rate_store = rate

                # update step length
                if rate < self.sampler.min_rate:
                    exp = 1.0
                elif rate >= self.sampler.max_rate:
                    exp = -1.0

                # try to adjust
                self.sampler.dt *= adjust_step_factor**exp

                # Check that step size is neither larger than max allowed value nor smaller than 
                # 10^-50 (useful for detecting errors).
                # Error check:
                if (self.sampler.dt < 1.0e-50):
                    if (message_prefix is not None):
                        prfx = message_prefix + " stepsizes got stepsize= '%e': too small. Is everything correct?\n" % \
                        (self.sampler.dt)
                    else:
                        prfx = " stepsizes got stepsize= '%e': too small. Is everything correct?\n" % \
                        (self.sampler.dt)

                    exit_error(prfx, 25)

                # sampling demands a step size larger than dt_max. Set to dt_max then break
                if (self.sampler.dt>self.sampler.dt_max):
                    self.sampler.dt = self.sampler.dt_max
                    self.print_dt_change(steplength_in, self.sampler.dt, \
                        message_prefix)
                    break

        # close pool
        if (set_pool is not None):
            set_pool.terminate()
            set_pool.restart()

        end_time = time.time()
        duration = end_time - start_time
        return duration

def copy_walker(walker_in, copy_p = False):
    """Returns a copy of a walker"""

    if (copy_p):
        walker_out = NewWalker( absxmax = None, sampler_name = None, x = walker_in.x, p = walker_in.p, \
            pe = walker_in.pe, ke = walker_in.ke )
    else:
        walker_out = NewWalker( absxmax = None, sampler_name = None, x = walker_in.x, p = np.arange(1), \
            pe = walker_in.pe, ke = walker_in.ke )

    return walker_out

def write_walker(walker,open_file,string_prefix):
    """Write one walker."""
    wd = {}
    wd["x"] = walker.x.tolist()
    wd["p"] = walker.p.tolist()
    wd["pe"] = walker.pe
    wd["ke"] = walker.ke

    l = string_prefix+json.dumps(wd)+"\n"
    open_file.write(l)

def apply_pool(pool, func, iters):
    """If we are using only one processor, escape overhead of pathos multiprocessing"""

    if (pool is None):
        poolsize = 1
    else:
        poolsize = pool._ProcessPool__get_nodes()

    if (poolsize>1):
        outs = pool.map(func,iters)
    else:
        outs = [func(it) for it in iters]
    return outs

def exit_error(message, stat):
    sys.stderr.write(message)
    sys.exit(stat)
