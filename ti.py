#!/usr/bin/env python
"""This module implements thermodynamic integration for calculating the free energy of a crystal."""

import sampling
import numpy as np
import json
import time
import argparse
import copy
import os.path

from pprint import pprint

def build_ti(sampler_class, pe_method, force_method, initial_coords, \
    masses = 1.0, T = 1.0, n_bridge = 100, iters_to_waypoint = 10, \
    times_to_setdt = 10, run_token = 1, dt = 1.0e-1, traj_len = 100, ntraj_burnin=100, \
    ntraj_sample = 100, absxmax = 1.0e2, dt_max = None, min_rate = 0.6, max_rate = 0.7, \
    gaussianprior_std = None):
    """Builds an instance of ThermodynamicIntegration. Reads restart file if it exists, or initialises a 
    fresh run.
    
    Args:
        sampler_class : Sampler class from module sampling. Eg. sampling.Hmc .
        pe_method : A method for evaluating the potential energy.
        force_method : A method for evaluating the forces.
        initial_coords : A numpy array containing the coordinates (parameter values) corresponding
            to a local minimum of the potential energy surface. A quadratic potential with minimum
            at initial_coords will be fitted to the potential energy surface.
        masses (single float or numpy array of floats, with length 1 or length numdim): specifies the
            masses associated with each dimension of the configuration space ('parameter space'). 
            (Default: 1.0)
        T (float) : Dimensionless temperature of the system: T=1/beta. (Default 1.0).
        n_bridge (int) : Number of bridging distributions to place between the quadratic and target 
            distributions. (The total number of distributions sampled will be n_bridge + 2).
        iters_to_waypoint (int) : Restart information is written after every iters_to_waypoint 
            iterations. (Default 10). 
        times_to_setdt (int) : The step sizes (or equivalently time steps) are updated after every 
            n_bridge/times_to_setdt interations. (Default 10).
        run_token (int) : An inDimensionless teger for labelling the restart and output files for this
            calculation. (Default 1).
        dt (float) : Initial time step (or step size). This will be updated algorithmically, but a 
            good starting point saves time. (Default 1.0e-1).
        traj_len (int) : The number of time steps in a single trajectory. (Default 100).
        ntraj_burnin (int) : The number of trajectories of burn-in run per iteration. 
            (Default 100).
        ntraj_sample (int) : The number of trajectories of sampling run per iteration. 
            One data point is collected at the end of each trajectory. (Default 100).
        absxmax (single float or numpy array of floats, with length 1 or length numdim) : During the 
            main calculation, the sampler is restricted to a region x in [-absxmax,absxmax]. 
            (Default: 1.0e2).
        dt_max (float) : maximum step size (time step). (Default: median(absxmax), which is set in 
            module sampling.)
        min_rate (float) : minimum acceptance rate of trajectories. Used for setting step size (time 
            step). (Default: 0.6. The optimal acceptance rate for HMC on a multivariate Gaussian is 
            0.65 http://www.mcmchandbook.net/HandbookChapter5.pdf, section 5.4.4.3).
        max_rate (float) : maximum acceptance rate of trajectories. Used for setting step size (time 
            step). (Default 0.7. The optimal acceptance rate for HMC on a multivariate Gaussian is 
            0.65 http://www.mcmchandbook.net/HandbookChapter5.pdf, section 5.4.4.3).
        gaussianprior_std (single float or numpy array of floats, with length 1 or length numdim) : If
            this is set to a real value then an additional term is applied to (H)MC 
            acceptance/rejection such that the target distribution is proportional to a multivariate 
            Gaussian with this standard deviation for each dimension. (Default: None.)

    Return:
        this_ti: ThermodynamicIntegration class object.
        walker : sampling.NewWalker class object, corresponding to initial_coords after equilibration
            at temperature T.
    """

    restrtfl = "restart_ti_"+str(run_token)+".txt"

    if os.path.isfile("./"+restrtfl): # read restart data from restart file

        x0, ti_int_all, half_k, n_bridge, sampler, walker, llambda, ntraj_burnin, \
            ntraj_sample = read_waypoint(restrtfl, sampler_class, pe_method, force_method)

    else:

        sampler = build_sampler( sampler_class, pe_method, force_method, T = T, \
            dt = dt, traj_len = traj_len, absxmax = absxmax, dt_max = dt_max, \
            min_rate = min_rate, max_rate = max_rate, gaussianprior_std = gaussianprior_std )

        sampling.NewWalker.masses = masses
        sampling.NewWalker.numdim = len(initial_coords)
        walker = sampling.NewWalker(absxmax, sampler.name, pe_method = pe_method, x=initial_coords,\
            beta = 1.0/T)

        x0 = walker.x.copy()
        walker.pe = pe_method(walker.x)
        ti_int_all = []
        llambda = 0.0

        print "Start getting inital spring constants and doing burn in",time.ctime()
        half_k = get_half_k(walker, sampler, x0, 10*ntraj_burnin, 1000)

    print half_k

    this_ti = ThermodynamicIntegration(restrtfl, sampler, x0, ti_int_all, \
        half_k=half_k, Nbridge=n_bridge,ntraj_burnin=ntraj_burnin, \
        ntraj_sample=ntraj_sample, ntimes_set_dt=times_to_setdt, its_to_waypoint=iters_to_waypoint, \
        llambda=llambda)

    return this_ti, walker

def build_sampler( sampler_class, pe_method, force_method, T = 1.0e-4, \
    dt = 1.0e-1, traj_len = 100, absxmax = 1.0e2, dt_max = None, min_rate = 0.6, \
    max_rate = 0.7, gaussianprior_std = None ):
    """Builds a sampling.Sampler class object of type sampler_class. 
    
    Args:
        sampler_class : Sampler class from module sampling. Eg. sampling.Hmc
        pe_method : A method for evaluating the potential energy.
        force_method : A method for evaluating the forces.
        T (float) : Dimensionless temperature of the system: T=1/beta. (Default 1.0).
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
        sampling.Sampler class object of type sampler_class.

    """

    sampler = sampler_class( pe_method, force_method, dt, traj_len, absxmax, \
        dt_max, 1.0/T, min_rate, max_rate, gaussianprior_std )

    return sampler

class ThermodynamicIntegration:
    """This class implements the thermodynamic integration algorithm for estimating the free energy of
    a crystal. See Frenkel, D., & Smit, B. (2001). Understanding molecular simulation: from algorithms to 
    applications (Vol. 1). Elsevier.
    
    Args:
        restrtfl (str) : File name for restart file. File is updated every its_to_waypoint iterations.
        sampler_in : sampling.Sampler class object. Should sample the Boltzmann distribution for the true 
            potential energy surface at temperature T.
        x0 : A numpy array containing the coordinates (parameter values) corresponding
            to a local minimum of the potential energy surface. A quadratic potential with minimum
            at initial_coords will be fitted to the potential energy surface.
        ti_int_all : For a restart, this should be a list of samples gathered from completed steps of the 
            thermodynamic integration algorithm. Each element ti_int_all has the structure 
            [llambda, [ data points ]] where data points are values of 
            (quadratic potential - true potential (shifted to zero minimum value)), sampled accroding to the
            Boltzmann distribution for bridging_pe. (Default [], corresponding a new calculation.)
        half_k (single float or numpy array of floats, with length 1 or length numdim): The quadratic 
            potential is of the form dot_product( (x-x0), (elementwise multiplication of  half_k, (x-x0))).
        Nbridge (int) : Number of bridging distributions to place between the quadratic and target 
            distributions. (The total number of distributions sampled will be n_bridge + 2.) (Default: 100.)
        ntraj_burnin (int) : The number of trajectories of burn-in run per iteration. 
            (Default 100).
        ntraj_sample (int) : The number of trajectories of sampling run per iteration. 
            One data point is collected at the end of each trajectory. (Default 100).
        ntimes_set_dt (int) : The step sizes (or equivalently time steps) are updated after every 
            Nbridge/ntimes_set_dt interations. (Default 10).
        its_to_waypoint (int) : Restart information is written after every its_to_waypoint 
            iterations. (Default 10). 
        llambda : Starting value of llambda (in case of restarts). (Default: 0.0).

    Attributes:
        restrtfl (str) : File name for restart file. File is updated every its_to_waypoint iterations.
        sampler_in : sampling.Sampler class object. Should sample the Boltzmann distribution for the true 
            potential energy surface at temperature T.
        x0 : A numpy array containing the coordinates (parameter values) corresponding
            to a local minimum of the potential energy surface. A quadratic potential with minimum
            at initial_coords will be fitted to the potential energy surface.
        ti_int_all : For a restart, this should be a list of samples gathered from completed steps of the 
            thermodynamic integration algorithm. Each element ti_int_all has the structure 
            [llambda, [ data points ]] where data points are values of 
            (quadratic potential - true potential (shifted to zero minimum value)), sampled accroding to the
            Boltzmann distribution for bridging_pe. (Default: [], corresponding a new calculation.)
        step_start (int) : Initial iteration counter. (Default: len(ti_int_all) at start of calculation.) 
        half_k (single float or numpy array of floats, with length 1 or length numdim): The quadratic 
            potential is of the form dot_product( (x-x0), (elementwise multiplication of  half_k, (x-x0))).
        Nbridge (int) : Number of bridging distributions to place between the quadratic and target 
            distributions. (The total number of distributions sampled will be n_bridge + 2.) (Default: 100.)
        ntraj_burnin (int) : The number of trajectories of burn-in run per iteration. 
            (Default 100).
        ntraj_sample (int) : The number of trajectories of sampling run per iteration. 
            One data point is collected at the end of each trajectory. (Default 100).
        ntimes_set_dt (int) : The step sizes (or equivalently time steps) are updated after every 
            Nbridge/ntimes_set_dt interations. (Default 10).
        sampler_use : Same as sampler_in except this sampling.Sampler class object has 
            pe_method = self.bridging_pe and force_method = self.bridging_force, which interpolates between the 
            true and quadratic potentials, controlled by llambda.
        pe0 (float) : sampler_in.pe_method(x0)
        llamda (float) : mixing parameter that controls interpolation between the true and approximated 
            quadratic potentials.
        its_to_waypoint (int) : Restart information is written after every its_to_waypoint iterations. 
            (Default 10). 
    """


    def __init__(self,restrtfl, sampler_in, x0, ti_int_all = [], half_k=1.0, Nbridge=100, \
        ntraj_burnin = 100, ntraj_sample = 100, ntimes_set_dt = 10, \
        its_to_waypoint=10, llambda = 0.0 ):

        self.restrtfl = restrtfl
        self.sampler_in = sampler_in
        self.x0 = x0
        self.ti_int_all = ti_int_all
        self.step_start = len(ti_int_all)
        self.half_k = half_k
        self.Nbridge = Nbridge
        self.ntraj_burnin = ntraj_burnin
        self.ntraj_sample = ntraj_sample
        self.ntimes_set_dt = ntimes_set_dt
        self.sampler_use = copy.deepcopy(self.sampler_in)
        self.sampler_use.pe_method = self.bridging_pe
        self.sampler_use.force_method = self.bridging_force
        self.pe0 = sampler_in.pe_method(self.x0)
        self.llambda = llambda
        self.its_to_waypoint = its_to_waypoint

    def bridging_pe(self,x):
        """Bridging potential energy surface which interpolates between the true and quadratic approximation
        potential energy surfaces. Frenkel, Smit, "Understanding Molecular Simulation", (10.2.1).

        Args:
            x : A numpy array containing the coordinates (parameter values) at which to evaluate the bridging
                potential.

        Return:
            pe (float) : Value of bridging potential at x.
        """

        # Frenkel, Smit, "Understanding Molecular Simulation", (10.2.1)
        pe = self.pe0 + (1.0-self.llambda)*(self.sampler_in.pe_method(x)-self.pe0) + \
            self.llambda*np.inner((x-self.x0), np.multiply(self.half_k, (x-self.x0)))
        return pe

    def bridging_force(self,x):
        """Negative derivative of the bridging potential with respect to x. The bridging potential 
        interpolates between the true and quadratic approximation potential energy surfaces. 

        Args:
            x : A numpy array containing the coordinates (parameter values) at which to evaluate the 
                negative derivative of the bridging potential.

        Return:
            f (numpy array) : Negative derivative of the bridging potential, evaluated at x.
        """

        # Negative differential of bridging_pe wrt x
#        f = self.llambda*self.sampler_in.force_method(x) - \
#            2.0 * (1.0-self.llambda) * np.multiply( self.half_k, (x-self.x0))
        f = (1.0-self.llambda)*self.sampler_in.force_method(x) - \
            2.0 * self.llambda * np.multiply( self.half_k, (x-self.x0))
        return f

    def ti_main(self, walker_in):
        """Main thermodynamic integration routine.
        
        Args:
            walker_in : sampling.NewWalker class object. Ideally walker_in.x should be an equilibrium 
                sample from the Boltzmann distribution for sampler_in.pe_method.
        """

        walker = sampling.NewWalker(self.sampler_in.absxmax, self.sampler_in.name, \
            pe_method = self.sampler_use.pe_method, x=walker_in.x, p=walker_in.p)

        if (len(range(self.step_start,self.Nbridge+2))>0):  # skip this if we are restarting after the
                                                            # last sampling iteration 
            if (self.step_start==0): # Do extra burn-in for first iteration. We may well be starting 
                                     # from x0, as this is set by build_ti.
                this_step_setter = sampling.UpdateDt(self.sampler_use)
                this_step_setter.set([walker], '0. Set dt rough.', adjust_step_factor = 0.1)
                traj_burnin = sampling.Traj(self.sampler_use, 5*self.ntraj_burnin) # do 5x burn-in
                walker = traj_burnin.run_serial(walker)
                this_step_setter.set([walker], '0. Done 5x burn-in. Set dt rough.', \
                    adjust_step_factor = 0.1)
                traj_burnin = sampling.Traj(self.sampler_use, 10*self.ntraj_burnin) # do 10x burn-in
                walker = traj_burnin.run_serial(walker)
                this_step_setter.set([walker], '0. Done further 10x burn-in. Set dt rough.', \
                    adjust_step_factor = 0.1)
                this_step_setter.set([walker], '0. Set dt accurate.', adjust_step_factor = 0.9)
            else: # check step size
                this_step_setter = sampling.UpdateDt(self.sampler_use)
                this_step_setter.set([walker], '0. Set dt rough.', adjust_step_factor = 0.1)
                this_step_setter.set([walker], '0. Set dt accurate.', adjust_step_factor = 0.9)


        print "Start thermal integration ", time.ctime()
        for step in xrange(self.step_start,self.Nbridge+2):

            if (step%self.its_to_waypoint==0):
                self.write_waypoint(walker, step)

            self.llambda = step*1.0/(self.Nbridge+1.0)
            walker.pe = self.sampler_use.pe_method(walker.x) # update pe since pe function has changed

            traj_burnin = sampling.Traj(self.sampler_use, self.ntraj_burnin) # do burn-in
            walker = traj_burnin.run_serial(walker)

            traj_sample = sampling.Traj(self.sampler_use, self.ntraj_sample) # do main sampling

            if (step%(max(int(float(self.Nbridge)/self.ntimes_set_dt),1))==0 and step!=0):
                this_step_setter.set([walker], str(step)+'. Set dt ', adjust_step_factor = 0.9)

            self.ti_int_all.append([self.llambda,[]])
            for t in xrange(traj_sample.num_traj):
                walker, taccept = traj_sample.run(walker)
                x = walker.x
#                self.ti_int_all[-1][-1].append( (self.sampler_in.pe_method(x) - self.pe0) - \
#                    np.inner(x,np.multiply(self.half_k,x)) )
                self.ti_int_all[-1][-1].append( np.inner((x-self.x0),np.multiply(self.half_k,(x-self.x0))) - \
                    (self.sampler_in.pe_method(x) - self.pe0))
            print step, self.llambda,np.mean(self.ti_int_all[-1][-1]),np.std(self.ti_int_all[-1][-1])

        if (self.step_start<self.Nbridge+1):
            self.write_waypoint(walker, step)

        Es = [ np.mean(x[-1]) for x in self.ti_int_all ]
        fe = integrate_dF(Es, 1.0/(self.Nbridge+1.0), walker.numdim, \
            self.sampler_use.beta, self.pe0, self.half_k, self.sampler_use.absxmax)

        print "Final Free Energy: ",fe

    def write_waypoint(self,walker,step):
        """Writes the waypoint file to self.restrtfl"""

        with open(self.restrtfl,"w") as rout:
            ti_params = {}
            ti_params["iteration"] = step
            ti_params["restrtfl"] = self.restrtfl
            ti_params["x0"] = self.x0
            if type(ti_params["x0"]) is np.ndarray:
                ti_params["x0"] = ti_params["x0"].tolist()
            ti_params["ti_int_all"] = self.ti_int_all
            ti_params["half_k"] = self.half_k
            if type(ti_params["half_k"]) is np.ndarray:
                ti_params["half_k"] = ti_params["half_k"].tolist()
            ti_params["Nbridge"] = self.Nbridge
            ti_params["llambda"] = self.llambda
            ti_params["ntraj_burnin"] = self.ntraj_burnin
            ti_params["ntraj_sample"] = self.ntraj_sample
            ti_params["ntimes_set_dt"] = self.ntimes_set_dt

            ti_params["masses"] = sampling.NewWalker.masses
            if type(ti_params["masses"]) is np.ndarray:
                ti_params["masses"] = ti_params["masses"].tolist()
            ti_params["numdim"] = sampling.NewWalker.numdim

            l = "ti_params "+json.dumps(ti_params)+"\n"
            rout.write(l)

            sampler = self.sampler_in # read_waypoint will build ThermodynamicIntegration.sampler_in
            int_params = {}
            int_params["dt"] = self.sampler_use.dt # save most recent dt
            int_params["traj_len"] = sampler.traj_len
            int_params["absxmax"] = sampler.absxmax
            int_params["beta"] = sampler.beta
            int_params["dt_max"] = sampler.dt_max
            int_params["name"] = sampler.name
            int_params["min_rate"] = sampler.min_rate
            int_params["max_rate"] = sampler.max_rate
            int_params["gaussianprior_std"] = sampler.gaussianprior_std

            if type(int_params["absxmax"]) is np.ndarray:
                int_params["absxmax"] = int_params["absxmax"].tolist()

            l = "int_params "+json.dumps(int_params)+"\n"
            rout.write(l)

            sampling.write_walker(walker, rout,'config ')

def read_waypoint(restrtfl, sampler_class, pe_method,force_method):
    """Reads a waypoint from restrtfl

    Args:
        restrtfl (str) : file from which to read waypoint
        sampler_class : Sampler class from module sampling. Eg. sampling.Hmc .
        pe_method : A method for evaluating the true potential energy. (Intended to build sampler_in for 
            ThermodynamicIntegration class.)
        force_method : A method for evaluating the true forces. (Intended to build sampler_in for 
            ThermodynamicIntegration class.

    Return:
        x0 : A numpy array containing the coordinates (parameter values) corresponding
            to a local minimum of the potential energy surface. A quadratic potential with minimum
            at initial_coords will be fitted to the potential energy surface.
        ti_int_all : A list of samples gathered from completed steps of the thermodynamic integration
            algorithm. Each element ti_int_all has the structure [llambda, [ data points ]] where data
            points are values of (quadratic potential - true potential (shifted to zero minimum value)),
            sampled accroding to the Boltzmann distribution for bridging_pe.
        half_k (single float or numpy array of floats, with length 1 or length numdim): The quadratic 
            potential is of the form
            dot_product( (x-x0), (elementwise multiplication ofhalf_k, (x-x0))).
        Nbridge (int) : Number of bridging distributions to place between the quadratic and target 
            distributions. (The total number of distributions sampled will be n_bridge + 2.)
        sampler : sampling.Sampler class object corresponding to the true potential energy surface. Intended
            to be used as sampler_in in ThermodynamicIntegration class.
        walker : sampling.NewWalker class object, at time of writing waypoint file.
        llambda (float) : llambda parameter at time of writing waypoint file.
        ntraj_burnin (int) : The number of trajectories of burn-in run per iteration. 
        ntraj_sample (int) : The number of trajectories of sampling run per iteration. 
            One data point is collected at the end of each trajectory. 
    """

    with open(restrtfl) as rin:
        for line in rin:
            label = line.split()[0]
            if (label == "ti_params"):
                ti_params = json.loads( line[len(label):].strip() )

                x0 = np.asarray(ti_params["x0"])
                ti_int_all = ti_params["ti_int_all"] 
                half_k = np.asarray(ti_params["half_k"])
                Nbridge = ti_params["Nbridge"]
                llambda = ti_params["llambda"]
                ntraj_burnin = ti_params["ntraj_burnin"] 
                ntraj_sample = ti_params["ntraj_sample"] 

                sampling.NewWalker.masses = np.asarray(ti_params["masses"])
                sampling.NewWalker.numdim = ti_params["numdim"]

            elif (label == "int_params"):
                int_params = json.loads( line[len(label):].strip() )
                del int_params["name"]
                int_params["absxmax"] = np.asarray(int_params["absxmax"])
                sampler = sampler_class(pe_method,force_method,**int_params)

            elif (label == "config"):
                wd = json.loads( line[len(label):].strip() )
                wd["x"] = np.asarray(wd["x"])
                wd["p"] = np.asarray(wd["p"])
                walker = sampling.NewWalker(absxmax = None, sampler_name = None, **wd)

    return x0, ti_int_all, half_k, Nbridge, sampler, walker, llambda, ntraj_burnin, \
        ntraj_sample

def get_half_k(walker, sampler, x0, ntraj_burnin = 100, nsamples = 1000 ):
    """ Following Frenkel, Smit, "Understanding Molecular Simulation" p. 245, this routine aims
    to set the variance of the coordiates under MD on the true potential energy surface, equal 
    with the variance for MD on the springs. This routine takes the following steps
    1. Set rough value of dt.
    2. Do 20% burn in at dimensionless temperature T = 1.0/sampler.beta
    3. Set rough value of dt.
    4. Do 80% bin in at dimensionless temperature T = 1.0/sampler.beta
    5. Set accurate dt for true potential energy surface.
    6. Collect samples from potential energy surface.
    7. Use the variance of these samples in each dimension to infer the approximate value of
       half_k.

    Args:
        walker : A sampling.NewWalker class object.
        sampler : A sampling.Sampler class object that samples from the true potential energy surface.
        x0 : A numpy array containing the coordinates (parameter values) corresponding
            to a local minimum of the potential energy surface. In thermodynamic integration, a quadratic 
            potential with minimum at initial_coords is fitted to the potential energy surface.
        ntraj_burnin (int) : The number of trajectories of burn-in. (Default 100).
        ntraj_sample (int) : The number of trajectories of sampling run. One data point is collected at the 
            end of each trajectory. (Default 1000).

        Return:
            half_k (numpy array of floats with length walker.numdim): 1d array of floats specifying 
                the quadratic approximation to the true potential.
     """

    import sys

    walker_use = sampling.NewWalker(None,None,None, x = walker.x, p = walker.p, pe = walker.pe, \
        ke = walker.ke)

    # Turn off absxmax for fitting quadratic. This is reset at the end.
    save_absxmax = copy.deepcopy(sampler.absxmax)
    sampler.absxmax = sys.float_info.max
    
    # 1. Set initial dt
    this_step_setter = sampling.UpdateDt(sampler)
    this_step_setter.set([walker_use], 'Getting spring constants 0. ', adjust_step_factor = 0.1)

    # 2. Do 20% of burn in
    traj_burnin = sampling.Traj(sampler, int(np.ceil(ntraj_burnin*0.2)))
    walker_use = traj_burnin.run_serial(walker_use)

    # 3. Improve dt
    this_step_setter.set([walker_use], 'Getting spring constants 1. ', adjust_step_factor = 0.1)

    # 4. Do 80% of burn in
    traj_burnin = sampling.Traj(sampler, int(np.ceil(ntraj_burnin*0.8)))
    walker_use = traj_burnin.run_serial(walker_use)

    # 5. Set final dt
    this_step_setter.set([walker_use], 'Getting spring constants 2. ', adjust_step_factor = 0.1)
    this_step_setter.set([walker_use], 'Getting spring constants 3. ', adjust_step_factor = 0.9)
    
    # 6. Set half_k similar to Frenkel, Smit, "Understanding Molecular Simulation".
    #    This approach trys to set <(x-x0)^2> in the Einstein crystal to be equal to
    #    that for the true potential. An estimate of <(x-x0)^2> is therefore required for each 
    #    dimension.
    rsq = np.zeros(len(walker_use.x)) # This will hold <(x-x0)^2> for each dimension
    traj_sample = sampling.Traj(sampler, 1) # A sample will be drawn after each of nsamples 
                                            # trajectories
    for t in xrange(nsamples):
        walker_use, taccept = traj_sample.run(walker_use)
        rsq += (walker_use.x - x0)**2
    rsq /= nsamples

    half_k = (1.0/sampler.beta)*0.5/rsq # Adapted from Frenkel, Smit, "Understanding Molecular 
        # Simulation" Eq. (10.2.4). There dimensions are grouped into sets of three, here they are not.

    # reset absxmax to saved value
    sampler.absxmax = save_absxmax

    return half_k

def integrate_dF(Es,dlambda,dimens,beta,pe0, half_k, absxmax=None):
    """Calculate the absolute free energy of the system by numerical integration with Simpson's Rule.
    
    Args:
        Es (numpy array of floats): <(quadratic potential - true potential (shifted to zero minimum value))>
            for each value of llamda.
        dlambda (float) : llambda step size.
        beta (float) : 1.0/T (T is dimensionless temperature.)
        pe0 (float) : true potential energy function evaluated at x0, which was location of minimum of
            quadratic approximation to true potential energy function.
        half_k (numpy array of floats): 1d array of floats specifying the quadratic approximation to 
            the true potential.
        absxmax (None or float or numpy array of floats with length equal to that of half_k): If 
            absxmax is not None, then the integral for the quadratic potential is constrained to the
            region -absxmax_i < x_i < +absxmax_i for all i. (If absxmax is a float, then absxmax_i is 
            equal to the value of that float for all i.)

    Return:
        F (float) : Free energy of crystal.
    """

    from scipy.integrate import simps
    from scipy.special import erf

    T = 1.0/beta
    dF = - simps(Es,dx=dlambda) # integrate from 0 to 1, but dF is integral from 1 to 0
    F_Einstein = pe0 - 0.5*T*dimens*np.log(T*np.pi) + 0.5*T*np.sum(np.log(half_k))
    if (absxmax is not None):
        # account for absxmax bounds on integration
        # See Gradshteyn, Ryzhik. "Table of Integrals, Series and Products", 7th Ed. Eq. 3.321.2
        q = np.sqrt(half_k/T)
        F_Einstein += -T * np.sum( np.log(erf(np.multiply(absxmax,q))) ) 
        # This last line works if absxmax is a float or a numpy array of the same length as half_k

    F = F_Einstein + dF

#    sum_log_keff = dimens*np.log(T*np.pi) + 2.0*(F - pe0)/T
#    sum_log_sigma = -0.5*(sum_log_keff + dimens*np.log(2.0))

    return F_Einstein + dF #, sum_log_sigma
