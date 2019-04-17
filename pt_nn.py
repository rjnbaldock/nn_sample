#!/usr/bin/env python
"""This script applies parallel tempering (replica exchange (RE) molecular dynamics (MD) with MD replaced by
Hamiltonian Monte Carlo (HMC)."""

import sampling
import pt
import nn_pe_force
import numpy as np
import time

def read_runparameters():
    """ This routine reads a json object from the file pt_nn.config into a dictionary of parameters for
    the calculation. After loading the json object, pt_nn.config is overwritten with the dictionary of
    parameters used in the calculation. If pt_nn.config does not exist then default parameters are used.
    These default parameters are also used to fill missing keys from the input dictionary.
    
    Default runtime parameters:

        "runtoken" : 1           # (int) An integer used for labelling the files for this run
        "nproc" : 16             # (int) Number of processors to use.
        "nT" : 16                # (int) Number of temperatures to use.
        "Tmin" : 1.0e-2          # (float) Lowest temperature in ladder of temperatures.
        "Tmax" : 1.0e0           # (float) Maximum temperature in ladder of temperatures. 
        "absxmaxfac" : 5.0e1     # (float) The parameters of the NN will be constrained to be with a range 
                                 # "[-absxmaxabsxmax]" where absxmax = absxmaxfac/sqrt(k_i), with k_i the 
                                 # inward degree of the neuron to which parameter i transmits values.
        "gprior_std" : None      # (None or float) If this is set to a real value then an additional term is applied to (H)MC
                                 # acceptance/rejection such that the target distribution is proportional to
                                 # multivariate Gaussian with this standard deviation for each dimension. 
        "dt_initial" : 1.0e-1    # (float) Initial time step (or step size). This will be updated algorithmically, 
                                 # but a good starting point saves time.
        "num_traj" : 10          # (int) The number of trajectories run per iteration.
        "traj_len" : 100         # (int) The number of time steps in a single trajectory.
        "maxiter" : 10000        # (int) Max number of iterations to run.
        "iterstoswap" : 1        # (int) Configuration swaps between neighbouring temperatures are attempted every 
                                 # iterstoswap iterations.
        "iterstowaypoint" : 1    # (int) Restart information is written after every iterstowaypoint iterations.
        "iterstosetdt" : 25      # (int) The step sizes (or equivalently time steps) are updated after every 
                                 # iterstosetdt interations.
        "iterstowritestate" : 1  # (int) The latest potential energy values and coordinates are written out after 
                                 # every iterstowritestate iterations.
        "n_h_layers" : 3         # (int) The number of hidden layers.
        "nodes_per_h_layer" : 40 # (int) The number of nodes in each hidden layer.
        "image_sidel_use" : 16   # (int) Images will be transformed to have this many pixels along the side.
        "datapoints_per_class" : 50 # (int) Number of stratified samples to draw per class.

    Return:
        run_parameters_out : dictionary of runtime parameters
    """

    import json

    default_run_parameters = {
    "runtoken" : 1,
    "nproc" : 16,
    "nT" : 16,
    "Tmin" : 1.0e-2,
    "Tmax" : 1.0e0,
    "absxmaxfac" : 5.0e1,
    "gprior_std" : None,
    "dt_initial" : 1.0e-1,
    "num_traj" : 10,
    "traj_len" : 100,
    "maxiter" : 10000,
    "iterstoswap" : 1,
    "iterstowaypoint" : 1,
    "iterstosetdt" : 25,
    "iterstowritestate" : 1,
    "n_h_layers" : 3,
    "nodes_per_h_layer" : 40,
    "image_sidel_use" : 16,
    "datapoints_per_class" : 50,
    }

    run_parameters_out = {}
    try: # if pt_nn.config doesn't exist, use the default parameters
        with open('pt_nn.config', 'r') as f:
            rp = json.load(f)
            for key in rp.keys(): # Load only those keys already specified in default_run_parameters
                if key in default_run_parameters.keys():
                    run_parameters_out[key] = rp[key]
            for key in default_run_parameters.keys():
                if key not in run_parameters_out.keys():
                    run_parameters_out[key] = default_run_parameters[key]
            print "Read runtime parameters from pt_nn.config"
    except:
        run_parameters_out = default_run_parameters
        print "Using default runtime parameters"

    with open('pt_nn.config','w') as f:
        f.write(json.dumps(run_parameters_out))
        print "Written runtime parameters to pt_nn.config"

    # specify file for indices to read or write the indices of stratified data samples
    run_parameters_out["datafile"] = "data"+str(run_parameters_out["datapoints_per_class"])+".txt"

    return run_parameters_out

def calc_numdim(image_sidel_use,n_h_layers,nodes_per_h_layer,n_classes ):
    """This function calculates the total number of parameters (weights and biases) that will 
    feature in the specified NN.
    
    Args:
        image_sidel_use (int) : Images will be transformed to have this many pixels along the side.
        n_h_layers (int) : the number of hidden layers
        nodes_per_h_layer (int) : the number of nodes in each hidden layer
        n_classes (int) : the number of nodes in the final (output) layer

    Return:
        numdim (int) : the number of parameters that will feature in the NN.
    
    """

    n_nodes_lowerlayer, n_nodes_thislayer = image_sidel_use**2, nodes_per_h_layer # first hidden layer
    numdim = (1 + n_nodes_lowerlayer) * n_nodes_thislayer
    n_nodes_lowerlayer, n_nodes_thislayer = nodes_per_h_layer, nodes_per_h_layer # other hidden layers
    numdim += ( n_h_layers - 1 ) * (1 + n_nodes_lowerlayer) * n_nodes_thislayer
    n_nodes_lowerlayer, n_nodes_thislayer = nodes_per_h_layer, n_classes # output layer
    numdim += (1 + n_nodes_lowerlayer) * n_nodes_thislayer

    return numdim

run_params = read_runparameters()

these_masses = 1.0 # all parameters will have the same effective timestep.

# Set limits on parameter values for random (uniform) initialisation.
# Use standard [-1/sqrt(fan in) , 1/sqrt(fan in)]
intial_absxmax = 1.0/np.sqrt(nn_pe_force.calc_fan_in(run_params["image_sidel_use"]**2, \
    run_params["n_h_layers"],run_params["nodes_per_h_layer"],10))
nd = calc_numdim(run_params["image_sidel_use"], run_params["n_h_layers"], run_params["nodes_per_h_layer"], \
    10) # calculate total number of parameters for NN

# Initialise object to calculate pe and force values for NN.
# Uses data indicies from MNIST specified in run_params["datafile"] to get reproducible cost function.
# Data is stratified if no file is specified.
# run_params["datapoints_per_class"] data points, per class.
nnpef = nn_pe_force.build_repeatable_NNPeForces(indfl = run_params["datafile"], \
    image_sidel_use=run_params["image_sidel_use"], n_h_layers=run_params["n_h_layers"], \
    nodes_per_h_layer=run_params["nodes_per_h_layer"], \
    datapoints_per_class=run_params["datapoints_per_class"])

# Initialise random walkers or read restart file.
# Initialise parallel tempering object.
thispt = pt.build_pt(sampling.Hmc, pe_method = nnpef.pe, force_method = nnpef.forces, numdim = nd, \
    masses = these_masses, nT = run_params["nT"], nproc = run_params["nproc"], Tmin = run_params["Tmin"], \
    Tmax = run_params["Tmax"], max_iteration = run_params["maxiter"], \
    iters_to_swap = run_params["iterstoswap"], iters_to_waypoint = run_params["iterstowaypoint"], \
    iters_to_setdt = run_params["iterstosetdt"], iters_to_writestate = run_params["iterstowritestate"], \
    run_token = run_params["runtoken"], dt = run_params["dt_initial"], traj_len = run_params["traj_len"], \
    num_traj = run_params["num_traj"], absxmax = run_params["absxmaxfac"]*intial_absxmax, \
    initial_rand_bounds = intial_absxmax, gaussianprior_std = run_params["gprior_std"] )

# Update boundaries on parameters, for softer prior.
for this_traj in thispt.pt_trajs:
    this_traj.sampler.absxmax = run_params["absxmaxfac"]*intial_absxmax
    this_traj.sampler.dt_max = np.median(this_traj.sampler.absxmax)

# Run parallel tempering
print 'About to do pt. ',time.ctime()
thispt.ptmain()
print 'Done pt. ',time.ctime()
