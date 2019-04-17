#!/usr/bin/env python
"""This script calculates the free energy of a neural network, assuming the posterior of the network is localised
around a particular local minimum, specified in the file run_params["coords_file"]. The effective value of the log of the 
determinant of the Hessian at that minimum is also reported."""

import sampling
import ti
import nn_pe_force
import numpy as np
import time
import json

def read_runparameters():
    """ This routine reads a json object from the file ti_nn.config into a dictionary of parameters for
    the calculation. After loading the json object, ti_nn.config is overwritten with the dictionary of
    parameters used in the calculation. If ti_nn.config does not exist then default parameters are used.
    These default parameters are also used to fill missing keys from the input dictionary.
    
    Default runtime parameters:

        runtoken : 1        # (int) An integer used for labelling the files for this run
        coords_file : "localmin_data50_h3_flin.txt"  # (str) File containing sampling.NewWalker class object saved as 
                            # dict using sampling.write_walker . The coordinates should correspond to a local
                            # minimum of the cost function. These coordinates were found by minimising the 
                            # lowest energy parameters discovered in a parallel tempering run. The gradient 
                            # for every individual dimension is smaller that 1.0e-4. This potential energy 
                            # surface corresponds to a network with 3 hidden layers, 256 (=16**2) input 
                            # neurons, 3 hidden layers each containing 40 logistic neurons and 10 linear
                            # output neurons, terminated with a softmax. All hidden and output neurons use a
                            # bias. The potential is additionally specified by the data which is the 
                            # stratified sample from MNIST, stored in data50.txt, and comprising 50 data 
                            # points for each of the 10 digit classes.
        T : 1.0e-4          # (float) Dimensionless temperature T = 1.0/beta
        absxmaxfac : 5.0e1  # (float) The parameters of the NN will be constrained to be with a range 
                            # '[-absxmax,absxmax]' where absxmax = absxmaxfac/sqrt(k_i), with k_i the inward 
                            # degree of the neuron to which parameter i transmits values.
        gprior_std : None   # (None or float) If this is set to a real value then an additional term is applied to (H)MC 
                            # acceptance/rejection such that the target distribution is proportional to a 
                            # multivariate Gaussian with this standard deviation for each dimension. 
        dt_initial : 1.0e-1 # (float) Initial time step (or step size). This will be updated algorithmically, but a 
                            # good starting point saves time.

        ntraj_burnin : 100  # (int) The number of burn in trajectories to run for each bridging distribution
        ntraj_sample : 100  # (int) The number of sampling trajectories to run for each bridging distribution 
        traj_len : 100      # (int) The number of time steps to use for each trajectory

        nbridge : 100       # (int) Number of bridging distributions to use. Including sampling the distribution 
                            # corresponding to the true potential, and the quadratic approximation 
                            # potential, the nbridge+2 distributions are sampled.
        ntimes_set_dt : 10  # (int) dt is updated after sampling every nbridge/ntimes_set_dt distributions.
        iterstowaypoint : 10 # (int) Restart information is written after sampling every iterstowaypoint distributions.

        n_h_layers : 3      # (int) The number of hidden layers.
        nodes_per_h_layer : 40 # (int) The number of nodes in each hidden layer.
        image_sidel_use : 16 # (int) Images will be transformed to have this many pixels along the side.
        datapoints_per_class : 50 # (int) Number of stratified samples to draw per class.

    Return:
        run_parameters_out : dictionary of runtime parameters
    """

    import json

    # Set run parameters
    default_run_parameters = {
        "runtoken" : 1,
        "coords_file" : "localmin_data50_h3_flin.txt",
        "T" : 1.0e-4,
        "absxmaxfac" : 1.0e5,
        "gprior_std" : None,
        "dt_initial" : 1.0e-1,
        "ntraj_burnin" : 100,
        "ntraj_sample" : 100,
        "traj_len" : 100,
        "nbridge" : 100,
        "ntimes_set_dt" : 10,
        "iterstowaypoint" : 10,
        "n_h_layers" : 3,
        "nodes_per_h_layer" : 40,
        "image_sidel_use" : 16,
        "datapoints_per_class" : 50
        }

    run_parameters_out = {}
    try: # if ti_nn.config doesn't exist, use the default parameters
        with open('ti_nn.config', 'r') as f:
            rp = json.load(f)
            for key in rp.keys(): # Load only those keys already specified in default_run_parameters
                if key in default_run_parameters.keys():
                    run_parameters_out[key] = rp[key]
            for key in default_run_parameters.keys():
                if key not in run_parameters_out.keys():
                    run_parameters_out[key] = default_run_parameters[key]
            print "Read runtime parameters from ti_nn.config"
    except:
        run_parameters_out = default_run_parameters
        print "Using default runtime parameters"

    with open('ti_nn.config','w') as f:
        f.write(json.dumps(run_parameters_out))
        print "Written runtime parameters to ti_nn.config"

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

def read_coords(cfile):
    """Routine to read sampling.NewWalker class object from file cfile. The routine anticipates a json dict on
    the first and only line of cfile, with no leading characters.
    
    Args: 
        cfile (str) : name of file from which to read sampling.NewWalker class object.

    Return:
        walker : sampling.NewWalker class object, read from cfile.
    """
    with open(cfile) as cin:
        lines=cin.readlines()
        line=lines[0]
        wd = json.loads( line.strip() )
        wd["x"] = np.asarray(wd["x"])
        wd["p"] = np.asarray(wd["p"])
        walker = sampling.NewWalker(absxmax = None, sampler_name = None, **wd)
    return walker

run_params = read_runparameters() # get run parameters from file

these_masses = 1.0 # all parameters will have the same effective timestep.

# Parameter values are subject to a (uniform) prior, restricting them to a region
# [-run_params["absxmaxfac"]/sqrt(fan in) , +run_params["absxmaxfac"]/sqrt(fan in)]
absxmax = run_params["absxmaxfac"]/np.sqrt(nn_pe_force.calc_fan_in(run_params["image_sidel_use"]**2, \
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

init_coords = read_coords(run_params["coords_file"]) # Read location of local minimum for cost function.

# Initialise ti.ThermodynamicIntegration class object and equilibrate walker for Boltzmann distribution on true 
# pe surface (given by nnpef.pe) at temperature T.
this_ti, walker = ti.build_ti(sampling.Hmc, pe_method = nnpef.pe, force_method = nnpef.forces, \
    initial_coords = init_coords.x, masses = these_masses, T = run_params["T"], \
    n_bridge = run_params["nbridge"], iters_to_waypoint = run_params["iterstowaypoint"], \
    times_to_setdt = run_params["ntimes_set_dt"], run_token = run_params["runtoken"], \
    dt = run_params["dt_initial"], traj_len = run_params["traj_len"], \
    ntraj_burnin = run_params["ntraj_burnin"], ntraj_sample = run_params["ntraj_sample"], \
    absxmax = absxmax, gaussianprior_std = run_params["gprior_std"])

# Run thermodynamic integration. 
print 'About to do ti. ',time.ctime()
this_ti.ti_main(walker)
print 'Done ti. ',time.ctime()
