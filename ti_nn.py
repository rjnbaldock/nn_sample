#!/usr/bin/env python
"""This script calculates the free energy of a neural network, assuming the posterior of the network is localised
around a particular local minimum, specified in the file COORDS_FILE. The effective value of the log of the 
determinant of the Hessian at that minimum is also reported."""

import sampling
import ti
import nn_pe_force
import numpy as np
import time
import json

# Set run parameters
RUNTOKEN = 1        # An integer used for labelling the files for this run
COORDS_FILE = "localmin_data50_h3.txt"  # File containing sampling.NewWalker class object saved as dict using 
                    # sampling.write_walker . The coordinates should correspond to a local
                    # minimum of the cost function. These coordinates were found by minimising the lowest energy
                    # parameters discovered in a parallel tempering run. The gradient for every individual 
                    # dimension is smaller that 1.0e-4.
T = 1.0e-4          # Dimensionless temperature T = 1.0/beta
ABSXMAXFAC = 1.0e5  # The parameters of the NN will be constrained to be with a range 
                    # '[-absxmax,absxmax]' where absxmax = ABSXMAXFAC/sqrt(k_i), with k_i the inward 
                    # degree of the neuron to which parameter i transmits values.
GPRIOR_STD = None   # If this is set to a real value then an additional term is applied to (H)MC 
                    # acceptance/rejection such that the target distribution is proportional to a 
                    # multivariate Gaussian with this standard deviation for each dimension. 
DT_INITIAL = 1.0e-1 # Initial time step (or step size). This will be updated algorithmically, but a 
                    # good starting point saves time.

NTRAJ_BURNIN = 100  # The number of burn in trajectories to run for each bridging distribution
NTRAJ_SAMPLE = 100  # The number of sampling trajectories to run for each bridging distribution 
TRAJ_LEN = 100      # The number of time steps to use for each trajectory

NBRIDGE = 100       # Number of bridging distributions to use. Including sampling the distribution corresponding 
                    # to the true potential, and the quadratic approximation potential, the NBRIDGE+2 
                    # distributions are sampled.
NTIMES_SET_DT = 10  # dt is updated after sampling every NBRIDGE/NTIMES_SET_DT distributions.
ITERSTOWAYPOINT = 10 # Restart information is written after sampling every ITERSTOWAYPOINT distributions.

N_H_LAYERS = 3      # The number of hidden layers.
NODES_PER_H_LAYER = 40 # The number of nodes in each hidden layer.
IMAGE_SIDEL_USE = 16 # Images will be transformed to have this many pixels along the side.
DATAPOINTS_PER_CLASS = 50 # Number of stratified samples to draw per class.
DATAFILE = "data50.txt" # Name of file for storing or recovering the indicies of data points.
N_CLASSES = 10      # The number of nodes in the final (output) layer.

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

these_masses = 1.0 # all parameters will have the same effective timestep.

# Parameter values are subject to a (uniform) prior, restricting them to a region
# [-ABSXMAXFAC/sqrt(fan in) , +ABSXMAXFAC/sqrt(fan in)]
absxmax = ABSXMAXFAC/np.sqrt(nn_pe_force.calc_fan_in(IMAGE_SIDEL_USE**2,N_H_LAYERS,NODES_PER_H_LAYER,N_CLASSES))
nd = calc_numdim(IMAGE_SIDEL_USE, N_H_LAYERS, NODES_PER_H_LAYER, N_CLASSES) # calculate total number of parameters for NN

# Initialise object to calculate pe and force values for NN.
# Uses data indicies from MNIST specified in DATAFILE to get reproducible cost function.
# Data is stratified if no file is specified.
# DATAPOINTS_PER_CLASS data points, per class.
nnpef = nn_pe_force.build_repeatable_NNPeForces(indfl = DATAFILE,image_sidel_use=IMAGE_SIDEL_USE, \
    n_h_layers=N_H_LAYERS, nodes_per_h_layer=NODES_PER_H_LAYER, datapoints_per_class=DATAPOINTS_PER_CLASS)

init_coords = read_coords(COORDS_FILE) # Read location of local minimum for cost function.

# Initialise ti.ThermodynamicIntegration class object and equilibrate walker for Boltzmann distribution on true 
# pe surface (given by nnpef.pe) at temperature T.
this_ti, walker = ti.build_ti(sampling.Hmc, pe_method = nnpef.pe, force_method = nnpef.forces, \
    initial_coords = init_coords.x, masses = these_masses, T = T, n_bridge = NBRIDGE, \
    iters_to_waypoint = ITERSTOWAYPOINT, times_to_setdt = NTIMES_SET_DT, run_token = RUNTOKEN, \
    dt = DT_INITIAL, traj_len = TRAJ_LEN, ntraj_burnin = NTRAJ_BURNIN, ntraj_sample = NTRAJ_SAMPLE, \
    absxmax = absxmax, gaussianprior_std = GPRIOR_STD)

# Run thermodynamic integration. 
print 'About to do ti. ',time.ctime()
this_ti.ti_main(walker)
print 'Done ti. ',time.ctime()
