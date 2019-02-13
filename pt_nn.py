import sampling
import pt
import nn_pe_force
import numpy as np
import time

# Set run parameters
RUNTOKEN = 1        # An integer used for labelling the files for this run

NPROC = 16          # Number of processors to use.
NT = 16             # Number of temperatures to use.
TMIN = 1.0e-2       # Lowest temperature in ladder of temperatures.
TMAX = (1.0e0)      # Maximum temperature in ladder of temperatures. 
ABSXMAXFAC = 1.0e3  # The parameters of the NN will be constrained to be with a range 
                    # '[-absxmax,absxmax]' where absxmax = ABSXMAXFAC/sqrt(k_i), with k_i the inward 
                    # degree of the neuron to which parameter i transmits values.
GPRIOR_STD = None   # If this is set to a real value then an additional term is applied to (H)MC 
                    # acceptance/rejection such that the target distribution is proportional to a 
                    # multivariate Gaussian with this standard deviation for each dimension. 


DT_INITIAL = 1.0e-1 # Initial time step (or step size). This will be updated algorithmically, but a 
                    # good starting point saves time.

NUM_TRAJ = 10       # The number of trajectories run per iteration.
TRAJ_LEN = 100      # The number of time steps in a single trajectory.

MAXITER = 100000    # Max number of iterations to run.
ITERSTOSWAP = 1     # Configuration swaps between neighbouring temperatures are attempted every 
                    # ITERSTOSWAP iterations.
ITERSTOWAYPOINT = 1 # Restart information is written after every ITERSTOWAYPOINT iterations.
ITERSTOSETDT = 25   # The step sizes (or equivalently time steps) are updated after every 
                    # ITERSTOSETDT interations.
ITERSTOWRITESTATE = 1 # The latest potential energy values and coordinates are written out after 
                    # every ITERSTOWRITESTATE iterations.

N_H_LAYERS = 1      # The number of hidden layers.
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

these_masses = 1.0 # all parameters will have the same effective timestep.

# Set limits on parameter values for random (uniform) initialisation.
# Use standard [-1/sqrt(fan in) , 1/sqrt(fan in)]
intial_absxmax = 1.0/np.sqrt(nn_pe_force.calc_fan_in(IMAGE_SIDEL_USE**2,N_H_LAYERS,NODES_PER_H_LAYER,N_CLASSES))
nd = calc_numdim(IMAGE_SIDEL_USE, N_H_LAYERS, NODES_PER_H_LAYER, N_CLASSES) # calculate total number of parameters for NN

# Initialise object to calculate pe and force values for NN.
# Uses data indicies from MNIST specified in DATAFILE to get reproducible cost function.
# Data is stratified if no file is specified.
# DATAPOINTS_PER_CLASS data points, per class.
nnpef = nn_pe_force.build_repeatable_NNPeForces(indfl = DATAFILE,image_sidel_use=IMAGE_SIDEL_USE, \
    n_h_layers=N_H_LAYERS, nodes_per_h_layer=NODES_PER_H_LAYER, datapoints_per_class=DATAPOINTS_PER_CLASS)

# Initialise random walkers or read restart file.
# Initialise parallel tempering object.
thispt = pt.build_pt(sampling.Hmc, pe_method = nnpef.pe, force_method = nnpef.forces, numdim = nd, \
    masses = these_masses, nT = NT, nproc = NPROC, Tmin = TMIN, Tmax = TMAX, \
    max_iteration = MAXITER, iters_to_swap = ITERSTOSWAP, iters_to_waypoint = ITERSTOWAYPOINT, \
    iters_to_setdt = ITERSTOSETDT, iters_to_writestate = ITERSTOWRITESTATE, run_token = RUNTOKEN, \
    dt = DT_INITIAL, traj_len = TRAJ_LEN, num_traj = NUM_TRAJ, absxmax = ABSXMAXFAC*intial_absxmax, \
    initial_rand_bounds = intial_absxmax, gaussianprior_std = GPRIOR_STD )

# Update boundaries on parameters, for softer prior.
for this_traj in thispt.pt_trajs:
    this_traj.sampler.absxmax = ABSXMAXFAC*intial_absxmax
    this_traj.sampler.dt_max = np.median(this_traj.sampler.absxmax)

# Run parallel tempering
print 'About to do pt. ',time.ctime()
thispt.ptmain()
print 'Done pt. ',time.ctime()
