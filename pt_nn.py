import sampling
import pt
import nn_pe_force
import numpy as np
import time

# Set run parameters
RUNTOKEN = 1

NPROC = 16
NT = 16
TMIN = 1.0e-2
TMAX = (1.0e0)
ABSXMAXFAC = 1.0e3
GPRIOR_STD = None
DT_INITIAL = 1.0e-1

NUM_TRAJ = 10
TRAJ_LEN = 100

MAXITER = 100000
ITERSTOSWAP = 1
ITERSTOWAYPOINT = 1
ITERSTOSETDT = 25
ITERSTOWRITESTATE = 1

N_H_LAYERS = 1
NODES_PER_H_LAYER = 40
IMAGE_SIDEL_USE = 16
DATAPOINTS_PER_CLASS = 50
DATAFILE = "data50.txt"
N_CLASSES = 10

def calc_numdim(image_sidel_use,n_h_layers,nodes_per_h_layer,n_classes ):
    """This function calculates the total number of parameters that will feature in the specified NN."""

    n_nodes_lowerlayer, n_nodes_thislayer = image_sidel_use**2, nodes_per_h_layer # first hidden layer
    numdim = (1 + n_nodes_lowerlayer) * n_nodes_thislayer
    n_nodes_lowerlayer, n_nodes_thislayer = nodes_per_h_layer, nodes_per_h_layer # other hidden layers
    numdim += ( n_h_layers - 1 ) * (1 + n_nodes_lowerlayer) * n_nodes_thislayer
    n_nodes_lowerlayer, n_nodes_thislayer = nodes_per_h_layer, n_classes # output layer
    numdim += (1 + n_nodes_lowerlayer) * n_nodes_thislayer

    return numdim

these_masses = 1.0 # all parameters will have the same effective timestep.

# Set limits on parameter values for random (uniform) initialisation.
# Use standard [-1/\sqrt(fan in) , 1/sqrt(fan in)]
intial_absxmax = 1.0/np.sqrt(nn_pe_force.calc_fan_in(IMAGE_SIDEL_USE**2,N_H_LAYERS,NODES_PER_H_LAYER,N_CLASSES))
nd = calc_numdim(IMAGE_SIDEL_USE, N_H_LAYERS, NODES_PER_H_LAYER, N_CLASSES) # calculate total number of parameters for NN

# initialise object to calculate pe and force values for NN
# uses data indicies from MNIST specified in DATAFILE to get reproducible cost function.
# data is stratified if no file is specified.
# DATAPOINTS_PER_CLASS data points, per class...
nnpef = nn_pe_force.build_repeatable_NNPeForces(indfl = DATAFILE,image_sidel_use=IMAGE_SIDEL_USE, \
    n_h_layers=N_H_LAYERS, nodes_per_h_layer=NODES_PER_H_LAYER, datapoints_per_class=DATAPOINTS_PER_CLASS)

# initialise random walkers or read restart file.
# initialise parallel tempering object.
thispt = pt.build_pt(sampling.Hmc, pe_method = nnpef.pe, force_method = nnpef.forces, numdim = nd, \
    masses = these_masses, nT = NT, nproc = NPROC, Tmin = TMIN, Tmax = TMAX, \
    max_iteration = MAXITER, iters_to_swap = ITERSTOSWAP, iters_to_waypoint = ITERSTOWAYPOINT, \
    iters_to_setdt = ITERSTOSETDT, iters_to_writestate = ITERSTOWRITESTATE, run_token = RUNTOKEN, \
    dt = DT_INITIAL, traj_len = TRAJ_LEN, num_traj = NUM_TRAJ, absxmax = ABSXMAXFAC*intial_absxmax, \
    initial_rand_bounds = intial_absxmax, gaussianprior_std = GPRIOR_STD )

# update boundaries on parameters, for softer prior.
for this_traj in thispt.pt_trajs:
    this_traj.sampler.absxmax = ABSXMAXFAC*intial_absxmax
    this_traj.sampler.dt_max = np.median(this_traj.sampler.absxmax)

# run parallel tempering
print 'About to do pt. ',time.ctime()
thispt.ptmain()
print 'Done pt. ',time.ctime()
