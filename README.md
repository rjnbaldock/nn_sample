# nn_sample

### Research

Please take a look at https://arxiv.org/abs/1904.04154 for some research performed using this repository. The same document is also availble here as report.pdf.

### Prerequisites

This project is written in python2.7 and requires installation of the python modules numpy, scipy and pathos. https://pypi.org/project/pathos/ 

numpy, scipy and pathos can be installed by doing
```
$ pip install -r requirements.txt
```
PyTorch and torchvision are also required. See https://pytorch.org/get-started/locally/ for installation instructions.

### Running replica-exchange
An example replica-exchange calculation can be performed by doing

```
$ export OMP_NUM_THREADS=1
$ python pt_nn.py
```

This will read run parameters from a dictionary (json object) in file pt_nn.config.
Default parameters are given in the following table.

| Parameter | Default value | Meaning |
|:-------------:|:-------------:|:-------------:|
| "runtoken" | 1           | An integer used for labelling the files for this run |
| "nproc" | 16             | Number of processors to use. |
| "nT" | 16                | Number of temperatures to use. |
| "Tmin" | 1.0e-2          | Lowest temperature in ladder of temperatures. |
| "Tmax" | 1.0e0         | Maximum temperature in ladder of temperatures.  |
| "absxmaxfac" | 5.0e1    | The parameters of the NN will be constrained to be with a range [-absxmax : absxmax] where absxmax = absxmaxfac/sqrt(k_i), with k_i the inward degree of the neuron to which parameter i transmits values. |
| "gprior_std" | None      | If this is set to a real value then an additional term is applied to (H)MC acceptance/rejection such that the target distribution is proportional to multivariate Gaussian with this standard deviation for each dimension. |
| "dt_initial" | 1.0e-1    | Initial time step (or step size). This will be updated algorithmically, but a good starting point saves time. |
| "num_traj" | 10**1          | The number of trajectories run per iteration. |
| "traj_len" | 10**2         | The number of time steps in a single trajectory. |
| "maxiter" | 10**4       | Max number of iterations to run. |
| "iterstoswap" | 1        | Configuration swaps between neighbouring temperatures are attempted every iterstoswap iterations. |
| "iterstowaypoint" | 1    | Restart information is written after every iterstowaypoint iterations. |
| "iterstosetdt" | 25      | The step sizes (or equivalently time steps) are updated after every iterstosetdt interations. |
| "iterstowritestate" | 1  | The latest potential energy values and coordinates are written out after every iterstowritestate iterations. |
| "n_h_layers" | 1         | The number of hidden layers. |
| "nodes_per_h_layer" | 40 | The number of nodes in each hidden layer. |
| "image_sidel_use" | 16   | Images will be transformed to have this many pixels along the side. |
| "datapoints_per_class" | 50 | Number of stratified samples to draw per class. |

Doing 
```
$ export OMP_NUM_THREADS=1
```
is important for good performance when combining pathos multiprocessing and PyTorch.

### Running thermodynamic integration

An example thermodynamic integration calculation can be run by doing
```
$ python ti_nn.py
```
If you wish to perform multiple thermodynamic integration calculations in parallel, then it is advisable to first set 
```
$ export OMP_NUM_THREADS=1
```

### Data Sets

To ensure reproducability, data sets used in the calculations shown in report.pdf are included here.

| Data Set      | File Name     |
|:-------------:|:-------------:|
| D_50          | data5.txt     |
| D_500         | data50.txt    |
| D_5000        | data500.txt   |

### Comment

In the code I have referred "Parallel Tempering". This is the name for Replica Exchange Molecular Dynamics when you replace the Molecular Dynamics (MD) with Markov chain Monte Carlo (MCMC). My code is completely agnostic to the Monte Carlo approach used. I've implemented Hamiltonian Monte Carlo (HMC), which is intermediate to MD and MCMC. I chose to use the name Parallel Tempering when writing the code.
