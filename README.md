# nn_sample

### Research

Please take a look at https://arxiv.org/abs/1904.04154 for some research performed using this repository. The same document is also available here as report.pdf.

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

| Parameter | Type | Default value | Meaning |
|:-------------:|:-------------:|:-------------:|:-------------:|
| runtoken | Integer | 1           | An integer used for labelling the files for this run |
| nproc | Integer | 16             | Number of processors to use. |
| nT | Integer | 16                | Number of temperatures to use. |
| Tmin | Float | 1.0e-2          | Lowest temperature in ladder of temperatures. |
| Tmax | Float | 1.0e0         | Maximum temperature in ladder of temperatures.  |
| absxmaxfac | Float | 5.0e1    | The parameters of the NN will be constrained to be with a range [-absxmax : absxmax] where absxmax = absxmaxfac/sqrt(k_i), with k_i the inward degree of the neuron to which parameter i transmits values. |
| gprior_std | None or float | None      | If this is set to a real value then an additional term is applied to (H)MC acceptance/rejection such that the target distribution is proportional to multivariate Gaussian with this standard deviation for each dimension. |
| dt_initial | Float | 1.0e-1    | Initial time step (or step size). This will be updated algorithmically, but a good starting point saves time. |
| num_traj | Integer  | 10**1          | The number of trajectories run per iteration. |
| traj_len | Integer | 10**2         | The number of time steps in a single trajectory. |
| maxiter | Integer | 10**4       | Max number of iterations to run. |
| iterstoswap | Integer | 1        | Configuration swaps between neighbouring temperatures are attempted every iterstoswap iterations. |
| iterstowaypoint | Integer | 1    | Restart information is written after every iterstowaypoint iterations. |
| iterstosetdt | Integer | 25      | The step sizes (or equivalently time steps) are updated after every iterstosetdt iterations. |
| iterstowritestate | Integer | 1  | The latest potential energy values and coordinates are written out after every iterstowritestate iterations. |
| n_h_layers | Integer | 3         | The number of hidden layers. |
| nodes_per_h_layer | Integer | 40 | The number of nodes in each hidden layer. |
| image_sidel_use | Integer | 16   | Images will be transformed to have this many pixels along the side. |
| datapoints_per_class | Integer | 50 | Number of stratified samples to draw per class. |

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

As for pt_nn.py, this will read run parameters from a dictionary (json object) in file ti_nn.config.
Default parameters are given in the following table.

| Parameter | Type | Default value | Meaning |
|:-------------:|:-------------:|:-------------:|:-------------:|
| runtoken | Integer | 1        | An integer used for labelling the files for this run. |
| coords_file | String | "localmin_data50_h3_flin.txt"  | File containing sampling.NewWalker class object saved as dict using sampling.write_walker . The coordinates should correspond to a local minimum of the cost function. These coordinates were found by minimising the lowest energy parameters discovered in a parallel tempering run. The gradient for every individual dimension is smaller that 1.0e-4. This potential energy surface corresponds to a network with 3 hidden layers, 256 (=16**2) input neurons, 3 hidden layers each containing 40 logistic neurons and 10 linear output neurons, terminated with a softmax. All hidden and output neurons use a bias. The potential is additionally specified by the data which is the stratified sample from MNIST, stored in data50.txt, and comprising 50 data points for each of the 10 digit classes. |
| T | Float | 1.0e-4          | Dimensionless temperature T = 1.0/beta |
| absxmaxfac | Float | 1.0e5  | The parameters of the NN will be constrained to be with a range '[-absxmax,absxmax]' where absxmax = absxmaxfac/sqrt(k_i), with k_i the inward degree of the neuron to which parameter i transmits values. |
| gprior_std | None or float | None   | If this is set to a real value then an additional term is applied to (H)MC acceptance/rejection such that the target distribution is proportional to a multivariate Gaussian with this standard deviation for each dimension. |
| dt_initial | Float | 1.0e-1 | Initial time step (or step size). This will be updated algorithmically, but a good starting point saves time. |
| ntraj_burnin | Integer | 100  | The number of burn in trajectories to run for each bridging distribution. |
| ntraj_sample | Integer | 100  | The number of sampling trajectories to run for each bridging distribution. |
| traj_len | Integer | 100      | The number of time steps to use for each trajectory. |
| nbridge | Integer | 100       | Number of bridging distributions to use. Including sampling the distribution corresponding to the true potential, and the quadratic approximation potential, nbridge+2 distributions are sampled. |
| ntimes_set_dt | Integer | 10  | dt is updated after sampling every nbridge/ntimes_set_dt distributions. |
| iterstowaypoint | Integer | 10 | Restart information is written after sampling every iterstowaypoint distributions. |
| n_h_layers | Integer | 3      | The number of hidden layers. |
| nodes_per_h_layer | Integer | 40 | The number of nodes in each hidden layer. |
| image_sidel_use | Integer | 16 | Images will be transformed to have this many pixels along the side. |
| datapoints_per_class | Integer | 50 | Number of stratified samples to draw per class. |


If you wish to perform multiple thermodynamic integration calculations in parallel, then it is advisable to first set 
```
$ export OMP_NUM_THREADS=1
```

### Data Sets

To ensure reproducibility, data sets used in the calculations shown in report.pdf are included here.

| Data Set      | File Name     |
|:-------------:|:-------------:|
| D_50          | data5.txt     |
| D_500         | data50.txt    |
| D_5000        | data500.txt   |

### Comment

In the code I have referred "Parallel Tempering". This is the name for Replica Exchange Molecular Dynamics when you replace the Molecular Dynamics (MD) with Markov chain Monte Carlo (MCMC). My code is completely agnostic to the Monte Carlo approach used. I've implemented Hamiltonian Monte Carlo (HMC), which is intermediate to MD and MCMC. I chose to use the name Parallel Tempering when writing the code.
