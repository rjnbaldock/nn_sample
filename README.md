# nn_sample

### Prerequisites

This project is written in python2.7 and requires installation of the python module pathos. https://pypi.org/project/pathos/
```
$ pip install pathos
```
PyTorch is also required. See https://pytorch.org/get-started/locally/ for installation instructions.

### To run
An example replica exchange calculation can be performed by doing

```
$ export OMP_NUM_THREADS=1
$ python pt_nn.py
```

Doing 
```
$ export OMP_NUM_THREADS=1
```
is important for good performance when combining pathos multiprocessing and PyTorch.


Similarly, an example thermodynamic integration calculation can be run by doing
```
$ python ti_nn.py
```
If you wish to perform multiple thermodynamic integration calculations in parallel, then it is advisable to first set 
```
$ export OMP_NUM_THREADS=1
```

### Comment

This work and this repo are fresh! I'll soon complete this README, but for now, why not take a look at my report (report.pdf)?

In the code I have referred "Parallel Tempering". This is the name for Replica Exchange Molecular Dynamics when you replace the Molecular Dynamics (MD) with Markov chain Monte Carlo (MCMC). My code is completely agnostic to the Monte Carlo approach used. I've implemented Hamiltonian Monte Carlo (HMC), which is intermediate to MD and MCMC. I chose to use the name Parallel Tempering when writing the code. You'll be able to read about my implementation in the report very soon.
