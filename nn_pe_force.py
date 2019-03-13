#!/usr/bin/env python
"""This module calculates the potential energy field over the parameters of a classifier, which is
a feed forward neural network of logistic neurons terminated with linear layer to which I apply a 
softmax. It also calculates the negative gradient (called the force) for that field. The potential 
energy is just the cross entropy loss, which is the negative logarithm of the probability the 
classifier correctly assigns the labels of every image in the training set. The module also builds 
or imports a stratified subset of the MNIST training set, so that the energy field is reproducable."""

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import os.path
import numpy as np
import time

def build_repeatable_NNPeForces(indfl = "dataindices.txt", image_sidel_use = 20, n_h_layers = 2, \
    nodes_per_h_layer = 20,  datapoints_per_class = 100):
    """This function builds a repeatable (specific) dataset from MNIST with datapoints_per_class 
    items per class. The indicies of those data items are read/stored from/in infdl. This function
    returns a NNPeForces class object from which potential energies and forces can be returned over
    the parameters of the neural network. These energies and forces are functions of the (repeatable)
    data set."""

    transform = transforms.Compose([transforms.Resize((image_sidel_use,image_sidel_use)), \
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    print "Getting MNIST data "+str(time.ctime())

    n_classes = 10
    mnist_trainset = datasets.MNIST(root="./mnistdata", train=True, download=True, \
        transform=transform)
    mnist_testset = datasets.MNIST(root="./mnistdata", train=False, download=False, \
        transform=transform)

    print "About to build data subset "+str(time.ctime())
    train_images, train_labels, test_images, test_labels = \
        build_data_general(mnist_trainset, mnist_testset, image_sidel_use, \
        target_n_per_class = datapoints_per_class, indices_file = indfl)

    print "Finished building data subset "+str(time.ctime())

    n_pixels = image_sidel_use**2
    nn_pef = NNPeForces(train_images, train_labels, n_pixels, nodes_per_h_layer, n_classes, n_h_layers)

    return nn_pef # nn_pef IS THE VARIABLE TO CALL FOR FORCES AND POTENTIAL ENERGIES in other modules

class LogisticFeedForward(nn.Module):
    """This class defines a (feed forward) network of logistic neurons, terminated with a softmax
    and with the parameters specified in initial_param.
    
    Args:
        n_input_nodes (int) : the number of input nodes
        n_nodes_h_layers (int) : the number of nodes in each hidden layer
        n_output_classes (int) : the number of nodes in the final (output) layer
        n_h_layers (int) : the number of hidden layers
        initial_param (list) : List of lists containing the weights and bias values for neurons in 
            each layer. Each element of initial_param refers to a seperate layer of the network.
            The 0th element refers to the first layer and the last element to the output layer.
            Each element of initial_param has the form [weights, biases] where
            weights and biases are pytorch tensors.

    Attributes:
        n_input_nodes (int) : the number of input nodes
        n_nodes_h_layers (int) : the number of nodes in each hidden layer
        n_output_classes (int) : the number of nodes in the final (output) layer
        n_h_layers (int) : the number of hidden layers
        layers : list of nn.Linear layers which comprise the nn.

    """

    def __init__(self, n_input_nodes, n_nodes_h_layers, n_output_classes, n_h_layers, initial_param):
        super(LogisticFeedForward, self).__init__()
        self.n_input_nodes = n_input_nodes
        self.n_nodes_h_layers = n_nodes_h_layers
        self.n_output_classes = n_output_classes
        self.n_h_layers = n_h_layers
        self.layers = []

        for ind in xrange(self.n_h_layers):
            if (ind == 0):
                insize = self.n_input_nodes
                outsize = self.n_nodes_h_layers
            else:
                insize = self.n_nodes_h_layers
                outsize = self.n_nodes_h_layers
            params = initial_param[ind]

            self.set_h_layer(insize, outsize, params)

        self.set_h_layer(self.n_nodes_h_layers,self.n_output_classes,initial_param[-1])

    def forward(self,x):
        """Forward pass for the network. """

        for i in xrange(len(self.layers)-1):
            layer = self.layers[i]
            x = torch.sigmoid(layer(x))
        x = self.layers[-1](x)

        return x

    def set_h_layer(self, insize, outsize, params):
        """Creates a layer for the network and appends it to array self.layers."""
        layer = nn.Linear(insize, outsize)
        layer.weight.data = params[0]
        layer.bias.data = params[1]
        self.layers.append(layer)

    def return_params_as_array(self):
        """Returns network parameters as a 1-d numpy array."""
        out = None
        for layer in self.layers:
            w = layer.weight.data
            w = w.view(np.prod(w.size())) # reshape to a 1d array

            b = layer.bias.data
            if (out is None):
                out= torch.cat((w,b))
            else:
                out= torch.cat((out,w,b))

        return out.numpy()

class NNPeForces():
    """This class enables the evaluation of the potential energy (cross entropy loss) and 
    forces (negative gradient of the cross entropy loss) for a neural network.

    Args:
        images: training image data set
        labels: training image labels
        n_input_nodes (int) : the number of input nodes
        n_nodes_h_layers (int) : the number of nodes in each hidden layer
        n_output_classes (int) : the number of nodes in the final (output) layer
        n_h_layers (int) : the number of hidden layers

    Attributes:
        images: training image data set
        labels: training image labels
        n_input_nodes (int) : the number of input nodes
        n_nodes_h_layers (int) : the number of nodes in each hidden layer
        n_output_classes (int) : the number of nodes in the final (output) layer
        n_h_layers (int) : the number of hidden layers
        criterion: torch.nn loss function (hardcoded to cross entropy loss)
        wstore (numpy array) : last set of weights entered
        pestore (float) : potential energy for wstore
        fstore (numpy array) : forces for wstore

    """
    def __init__(self,images,labels, n_input_nodes, n_nodes_h_layers, n_output_classes, n_h_layers ):
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.images = images # fixed, full batch
        self.labels = labels # fixed, full batch
        self.n_input_nodes = n_input_nodes
        self.n_nodes_h_layers = n_nodes_h_layers
        self.n_output_classes = n_output_classes
        self.n_h_layers = n_h_layers

        self.wstore = None
        self.pestore = None
        self.fstore = None

    def pe(self,w):
        """Call this function obtain the potential energy.

        Args:
            w: 1-d numpy array of floats specifying the parameters of the neural network.

        Return:
            pe (float)
        """

        pe, forces = self.calc(w)
        return pe

    def forces(self,w):
        """Call this function to obtain the forces (negative gradient).

        Args:
            w: 1-d numpy array of floats specifying the parameters of the neural network.

        Return:
            forces (1-d numpy array of floats with same length as w)
        """

        pe, forces = self.calc(w, do_forces = True)
        return forces

    def calc(self,w, do_forces = False):
        """This is the (hidden) routine that actually calculates the potential energy and forces.
        The code is organised this way because a lot of the code is shared. If the network parameters 
        are the same as those presented last time, then the result is recalled from memory.
        
        Args:
            w:  1-d numpy array of floats specifying the parameters of the neural network.
            do_forces (logical, default True) : if True then the backwards pass is performed for the
                network to obtain the gradients of the parameters. (Or if stored last time, 
                the forces are instead recalled from memory.)

        Return:
            pe (float), forces (1-d numpy array of floats with same length as w).
        """

        if (self.wstore is not None):
            # quick lookup in case the method was called previously without changing w
            if (np.array_equal(w, self.wstore)): # we can look up answer
                if ((not do_forces) or (not (do_forces and (self.fstore is None)))):
                    # return this if forces were not required or if they were and self.fstore is not None
                    return self.pestore, self.fstore

        init_params = self.w_to_nn_params(w)
        net = LogisticFeedForward(n_input_nodes = self.n_input_nodes, \
            n_nodes_h_layers = self.n_nodes_h_layers, n_output_classes = self.n_output_classes, \
            n_h_layers = self.n_h_layers, initial_param=init_params)
        out = net(self.images)
        loss = self.criterion(out, self.labels)

        pe = loss.item()

        if (do_forces):
            net.zero_grad()
            loss.backward()
            forces = self.grads_to_forces(net)
        else:
            forces = None

        self.wstore = w.copy()
        self.pestore = pe
        if (forces is None):
            self.fstore = forces
        else:
            self.fstore = forces.copy()

        return self.pestore, self.fstore

    def w_to_nn_params(self,w):
        """Converts a 1-d numpy array (w) into a list of lists, containing PyTorch tensors specifying
        the parameters of the neural network.
        
        Args:
            w:  1-d numpy array of floats specifying the parameters of the neural network.
         
        Return:
        initial_param (list) : List of lists containing the weights and bias values for neurons in 
            each layer. Each element of initial_param refers to a seperate layer of the network.
            The 0th element refers to the first layer and the last element to the output layer.
            Each element of initial_param has the form [weights, biases] where
            weights and biases are pytorch tensors.
        """

        init_params = []
        wstrt, wend = 0, 0

        for ind in xrange(self.n_h_layers):
            init_params.append([])

            if (ind == 0): # first layer takes input from the input nodes
                inputs = self.n_input_nodes
            else:
                inputs = self.n_nodes_h_layers

            outputs = self.n_nodes_h_layers
            wend += inputs*outputs
            init_params[-1].append(torch.from_numpy(w[wstrt:wend]).reshape_as(torch.ones(outputs,inputs)).type(torch.FloatTensor))

            wstrt = wend
            wend += outputs
            init_params[-1].append(torch.from_numpy(w[wstrt:wend]).type(torch.FloatTensor))
            wstrt = wend

        init_params.append([])
        inputs = self.n_nodes_h_layers
        outputs = self.n_output_classes
        wstrt = wend
        wend += inputs*outputs
        init_params[-1].append(torch.from_numpy(w[wstrt:wend]).reshape_as(torch.ones(outputs,inputs)).type(torch.FloatTensor))

        wstrt = wend
        wend += outputs
        init_params[-1].append(torch.from_numpy(w[wstrt:wend]).type(torch.FloatTensor))

        return init_params

    def grads_to_forces(self,net):
        """Converts the grad of the network parameters, calculated by PyTorch, into a 1-d numpy array
        specifying the forces (negative gradient) for the parameters.
        
        Args:
            net:    Instance of LogisticFeedForward class, for which backward has been called.
            
        Return:
            f:      1-d numpy array of floats specifying the forces for the parameters).
            """

        f = None
        for layer in net.layers:

            f1 = layer.weight.grad.reshape(np.prod(layer.weight.grad.size()))

            f2 = layer.bias.grad
            if (f is None):
                f = torch.cat((f1,f2))
            else:
                f = torch.cat((f,f1,f2))

        f = -f.numpy()

        return f

def build_data_general(fulltrain_dataset, fulltest_dataset, image_sidel_use, \
    target_n_per_class = 100, indices_file = "dataindices.txt", \
    append_testset = True):
    """Builds and records a repeatable dataset using stratified sampling. Dataset indicies 
    written to indicies_file.
    
    Args: 
        fulltrain_dataset: Full training data set from which samples will be chosen or read.
        fulltest_dataset: The full test set.
        image_sidel_use (int) : Images will be transformed to have this many pixels along the side.
        target_n_per_class (int) : Number of stratified samples to draw per class. If there are fewer
            than target_n_per_class items in a class in fulltrain_dataset then, all the data in that 
            class will be returned.
        indices_file (str) : name of file for storing or recovering the indicies of data points in 
            fulltrain_dataset which comprise the stratified sample. If this file exists, the indicies 
            of the data points are read from the file. If the file does not exist then it is created 
            and the indicies of a random stratified data sample are written there.
        append_test (logical) : If True, then the remaining data excluded from the stratified sample,
            is appended to the test set data. Default=True.

    Return:
        train_images, train_labels, test_images, test_labels
    """

    from collections import Counter

    ind_train = []
    if (os.path.isfile("./"+indices_file)):

        print "Reading data indices from file "+indices_file

        with open(indices_file,"r") as indfl:
            for line in indfl:
                i = int(line.split()[0])
                ind_train.append(i)

        fulltrainsize = len(fulltrain_dataset)
        ind_remain = [elem for elem in xrange(fulltrainsize) if elem not in ind_train]

    else: # indices_file does not exist

        print "Creating stratified sample set."

        ind_train, ind_remain = sampleFromClass(fulltrain_dataset, target_n_per_class)
        # save data indices for next restart
        with open(indices_file,"w") as indfl:
            for ind in ind_train:
                l = str(ind)+"\n"
                indfl.write(l)

    my_trainset = torch.utils.data.dataset.Subset(fulltrain_dataset, ind_train)
    remaining_data = torch.utils.data.dataset.Subset(fulltrain_dataset, ind_remain)
    if (append_testset):
        my_testset = torch.utils.data.ConcatDataset([fulltest_dataset,remaining_data])
    else:
        my_testset = fulltest_dataset

    train_loader = torch.utils.data.DataLoader(dataset=my_trainset, batch_size=len(my_trainset))
    test_loader = torch.utils.data.DataLoader(dataset=my_testset, batch_size=len(my_testset))

    i, train_data = list(enumerate(train_loader))[0] # this module, which defines a (noiseless) energy
                                        # function over the parameter space requires full batch error
    train_images, train_labels = train_data
    train_images = Variable(train_images.view(-1, image_sidel_use*image_sidel_use ))
    train_labels = Variable(train_labels)

    i, test_data = list(enumerate(test_loader))[0] # this module, which defines a (noiseless) energy
                                        # function over the parameter space requires full batch error
    test_images, test_labels = test_data
    test_images = Variable(test_images.view(-1, image_sidel_use*image_sidel_use ))
    test_labels = Variable(test_labels)

    print "Training data label : # occurences in stratified data set"
    train_counts = Counter( train_labels.numpy() )
    for key in train_counts.keys():
        print key,' : ', train_counts[key]

    return train_images, train_labels, test_images, test_labels

def sampleFromClass(dataset, target_num_per_class):
    """Get stratified samples from dataset.
    This function adapted from ShitalShah's stackoverflow post here:
    https://stackoverflow.com/questions/50544730/split-dataset-train-and-test-in-pytorch-using-custom-dataset.
    
    Args:
        dataset: PyTorch data set
        target_num_per_class (int) : Number of stratified samples to draw per class. If there are 
            fewer than target_n_per_class items in a class in fulltrain_dataset then, all the data 
            in that class will be returned.

    Return:
        indices_train, indices_remain:  Lists of integers specifying the indices of data points from 
            dataset. 
            indices_train: indices of data points that make up stratified sample (new training set).
            indices_remain: indices of points not in indices_train.
    """
    class_counts = {}
    indices_train = []
    indices_remain = []

    # sequence data in random order
    all_labels = []
    all_data = []
    for data, label in dataset:
        all_labels.append(label)
        all_data.append(data)
    rand_indices = np.arange(len(all_labels))
    np.random.shuffle(rand_indices)

    for ind in rand_indices:
        label = all_labels[ind]
        data = all_data[ind]

        c = label.item()
        class_counts[c] = class_counts.get(c, 0) + 1
        if class_counts[c] <= target_num_per_class:
                indices_train.append(ind)
        else:
                indices_remain.append(ind)

    return indices_train, indices_remain

def calc_fan_in(n_inputs_nodes,n_h_layers,nodes_per_h_layer,n_classes):
    """For each weight or bias term (returned in a single 1d numpy array), this function identifies
    the neuron to which that weight (bias term) is connected and returns the total fan in to that 
    node as a float. The fan in is the number of inwards pointing weights + 1.
    
    Args:
        n_input_nodes (int) : the number of input nodes
        n_h_layers (int) : the number of hidden layers
        nodes_per_h_layers (int) : the number of nodes in each hidden layer
        n_classes (int) : the number of nodes in the final (output) layer

    Return:
        fan:    1-d numpy array, with one element for each parameter in the network, each specifying
            the fan in to the node to which the weight connects inwards.
    """

    fan = np.asarray([])
    for i in xrange(n_h_layers+1):
        if (i==0): # first layer
            ins_per_node = n_inputs_nodes + 1 # including bias
            num_nodes = nodes_per_h_layer
        elif (i!=n_h_layers): # other hidden layers 
            ins_per_node = nodes_per_h_layer + 1 # including bias
            num_nodes = nodes_per_h_layer
        else: # output layer
            ins_per_node = nodes_per_h_layer + 1 # including bias
            num_nodes = n_classes
        num_weights = ins_per_node*num_nodes
        mass = ins_per_node
        fan = np.concatenate((fan, mass*np.ones(num_weights)))

    return fan
