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
        build_data_general(mnist_trainset, mnist_testset, image_sidel_use, nodes_per_h_layer, \
        target_n_per_class = datapoints_per_class, indices_file = indfl)

    print "Finished building data subset "+str(time.ctime())

    n_pixels = image_sidel_use**2
    nn_pef = NNPeForces(train_images, train_labels, n_pixels, nodes_per_h_layer, n_classes, n_h_layers)

    return nn_pef # nn_pef IS THE VARIABLE TO CALL FOR FORCES AND POTENTIAL ENERGIES in other modules

class LogisticFeedForward(nn.Module):

    def __init__(self, n_input_nodes, n_nodes_h_layers, n_output_classes, n_h_layers, initial_param):
        super(LogisticFeedForward, self).__init__()
        self.n_input_nodes = n_input_nodes
        self.n_nodes_h_layers = n_nodes_h_layers
        self.n_output_classes = n_output_classes
        self.n_h_layers = n_h_layers
        self.layers = []

        for ind in xrange(self.n_h_layers):
            lname = "hlinear"+str(ind)
            if (ind == 0):
                insize = self.n_input_nodes
                outsize = self.n_nodes_h_layers
            else:
                insize = self.n_nodes_h_layers
                outsize = self.n_nodes_h_layers
            params = initial_param[ind]

            self.set_h_layer(lname, insize, outsize, params)

        self.set_h_layer('hlinearout',self.n_nodes_h_layers,self.n_output_classes,initial_param[-1])

    def forward(self,x):

        for layer in self.layers:
            x = torch.sigmoid(layer(x))
        return x

    def set_h_layer(self,layername, insize, outsize, params):
        self.layername = nn.Linear(insize, outsize)
        self.layername.weight.data = params[0]
        self.layername.bias.data = params[1]
        self.layers.append(self.layername)

    def return_params_as_array(self):
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

        pe, forces = self.calc(w)
        return pe

    def forces(self,w):

        pe, forces = self.calc(w, do_forces = True)
        return forces

    def calc(self,w, do_forces = False):

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

def build_data_general(fulltrain_dataset, fulltest_dataset, image_sidel_use, nodes_per_h_layer, \
    target_n_per_class = 100, indices_file = "dataindices.txt", \
    append_testset = True):
    """Builds and records a repeatable dataset using stratified sampling. Dataset indicies 
    written to indicies_file."""

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
    """Get stratified samples from dataset."""
    """This function adapted from ShitalShah's stackoverflow post here:
    https://stackoverflow.com/questions/50544730/split-dataset-train-and-test-in-pytorch-using-custom-dataset."""
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
    node as a float. The fan in is the number of inwards pointing weights + 1."""

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
