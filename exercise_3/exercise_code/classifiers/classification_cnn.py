"""ClassificationCNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ClassificationCNN(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, kernel_size=7,
                 stride_conv=1, weight_scale=0.001, pool=2, stride_pool=2, hidden_dim=100,
                 num_classes=10, dropout=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - kernel_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride_conv: Stride for the convolution layer.
        - stride_pool: Stride for the max pooling layer.
        - weight_scale: Scale for the convolution weights initialization
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(ClassificationCNN, self).__init__()
        channels, height, width = input_dim

        ########################################################################
        # TODO: Initialize the necessary trainable layers to resemble the      #
        # ClassificationCNN architecture  from the class docstring.            #
        #                                                                      #
        # In- and output features should not be hard coded which demands some  #
        # calculations especially for the input of the first fully             #
        # convolutional layer.                                                 #
        #                                                                      #
        # The convolution should use "same" padding which can be derived from  #
        # the kernel size and its weights should be scaled. Layers should have #
        # a bias if possible.                                                  #
        #                                                                      #
        # Note: Avoid using any of PyTorch's random functions or your output   #
        # will not coincide with the Jupyter notebook cell.                    #
        ########################################################################
        
        ### lecture (dimensions of cnns)
        height_same_pad = int(((height - 1)*stride_conv - height + kernel_size)/2) # H=(H-F+2P)/S + 1
        width_same_pad = int(((width - 1)*stride_conv - width + kernel_size)/2) # W=(W-F+2P)/S + 1
        
        height_post_conv1 = int((height + height_same_pad + width_same_pad - kernel_size)/stride_conv) + 1
        width_post_conv1 = int((width + height_same_pad + width_same_pad - kernel_size)/stride_conv) + 1
        in_features_fc_height = int((height_post_conv1 - pool)/stride_pool) + 1
        in_features_fc_width = int((width_post_conv1 - pool)/stride_pool) + 1
        ###
        
        self.conv1 = nn.Conv2d(channels, num_filters, kernel_size, stride=stride_conv, 
                               padding=(height_same_pad, width_same_pad), bias=True)
        with torch.no_grad():
            self.conv1.weight = self.conv1.weight.mul_(weight_scale) 
        
        self.max_pool = nn.MaxPool2d(pool, stride=stride_pool)
        self.fc1 = nn.Linear(in_features_fc_height*in_features_fc_width*num_filters, hidden_dim, bias=True)
        self.dropout1 = nn.Dropout2d(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=True)
    
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x):
        """
        Forward pass of the neural network. Should not be called manually but by
        calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        ########################################################################
        # TODO: Chain our previously initialized fully-connected neural        #
        # network layers to resemble the architecture drafted in the class     #
        # docstring. Have a look at the Variable.view function to make the     #
        # transition from the spatial input image to the flat fully connected  #
        # layers.                                                              #
        ########################################################################
        
        x = self.conv1(x)
        x = self.max_pool(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x


    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
