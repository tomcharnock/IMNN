"""Simple network builder

This module provides the methods necessary to build simple networks for use in
the information maximising neural network. It is highly recommeneded to build
your own network from scratch rather than using this blindly - you will almost
certainly get better results.
"""


__version__ = "0.1dev3"
__author__ = "Tom Charnock"


import tensorflow as tf
import numpy as np


class network():
    """Simple network builder
    """

    def __init__(self, parameters):
        self.layers =
        self.bb =
        self.activation =

    def dense(self, input_tensor, l):
        """Builds a dense layer

        Parameters
        __________
        input_tensor : :obj:`TF tensor`
            input tensor to the layer
        l : int
            counter for the number of layers
        previous_layer : int
            shape of previous layer
        weight_shape : tuple
            shape of the weight kernel
        bias_shape : tuple
            shape of the bias kernel
        weights : :obj:`TF tensor`
            the weight kernel for the dense layer
        biases : :obj:`TF tensor
            the biases for the dense layer
        dense : :obj:`TF tensor`
            non-activated output dense layer

        Returns
        _______
        :obj:`TF tensor`
            activated output from the dense layer
        """
        previous_layer = int(input_tensor.get_shape().as_list()[-1])
        weight_shape = (previous_layer, self.layers[l])
        bias_shape = (self.layers[l])
        weights = tf.get_variable("weights", weight_shape,
                                  initializer=tf.variance_scaling_initializer(
                                  ))
        biases = tf.get_variable("biases", bias_shape,
                                 initializer=tf.constant_initializer(self.bb))
        dense = tf.add(tf.matmul(input_tensor, weights), biases)
        return self.activation(dense, name='dense_' + str(l))

    def conv(self, input_tensor, l):
        """Builds a dense layer

        Parameters
        __________
        input_tensor : :obj:`TF tensor`
            input tensor to the layer
        l : int
            counter for the number of layers
        previous_filters : int
            number of filters in the previous layer
        weight_shape : tuple
            shape of the weight kernel
        stride_shape : list
            the size of strides to make in the convolution
        bias_shape : tuple
            shape of the bias kernel
        weights : :obj:`TF tensor`
            the weight kernel for the dense layer
        biases : :obj:`TF tensor
            the biases for the dense layer
        dense : :obj:`TF tensor`
            non-activated output dense layer

        Returns
        _______
        :obj:`TF tensor`
            activated output from the dense layer
        """
        previous_filters = int(input_tensor.get_shape().as_list()[-1])
        if len(self.layers[l][1]) == 1:
            convolution = tf.nn.conv1d
            weight_shape = (self.layers[l][1][0], previous_filters,
                            self.layers[l][0])
            stride_shape = self.layers[l][2][0]
        elif len(self.layers[l][1]) == 2:
            convolution = tf.nn.conv2d
            weight_shape = (self.layers[l][1][0], self.layers[l][1][1],
                            previous_filters, self.layers[l][0])
            stride_shape = [1] + self.layers[l][2] + [1]
        else:
            convolution = tf.nn.conv3d
            weight_shape = (self.layers[l][1][0], self.layers[l][1][1],
                            self.layers[l][1][2], previous_filters,
                            self.layers[l][0])
            stride_shape = [1] + self.layers[l][2] + [1]
        bias_shape = (self.layers[l][0])
        weights = tf.get_variable("weights", weight_shape,
                                  initializer=tf.variance_scaling_initializer(
                                  ))
        biases = tf.get_variable("biases", bias_shape,
                                 initializer=tf.constant_initializer(self.bb))
        conv = tf.add(convolution(input_tensor, weights, stride_shape,
                                  padding=self.layers[l][3]), biases)
        return n.activation(conv, name='conv_' + str(l))

    def build_network(self, input_tensor):
        """Construct a simple network for the IMNN

        Parameters
        __________
        input_tensor : :obj:`TF tensor`
            input tensor to the network
        layer : :obj:`list` of :obj:`TF tensor`
            a list containing each output of the previous layer

        Returns
        _______

        """
        layer = [input_tensor]
        for l in range(1, len(self.layers)):
            if type(self.layers[l]) == list:
                if len(layer[-1].get_shape(
                        ).as_list()) < len(self.layers[l]) - 1:
                    layer.append(
                        tf.reshape(
                            layer[-1], [-1] + layer[-1].get_shape(
                            ).as_list()[1:] +
                            [1 for i in range(
                                len(self.layers[l])
                                - len(layer[-1].get_shape().as_list()) - 1)]))
                with tf.variable_scope('layer_' + str(l)):
                    layer.append(self.conv(layer[-1], l, drop_val))
            else:
                if len(layer[-1].get_shape()) > 2:
                    layer.append(
                        tf.reshape(
                            layer[-1], (-1, np.prod(layer[-1].get_shape(
                            ).as_list()[1:]))))
                with tf.variable_scope('layer_' + str(l)):
                    layer.append(n.dense(layer[-1], l, drop_val))
        return layer[-1]
