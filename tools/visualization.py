"""
Visualization capabilities.
"""

import cv2
import numpy

def get_layers(net):
    """
    Get the layer names of the network.
    
    :param net: caffe network
    :type net: caffe.Net
    :return: layer names
    :rtype: [string]
    """
    
    return [layer for layer in net.params.keys()]

def visualize_kernels(net, layer, zoom = 5):
    """
    Visualize kernels in the given layer.
    
    :param net: caffe network
    :type net: caffe.Net
    :param layer: layer name
    :type layer: string
    :param zoom: the number of pixels (in width and height) per kernel weight
    :type zoom: int
    :return: image visualizing the kernels in a grid
    :rtype: numpy.ndarray
    """
    
    assert layer in get_layers(net), "layer %s not found" % layer
    
    num_kernels = net.params[layer][0].data.shape[0]
    num_channels = net.params[layer][0].data.shape[1]
    kernel_height = net.params[layer][0].data.shape[2]
    kernel_width = net.params[layer][0].data.shape[3]
    
    image = numpy.zeros((num_kernels*zoom*kernel_height, num_channels*zoom*kernel_width))
    for k in range(num_kernels):
        for c in range(num_channels):
            kernel = net.params[layer][0].data[k, c, :, :]
            kernel = cv2.resize(kernel, (zoom*kernel_height, zoom*kernel_width), kernel, 0, 0, cv2.INTER_NEAREST)
            kernel = (kernel - numpy.min(kernel))/(numpy.max(kernel) - numpy.min(kernel))
            image[k*zoom*kernel_height:(k + 1)*zoom*kernel_height, c*zoom*kernel_width:(c + 1)*zoom*kernel_width] = kernel
    
    return image
    
def visualize_weights(net, layer, zoom = 2):
    """
    Visualize weights in a fully conencted layer.
    
    :param net: caffe network
    :type net: caffe.Net
    :param layer: layer name
    :type layer: string
    :param zoom: the number of pixels (in width and height) per weight
    :type zoom: int
    :return: image visualizing the kernels in a grid
    :rtype: numpy.ndarray
    """
    
    assert layer in get_layers(net), "layer %s not found" % layer
    
    weights = net.params[layer][0].data
    weights = (weights - numpy.min(weights))/(numpy.max(weights) - numpy.min(weights))
    return cv2.resize(weights, (weights.shape[0]*zoom, weights.shape[1]*zoom), weights, 0, 0, cv2.INTER_NEAREST)