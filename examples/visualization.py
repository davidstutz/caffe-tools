"""
Visualize trained network weights.

.. argparse::
   :ref: examples.visualization.get_parser
   :prog: visualization
"""

import os
import cv2
import numpy
import argparse
from matplotlib import pyplot

# To silence Caffe! Must be added before importing Caffe or modules which
# are importing Caffe.
os.environ['GLOG_minloglevel'] = '3'
import caffe
import tools.visualization

caffe.set_mode_gpu()

def get_parser():
    """
    Get the parser.
    
    :return: parser
    :rtype: argparse.ArgumentParser
    """
    
    parser = argparse.ArgumentParser(description = 'Caffe example on Cifar-10.')
    parser.add_argument('--prototxt', dest = 'prototxt', type = str,
                        help = 'path to the prototxt network definition',
                        default = 'examples/cifar10/train_lmdb')
    parser.add_argument('--caffemodel', dest = 'caffemodel', type = str,
                        help = 'path to the caffemodel',
                        default = 'examples/cifar10/test_lmdb')
    parser.add_argument('--output', dest = 'output', type = str,
                        help = 'output directory for visualizations',
                        default = 'examples/output')
    
    return parser

def main():
    """
    Visualize weights of the network.
    """
    
    assert os.path.exists(args.prototxt), "prototxt %s not found" % args.prototxt
    assert os.path.exists(args.caffemodel), "caffemodel %s not found" % args.caffemodel
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    layers = tools.visualization.get_layers(net)
    
    for layer in layers:
        if layer.find('conv') >= 0:
            kernels = tools.visualization.visualize_kernels(net, layer, 5)
            
            cv2.imwrite(args.output + '/' + layer + '.png', (kernels*255).astype(numpy.uint8))
            #pyplot.imshow(kernels, interpolation = 'none')
            #pyplot.colorbar()
            #pyplot.savefig(args.output + '/' + layer + '.png')
            #pyplot.clf()
            
        elif layer.find('fc') >= 0 or layer.find('score') >= 0:
            weights = tools.visualization.visualize_weights(net, layer, 5)
            
            cv2.imwrite(args.output + '/' + layer + '.png', (weights*255).astype(numpy.uint8))
            #pyplot.imshow(weights, interpolation = 'none')
            #pyplot.colorbar()
            #pyplot.savefig(args.output + '/' + layer + '.png')
            #pyplot.clf()
        
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    main()