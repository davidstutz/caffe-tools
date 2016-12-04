"""
Example for classification on Cifar10 [1]

.. code-block:: python

    [1] A. Krizhevsky.
        Learning Multiple Layers of Features from Tiny Images. 
        2009.

**Note: the LMDBs can also be found in the data repository, see README.**

Use ``caffe/data/cifar10/get_cifar10.sh`` to download Cifar10 and
``caffe/examples/create_cifar10.sh`` to create the corresponding LMDBs.
Copy them over into ``examples/cifar10`` for the following data structure:

.. code-block:: python

    examples/cifar10
    |- train_lmdb/
    |- test_lmdb/

.. argparse::
   :ref: examples.cifar10.get_parser
   :prog: cifar10
"""

import os
import cv2
import glob
import numpy
import argparse
from matplotlib import pyplot

# To silence Caffe! Must be added before importing Caffe or modules which
# are importing Caffe.
os.environ['GLOG_minloglevel'] = '3'
import caffe
import tools.solvers
import tools.lmdb_io
import tools.pre_processing
import tools.prototxt

caffe.set_mode_gpu()

def get_parser():
    """
    Get the parser.
    
    :return: parser
    :rtype: argparse.ArgumentParser
    """
    
    parser = argparse.ArgumentParser(description = 'Caffe example on Cifar-10.')
    parser.add_argument('mode', default = 'train',
                        help = 'mode to run: "train" or "resume"')
    parser.add_argument('--train_lmdb', dest = 'train_lmdb', type = str,
                        help = 'path to train LMDB',
                        default = 'examples/cifar10/train_lmdb')
    parser.add_argument('--test_lmdb', dest = 'test_lmdb', type = str,
                        help = 'path to test LMDB',
                        default = 'examples/cifar10/test_lmdb')
    parser.add_argument('--working_directory', dest = 'working_directory', type = str,
                        help = 'path to a directory (created if not existent) where to store the created .prototxt and snapshot files',
                        default = 'examples/cifar10')
    parser.add_argument('--iterations', dest = 'iterations', type = int,
                        help = 'number of iterations to train or resume',
                        default = 10000)
    parser.add_argument('--image', dest = 'image', type = str,
                        help = 'path to image for testing',
                        default = 'examples/cifar10/test_dog.png')
    return parser

def main_train():
    """
    Train a network on Cifar10 on scratch.
    """
        
    def network(lmdb_path, batch_size):
        """
        The network definition given the LMDB path and the used batch size.
        
        :param lmdb_path: path to LMDB to use (train or test LMDB)
        :type lmdb_path: string
        :param batch_size: batch size to use
        :type batch_size: int
        :return: the network definition as string to write to the prototxt file
        :rtype: string
        """
        
        net = caffe.NetSpec()
        
        net.data, net.labels = caffe.layers.Data(batch_size = batch_size, 
                                                 backend = caffe.params.Data.LMDB, 
                                                 source = lmdb_path, 
                                                 transform_param = dict(scale = 1./255), 
                                                 ntop = 2)

        net.conv1 = caffe.layers.Convolution(net.data, kernel_size = 5, num_output = 32, pad = 2,
                                             stride = 1, weight_filler = dict(type = 'xavier'))
        net.pool1 = caffe.layers.Pooling(net.conv1, kernel_size = 3, stride = 2, 
                                         pool = caffe.params.Pooling.MAX)
        net.relu1 = caffe.layers.ReLU(net.pool1, in_place = True)
        net.norm1 = caffe.layers.LRN(net.relu1, local_size = 3, alpha = 5e-05, 
                                     beta = 0.75, norm_region = caffe.params.LRN.WITHIN_CHANNEL)
        net.conv2 = caffe.layers.Convolution(net.relu1, kernel_size = 5, num_output = 32, pad = 2,
                                             stride = 1, weight_filler = dict(type = 'xavier'))
        net.relu2 = caffe.layers.ReLU(net.conv2, in_place = True)
        net.pool2 = caffe.layers.Pooling(net.relu2, kernel_size = 3, stride = 2,
                                         pool = caffe.params.Pooling.AVE)
        net.norm2 = caffe.layers.LRN(net.pool2, local_size = 3, alpha = 5e-05, beta = 0.75,
                                     norm_region = caffe.params.LRN.WITHIN_CHANNEL)
        net.conv3 = caffe.layers.Convolution(net.norm2, kernel_size = 5, num_output = 64, pad = 2,
                                             stride = 1, weight_filler = dict(type = 'xavier'))
        net.relu3 = caffe.layers.ReLU(net.conv3, in_place = True)
        net.pool3 = caffe.layers.Pooling(net.relu3, kernel_size = 3, stride = 2,
                                         pool = caffe.params.Pooling.AVE)
        net.score = caffe.layers.InnerProduct(net.pool3, num_output = 10)
        net.loss =  caffe.layers.SoftmaxWithLoss(net.score, net.labels)
        
        return net.to_proto()
    
    def count_errors(scores, labels):
        """
        Utility method to count the errors given the ouput of the
        "score" layer and the labels.
        
        :param score: output of score layer
        :type score: numpy.ndarray
        :param labels: labels
        :type labels: numpy.ndarray
        :return: count of errors
        :rtype: int
        """
        
        return numpy.sum(numpy.argmax(scores, axis = 1) != labels) 
        
    train_prototxt_path = args.working_directory + '/train.prototxt'
    test_prototxt_path = args.working_directory + '/test.prototxt'
    deploy_prototxt_path = args.working_directory + '/deploy.prototxt'
    
    with open(train_prototxt_path, 'w') as f:
        f.write(str(network(args.train_lmdb, 128)))
    
    with open(test_prototxt_path, 'w') as f:
        f.write(str(network(args.test_lmdb, 1000)))
    
    tools.prototxt.train2deploy(train_prototxt_path, (1, 3, 32, 32), deploy_prototxt_path)
    
    solver_prototxt_path = args.working_directory + '/solver.prototxt'
    solver_prototxt = tools.solvers.SolverProtoTXT({
        'train_net': train_prototxt_path,
        'test_net': test_prototxt_path,
        'test_initialization': 'false', # no testing
        'test_iter': 0, # no testing
        'test_interval': 1000,
        'base_lr': 0.01,
        'lr_policy': 'inv',
        'gamma': 0.0001,
        'power': 0.75,
        'stepsize': 1000,
        'display': 100,
        'max_iter': 1000,
        'momentum': 0.95,
        'weight_decay': 0.0005,
        'snapshot': 0, # only at the end
        'snapshot_prefix': args.working_directory + '/snapshot',
        'solver_mode': 'CPU'
    })
    
    solver_prototxt.write(solver_prototxt_path)
    solver = caffe.SGDSolver(solver_prototxt_path)      
    callbacks = []
    
    # Callback to report loss in console. Also automatically plots the loss
    # and writes it to the given file. In order to silence the console,
    # use plot_loss instead of report_loss.
    report_loss = tools.solvers.PlotLossCallback(100, args.working_directory + '/loss.png')
    callbacks.append({
        'callback': tools.solvers.PlotLossCallback.report_loss,
        'object': report_loss,
        'interval': 1,
    })
    
    # Callback to report error in console.
    report_error = tools.solvers.PlotErrorCallback(count_errors, 60000, 10000, 
                                                   solver_prototxt.get_parameters()['snapshot_prefix'], 
                                                   args.working_directory + '/error.png')
    callbacks.append({
        'callback': tools.solvers.PlotErrorCallback.report_error,
        'object': report_error,
        'interval': 500,
    })
    
    # Callback to save an "early stopping" model.
    callbacks.append({
        'callback': tools.solvers.PlotErrorCallback.stop_early,
        'object': report_error,
        'interval': 500,
    })
    
    # Callback for reporting the gradients for all layers in the console.
    report_gradient = tools.solvers.PlotGradientCallback(100, args.working_directory + '/gradient.png')
    callbacks.append({
        'callback': tools.solvers.PlotGradientCallback.report_gradient,
        'object': report_gradient,
        'interval': 1,
    })   
    
    # Callback for saving regular snapshots using the snapshot_prefix in the
    # solver prototxt file.
    # Is added after the "early stopping" callback to avoid problems.
    callbacks.append({
        'callback': tools.solvers.SnapshotCallback.write_snapshot,
        'object': tools.solvers.SnapshotCallback(),
        'interval': 500,
    })
    
    monitoring_solver = tools.solvers.MonitoringSolver(solver)
    monitoring_solver.register_callback(callbacks)
    monitoring_solver.solve(args.iterations)

def main_resume():
    """
    Resume training; assumes training has been started using :func:`examples.cifar10.main_train`.
    """
    
    def count_errors(scores, labels):
        """
        Utility method to count the errors given the ouput of the
        "score" layer and the labels.
        
        :param score: output of score layer
        :type score: numpy.ndarray
        :param labels: labels
        :type labels: numpy.ndarray
        :return: count of errors
        :rtype: int
        """
        
        return numpy.sum(numpy.argmax(scores, axis = 1) != labels)   
        
    max_iteration = 0
    files = glob.glob(args.working_directory + '/*.solverstate')
    
    for filename in files:
        filenames = filename.split('_')
        iteration = filenames[-1][:-12]
        
        try:
            iteration = int(iteration)
            if iteration > max_iteration:
                max_iteration = iteration
        except:
            pass
    
    caffemodel = args.working_directory + '/snapshot_iter_' + str(max_iteration) + '.caffemodel'
    solverstate = args.working_directory + '/snapshot_iter_' + str(max_iteration) + '.solverstate'
    
    train_prototxt_path = args.working_directory + '/train.prototxt'
    test_prototxt_path = args.working_directory + '/test.prototxt'
    deploy_prototxt_path = args.working_directory + '/deploy.prototxt'
    solver_prototxt_path = args.working_directory + '/solver.prototxt'
    
    assert max_iteration > 0, "could not find a solverstate or snaphot file to resume"
    assert os.path.exists(caffemodel), "caffemodel %s not found" % caffemodel
    assert os.path.exists(solverstate), "solverstate %s not found" % solverstate
    assert os.path.exists(train_prototxt_path), "prototxt %s not found" % train_prototxt_path
    assert os.path.exists(test_prototxt_path), "prototxt %s not found" % test_prototxt_path
    assert os.path.exists(deploy_prototxt_path), "prototxt %s not found" % deploy_prototxt_path
    assert os.path.exists(solver_prototxt_path), "prototxt %s not found" % solver_prototxt_path
    
    solver = caffe.SGDSolver(solver_prototxt_path)
    solver.restore(solverstate)
    
    solver.net.copy_from(caffemodel)
    
    solver_prototxt = tools.solvers.SolverProtoTXT()
    solver_prototxt.read(solver_prototxt_path)     
    callbacks = []
    
    # Callback to report loss in console.
    report_loss = tools.solvers.PlotLossCallback(100, args.working_directory + '/loss.png')
    callbacks.append({
        'callback': tools.solvers.PlotLossCallback.report_loss,
        'object': report_loss,
        'interval': 1,
    })
    
    # Callback to report error in console.
    report_error = tools.solvers.PlotErrorCallback(count_errors, 60000, 10000, 
                                                   solver_prototxt.get_parameters()['snapshot_prefix'], 
                                                   args.working_directory + '/error.png')
    callbacks.append({
        'callback': tools.solvers.PlotErrorCallback.report_error,
        'object': report_error,
        'interval': 500,
    })
    
    # Callback to save an "early stopping" model.
    callbacks.append({
        'callback': tools.solvers.PlotErrorCallback.stop_early,
        'object': report_error,
        'interval': 500,
    })
    
    # Callback for reporting the gradients for all layers in the console.
    report_gradient = tools.solvers.PlotGradientCallback(100, args.working_directory + '/gradient.png')
    callbacks.append({
        'callback': tools.solvers.PlotGradientCallback.report_gradient,
        'object': report_gradient,
        'interval': 1,
    })
    
    # Callback for saving regular snapshots using the snapshot_prefix in the
    # solver prototxt file.
    # Is added after the "early stopping" callback to avoid problems.
    callbacks.append({
        'callback': tools.solvers.SnapshotCallback.write_snapshot,
        'object': tools.solvers.SnapshotCallback(),
        'interval': 500,
    })  
    
    monitoring_solver = tools.solvers.MonitoringSolver(solver, max_iteration)
    monitoring_solver.register_callback(callbacks)
    monitoring_solver.solve(args.iterations)

def main_test():
    """
    Test the latest model obtained by :func:`examples.cifar10.main_train`
    or :func:`examples.cifar10.main_resume` on the given input image.
    """
    
    max_iteration = 0
    files = glob.glob(args.working_directory + '/*.solverstate')
    
    for filename in files:
        filenames = filename.split('_')
        iteration = filenames[-1][:-12]
        
        try:
            iteration = int(iteration)
            if iteration > max_iteration:
                max_iteration = iteration
        except:
            pass
    
    caffemodel = args.working_directory + '/snapshot_iter_' + str(max_iteration) + '.caffemodel'
    deploy_prototxt_path = args.working_directory + '/deploy.prototxt'
    
    assert max_iteration > 0, "could not find a solverstate or snaphot file to resume"
    assert os.path.exists(caffemodel), "caffemodel %s not found" % caffemodel
    assert os.path.exists(deploy_prototxt_path), "prototxt %s not found" % deploy_prototxt_path
    
    net = caffe.Net(deploy_prototxt_path, caffemodel, caffe.TEST)
    transformer = caffe.io.Transformer({'data': (1, 3, 32, 32)})
    transformer.set_transpose('data', (2, 0, 1))    
    transformer.set_raw_scale('data', 1/255.)
    
    assert os.path.exists(args.image), "image %s not found" % args.image
    image = cv2.imread(args.image)
    cv2.imshow('image', image)
    
    net.blobs['data'].reshape(1, 3, 32, 32)
    net.blobs['data'].data[...] = transformer.preprocess('data', image)    
    
    net.forward()
    scores = net.blobs['score'].data
    
    x = range(10)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    pyplot.bar(x, scores[0, :], 1/1.5, color = 'blue')
    pyplot.xticks(x, classes, rotation = 90)
    pyplot.gcf().subplots_adjust(bottom = 0.2)
    pyplot.show()
    
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    if args.mode == 'train':
        main_train()
    elif args.mode == 'resume':
        main_resume()
    elif args.mode == 'test':
        main_test()
    else:
        print('Invalid mode')