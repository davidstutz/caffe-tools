"""
Example for classification on MNIST [1].

.. code-block:: python

    [1] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner.
        Gradient-based learning applied to document recognition.
        Proceedings of the IEEE, 86(11), 1998.

**Note: the LMDBs can also be found in the data repository, see README.**

Use ``caffe/data/mnist/get_mnist.sh`` and ``caffe/examples/mnist/create_mnist.sh``
to convert MNIST to LMDBs. Copy the LMDBs to ``examples/mnist`` to get the 
following directory structure:

.. code-block:: python

    examples/mnist/
    |- train_lmdb
    |- test_lmdb
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
import tools.prototxt
import tools.pre_processing

caffe.set_mode_gpu()

def get_parser():
    """
    Get the parser.
    
    :return: parser
    :rtype: argparse.ArgumentParser
    """
    
    parser = argparse.ArgumentParser(description = 'Caffe example on MNIST.')
    parser.add_argument('mode', default = 'train')
    parser.add_argument('--train_lmdb', dest = 'train_lmdb', type = str,
                        help = 'path to train LMDB',
                        default = 'examples/mnist/train_lmdb')
    parser.add_argument('--test_lmdb', dest = 'test_lmdb', type = str,
                        help = 'path to test LMDB',
                        default = 'examples/mnist/test_lmdb')
    parser.add_argument('--working_directory', dest = 'working_directory', type = str,
                        help = 'path to a directory (created if not existent) where to store the created .prototxt and snapshot files',
                        default = 'examples/mnist')
    parser.add_argument('--iterations', dest = 'iterations', type = int,
                        help = 'number of iterations to train or resume',
                        default = 10000)
    parser.add_argument('--image', dest = 'image', type = str,
                        help = 'path to image for testing',
                        default = 'examples/mnist/test_1.png')
                        
    return parser

def main_train():
    """
    Train a network for MNIST from scratch.
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

        net.conv1 = caffe.layers.Convolution(net.data, kernel_size = 5, num_output = 20, 
                                             weight_filler = dict(type = 'xavier'))
        net.pool1 = caffe.layers.Pooling(net.conv1, kernel_size = 2, stride = 2, 
                                         pool = caffe.params.Pooling.MAX)
        net.conv2 = caffe.layers.Convolution(net.pool1, kernel_size = 5, num_output = 50, 
                                             weight_filler = dict(type = 'xavier'))
        net.pool2 = caffe.layers.Pooling(net.conv2, kernel_size = 2, stride = 2, 
                                         pool = caffe.params.Pooling.MAX)
        net.fc1 =   caffe.layers.InnerProduct(net.pool2, num_output = 500, 
                                              weight_filler = dict(type = 'xavier'))
        net.relu1 = caffe.layers.ReLU(net.fc1, in_place = True)
        net.score = caffe.layers.InnerProduct(net.relu1, num_output = 10, 
                                              weight_filler = dict(type = 'xavier'))
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
    
    assert os.path.exists(args.train_lmdb), "LMDB %s not found" % args.train_lmdb
    assert os.path.exists(args.test_lmdb), "LMDB %s not found" % args.test_lmdb
    
    if not os.path.exists(args.working_directory):
        os.makedirs(args.working_directory)
    
    train_prototxt_path = args.working_directory + '/train.prototxt'
    test_prototxt_path = args.working_directory + '/test.prototxt'
    deploy_prototxt_path = args.working_directory + '/deploy.prototxt'
    
    with open(train_prototxt_path, 'w') as f:
        f.write(str(network(args.train_lmdb, 128)))
    
    with open(test_prototxt_path, 'w') as f:
        f.write(str(network(args.test_lmdb, 1000)))
    
    tools.prototxt.train2deploy(train_prototxt_path, (1, 1, 28, 28), deploy_prototxt_path)
    
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

def main_train_augmented():
    """
    Train a network from scratch on augmented MNIST. Augmentation is done on the
    fly and only involves multiplicative Gaussian noise.
    
    Uses the same working directory as :func:`examples.mnist.main_train`, i.e.
    the corresponding snapshots will be overwritten if not changed via
    ``--working_directory``.
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
        net.augmented_data = caffe.layers.Python(net.data, python_param = dict(module = 'tools.layers', layer = 'DataAugmentationMultiplicativeGaussianNoiseLayer'))
        net.augmented_labels = caffe.layers.Python(net.labels, python_param = dict(module = 'tools.layers', layer = 'DataAugmentationDoubleLabelsLayer'))
        
        net.conv1 = caffe.layers.Convolution(net.augmented_data, kernel_size = 5, num_output = 20, 
                                             weight_filler = dict(type = 'xavier'))
        net.pool1 = caffe.layers.Pooling(net.conv1, kernel_size = 2, stride = 2, 
                                         pool = caffe.params.Pooling.MAX)
        net.conv2 = caffe.layers.Convolution(net.pool1, kernel_size = 5, num_output = 50, 
                                             weight_filler = dict(type = 'xavier'))
        net.pool2 = caffe.layers.Pooling(net.conv2, kernel_size = 2, stride = 2, 
                                         pool = caffe.params.Pooling.MAX)
        net.fc1 =   caffe.layers.InnerProduct(net.pool2, num_output = 500, 
                                              weight_filler = dict(type = 'xavier'))
        net.relu1 = caffe.layers.ReLU(net.fc1, in_place = True)
        net.score = caffe.layers.InnerProduct(net.relu1, num_output = 10, 
                                              weight_filler = dict(type = 'xavier'))
        net.loss =  caffe.layers.SoftmaxWithLoss(net.score, net.augmented_labels)
        
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
    
    assert os.path.exists(args.train_lmdb), "LMDB %s not found" % args.train_lmdb
    assert os.path.exists(args.test_lmdb), "LMDB %s not found" % args.test_lmdb
    
    if not os.path.exists(args.working_directory):
        os.makedirs(args.working_directory)
        
    train_prototxt_path = args.working_directory + '/train.prototxt'
    test_prototxt_path = args.working_directory + '/test.prototxt'
    deploy_prototxt_path = args.working_directory + '/deploy.prototxt'
    
    with open(train_prototxt_path, 'w') as f:
        f.write(str(network(args.train_lmdb, 128)))
    
    with open(test_prototxt_path, 'w') as f:
        f.write(str(network(args.test_lmdb, 1000)))
    
    tools.prototxt.train2deploy(train_prototxt_path, (1, 1, 28, 28), deploy_prototxt_path)
    
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
    Resume training a network as started via :func:`examples.mnist.main_train`, 
    :func:`examples.mnist.main_train_augmented` or :func:`examples.mnist.main_train_autoencoder`.
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
        net.augmented_data = caffe.layers.Python(net.data, python_param = dict(module = 'tools.layers', layer = 'DataAugmentationMultiplicativeGaussianNoiseLayer'))
        net.augmented_labels = caffe.layers.Python(net.labels, python_param = dict(module = 'tools.layers', layer = 'DataAugmentationDoubleLabelsLayer'))
        
        net.conv1 = caffe.layers.Convolution(net.augmented_data, kernel_size = 5, num_output = 20, 
                                             weight_filler = dict(type = 'xavier'))
        net.pool1 = caffe.layers.Pooling(net.conv1, kernel_size = 2, stride = 2, 
                                         pool = caffe.params.Pooling.MAX)
        net.conv2 = caffe.layers.Convolution(net.pool1, kernel_size = 5, num_output = 50, 
                                             weight_filler = dict(type = 'xavier'))
        net.pool2 = caffe.layers.Pooling(net.conv2, kernel_size = 2, stride = 2, 
                                         pool = caffe.params.Pooling.MAX)
        net.fc1 =   caffe.layers.InnerProduct(net.pool2, num_output = 500, 
                                              weight_filler = dict(type = 'xavier'))
        net.relu1 = caffe.layers.ReLU(net.fc1, in_place = True)
        net.score = caffe.layers.InnerProduct(net.relu1, num_output = 10, 
                                              weight_filler = dict(type = 'xavier'))
        net.loss =  caffe.layers.SoftmaxWithLoss(net.score, net.augmented_labels)
        
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
    transformer = caffe.io.Transformer({'data': (1, 1, 28, 28)})
    transformer.set_transpose('data', (2, 0, 1))    
    transformer.set_raw_scale('data', 1/255.)
    
    assert os.path.exists(args.image), "image %s not found" % args.image
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = 255 - image
    image.resize((28, 28, 1))
    cv2.imshow('image', image)
    
    net.blobs['data'].reshape(1, 1, 28, 28)
    net.blobs['data'].data[...] = transformer.preprocess('data', image)    
    
    net.forward()
    scores = net.blobs['score'].data
    
    x = range(10)
    pyplot.bar(x, scores[0, :], 1/1.5, color = 'blue')
    pyplot.gcf().subplots_adjust(bottom = 0.2)
    pyplot.show()
    
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    if args.mode == 'train':
        main_train()
    elif args.mode == 'train_augmented':
        main_train_augmented()
    elif args.mode == 'resume':
        main_resume()
    elif args.mode == 'test':
        main_test()
    else:
        print('Invalid mode.')