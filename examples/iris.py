"""
Example for classification on Iris.

**Note: the LMDBs can also be found in the data repository, see README.**

To acquire the dataset, follow http://archive.ics.uci.edu/ml/datasets/Iris.
The downloaded dataset should be saved in ``examples/iris/iris.data.txt``.
:func:`examples.iris.main_convert` will then convert data to LMDBs to obtain the
following data structure:

.. code-block:: python

    examples/iris/
    |- test_lmdb
    |- train_lmdb
    |- iris.data.txt

.. argparse::
   :ref: examples.iris.get_parser
   :prog: iris
"""

import os
import numpy
import shutil
import argparse

# To silence Caffe! Must be added before importing Caffe or modules which
# are importing Caffe.
os.environ['GLOG_minloglevel'] = '3'
import caffe
import tools.solvers
import tools.lmdb_io
import tools.pre_processing

def get_parser():
    """
    Get the parser.
    
    :return: parser
    :rtype: argparse.ArgumentParser
    """
    
    parser = argparse.ArgumentParser(description = 'Deep learning for Iris.')
    parser.add_argument('mode', default = 'convert')
    parser.add_argument('--file', default = 'examples/iris/iris.data.txt', type = str,
                       help = 'path to the iris data file')
    parser.add_argument('--split', default = 0.8, type = float,
                        help = 'fraction of samples to use for taining')
    parser.add_argument('--train_lmdb', default = 'examples/iris/train_lmdb', type = str,
                       help = 'path to train LMDB')
    parser.add_argument('--test_lmdb', default = 'examples/iris/test_lmdb', type = str,
                       help = 'path to test LMDB')
    parser.add_argument('--working_directory', dest = 'working_directory', type = str,
                        help = 'path to a directory (created if not existent) where to store the created .prototxt and snapshot files',
                        default = 'examples/mnist')
                       
    return parser
   
def main_convert():
    """
    Convert the Iris dataset to LMDB.
    """
        
    lmdb_converted = args.working_directory + '/lmdb_converted'
    lmdb_shuffled = args.working_directory + '/lmdb_shuffled'
    
    if os.path.exists(lmdb_converted):
        shutil.rmtree(lmdb_converted)
    if os.path.exists(lmdb_shuffled):
        shutil.rmtree(lmdb_shuffled)
        
    assert os.path.exists(args.file), "file %s could not be found" % args.file
    assert not os.path.exists(args.train_lmdb), "LMDB %s already exists" % args.train_lmdb
    assert not os.path.exists(args.test_lmdb), "LMDB %s already exists" % args.test_lmdb
    
    pp_in = tools.pre_processing.PreProcessingInputCSV(args.file, delimiter = ',', 
                                                       label_column = 4, 
                                                       label_column_mapping = {
                                                           'Iris-setosa': 0, 
                                                           'Iris-versicolor': 1, 
                                                           'Iris-virginica': 2
                                                       })
    pp_out_converted = tools.pre_processing.PreProcessingOutputLMDB(lmdb_converted)
    pp_convert = tools.pre_processing.PreProcessingNormalize(pp_in, pp_out_converted, 7.9)
    pp_convert.run()    
    
    pp_in_converted = tools.pre_processing.PreProcessingInputLMDB(lmdb_converted)
    pp_out_shuffled = tools.pre_processing.PreProcessingOutputLMDB(lmdb_shuffled)
    pp_shuffle = tools.pre_processing.PreProcessingShuffle(pp_in_converted, pp_out_shuffled)
    pp_shuffle.run()
    
    pp_in_shuffled = tools.pre_processing.PreProcessingInputLMDB(lmdb_shuffled)
    pp_out_train = tools.pre_processing.PreProcessingOutputLMDB(args.train_lmdb)
    pp_out_test = tools.pre_processing.PreProcessingOutputLMDB(args.test_lmdb)
    pp_split = tools.pre_processing.PreProcessingSplit(pp_in_shuffled, (pp_out_train, pp_out_test), (0.9, 0.1))
    pp_split.run()
    
    # to make sure
    print('Train:')
    lmdb = tools.lmdb_io.LMDB(args.train_lmdb)
    images, labels, keys = lmdb.read()
    
    for n in range(len(images)):
        print images[n].reshape((4)), labels[n]
    
    print('Test:')
    lmdb = tools.lmdb_io.LMDB(args.test_lmdb)
    images, labels, keys = lmdb.read()
    
    for n in range(len(images)):
        print(images[n].reshape((4)), labels[n])

def main_train():
    """
    Train a network from scratch on Iris using data augmentaiton to get more
    training samples.
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
        net.data, net.labels = caffe.layers.Data(batch_size = batch_size, backend = caffe.params.Data.LMDB, 
                                                source = lmdb_path, ntop = 2)
        net.data_aug = caffe.layers.Python(net.data, 
                                           python_param = dict(module = 'tools.layers', layer = 'DataAugmentationRandomMultiplicativeNoiseLayer'))
        net.labels_aug = caffe.layers.Python(net.labels,
                                             python_param = dict(module = 'tools.layers', layer = 'DataAugmentationDuplicateLabelsLayer'))
        net.fc1 = caffe.layers.InnerProduct(net.data_aug, num_output = 12,
                                            bias_filler = dict(type = 'xavier', std = 0.1),
                                            weight_filler = dict(type = 'xavier', std = 0.1))
        net.sigmoid1 = caffe.layers.Sigmoid(net.fc1)
        net.fc2 = caffe.layers.InnerProduct(net.sigmoid1, num_output = 3,
                                            bias_filler = dict(type = 'xavier', std = 0.1),
                                            weight_filler = dict(type = 'xavier', std = 0.1))
        net.score = caffe.layers.Softmax(net.fc2)
        net.loss = caffe.layers.MultinomialLogisticLoss(net.score, net.labels_aug)
        
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
    
    prototxt_train = args.working_directory + '/train.prototxt'
    prototxt_test = args.working_directory + '/test.prototxt'
    
    with open(prototxt_train, 'w') as f:
        f.write(str(network(args.train_lmdb, 6)))
        
    with open(prototxt_test, 'w') as f:
        f.write(str(network(args.test_lmdb, 6)))
    
    prototxt_solver = args.lmdb + '_solver.prototxt'
    solver_prototxt = tools.solvers.SolverProtoTXT({
        'train_net': prototxt_train,
        'test_net': prototxt_test,
        'test_initialization': 'false', # no testing
        'test_iter': 0, # no testing
        'test_interval': 100000,
        'base_lr': 0.001,
        'lr_policy': 'step',
        'gamma': 0.01,
        'stepsize': 1000,
        'display': 100,
        'max_iter': 1000,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'snapshot': 0, # only at the end
        'snapshot_prefix': args.working_directory + '/snapshot',
        'solver_mode': 'CPU'
    })
    
    solver_prototxt.write(prototxt_solver)
    solver = caffe.SGDSolver(prototxt_solver)
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

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    if args.mode == 'convert':
        main_convert()
    elif args.mode == 'train':
        main_train()
    else:
        print('Invalid mode.')