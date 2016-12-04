"""
Example for edge detection on BSDS500 [1]

.. code-block:: python

    [1] P. Arbelaez, M. Maire, C. Fowlkes and J. Malik.
        Contour Detection and Hierarchical Image Segmentation.
        IEEE TPAMI, Vol. 33, No. 5, 2011.

**Note: the LMDBs can also be found in the data repository, see README.**

In order for the example to work, there are two options: Either download the
BSDS500 dataset with CSV ground truths or directly download the corresponding
LMDBs. You can find both in the resources section of the repository.

In either case, the directory structure (after converting the datasets to
LMDBs, if applicable) should look as follows:

.. code-block:: python

    examples/bsds500
    |- csv_groundTruth/
       |- test/
       |- train/
       |- val/
    |- images/
       |- test/
       |- train/
       |- val/
    |- test_lmdb/
    |- train_lmdb/

.. argparse::
   :ref: examples.bsds500.get_parser
   :prog: bsds500
"""

import os
import cv2
import csv
import glob
import numpy
import random
import argparse

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
    
    parser = argparse.ArgumentParser(description = 'Deep learning for edge detection on BSDS500.')
    parser.add_argument('mode', default = 'convert',
                        help = 'Mode to run: "extract", "subsample_test" or "train"')
    parser.add_argument('--working_directory', default = 'examples/bsds500', type = str,
                       help = 'path to the working directory, see documentation of this example')
    parser.add_argument('--train_lmdb', default = 'examples/bsds500/train_lmdb', type = str,
                       help = 'path to train LMDB')
    parser.add_argument('--test_lmdb', default = 'examples/bsds500/test_lmdb', type = str,
                       help = 'path to test LMDB')
    parser.add_argument('--iterations', dest = 'iterations', type = int,
                        help = 'number of iterations to train or resume',
                        default = 10000)
                        
    return parser
   
def csv_read(csv_file, delimiter = ','):
    """
    Read a CSV file into a numpy.ndarray assuming that each row has the same
    number as columns.
    
    :param csv_file: path to CSV file
    :type csv_file: string
    :param delimiter: delimiter between cells
    :type delimiter: string
    :return: CSV contents as Numpy array as float
    :rtype: numpy.ndarray
    """
    
    cols = -1
    array = []
    
    with open(csv_file) as f:
        for cells in csv.reader(f, delimiter = delimiter):
                cells = [cell.strip() for cell in cells if len(cell.strip()) > 0]
                
                if len(cells) > 0:
                    if cols < 0:
                        cols = len(cells)
                    
                    assert cols == len(cells), "CSV file does not contain a consistent number of columns"
                    
                    cells = [float(cell) for cell in cells]
                    array.append(cells)
    
    return numpy.array(array)   
   
def main_extract():
    """
    Extracts train and test samples from the train and test images and ground truth
    in bsds500/csv_groundTruth and bsds500/images. For each positive edge pixels,
    a quadratic patch is extracted. For non-edge pixels, all patches are subsampled
    by only taking 20% of the patches.
    
    It might be beneficial to also run :func:`examples.bsds500.main_subsample_test`
    on the extracted test LMDB for efficient testing during training.
    """
    
    def extract(directory, lmdb_path):
        assert not os.path.exists(lmdb_path), "%s already exists" % lmdb_path

        segmentation_files = [filename for filename in os.listdir(args.working_directory + '/csv_groundTruth/' + directory) if filename[-4:] == '.csv']
        
        lmdb_path = args.working_directory + '/' + directory + '_lmdb'
        lmdb = tools.lmdb_io.LMDB(lmdb_path)
                    
        s = 1
        for segmentation_file in segmentation_files:
            image_file = args.working_directory + '/images/' + directory + '/' + segmentation_file[:-6] + '.jpg'
            image = cv2.imread(image_file)
            segmentation = csv_read(args.working_directory + '/csv_groundTruth/' + directory + '/' + segmentation_file)
            
            inner = segmentation[1:segmentation.shape[0] - 2, 1:segmentation.shape[1] - 2]
            inner_top = segmentation[0:segmentation.shape[0] - 3, 1:segmentation.shape[1] - 2]
            inner_left = segmentation[1:segmentation.shape[0] - 2, 0:segmentation.shape[1] - 3]
            
            segmentation[1:segmentation.shape[0] - 2, 1:segmentation.shape[1] - 2] = numpy.abs(inner - inner_top) + numpy.abs(inner - inner_left)
            
            segmentation[:, :2] = 0
            segmentation[:, segmentation.shape[1] - 3:] = 0
            segmentation[:2, :] = 0
            segmentation[segmentation.shape[0] - 3:, :] = 0
            
            segmentation[segmentation > 0] = 1
            
            images = []
            labels = []
            
            k = 3
            n = 0
            for i in range(k, segmentation.shape[0] - k):
                for j in range(k, segmentation.shape[1] - k):
                    
                    r = random.random()
                    patch = image[i - k:i + k + 1, j - k:j + k + 1, :]
                    
                    if segmentation[i, j] > 0:
                        images.append(patch)
                        labels.append(1)
                    elif r > 0.8:
                        images.append(patch)
                        labels.append(0)
                    
                    n += 1
            
            lmdb.write(images, labels)
            print(str(s) + '/' + str(len(segmentation_files)))
            s += 1
    
    extract('train', args.train_lmdb)
    extract('val', args.test_lmdb)
    
def main_subsample_test():
    """
    Subsample the test LMDB by only taking 5% of the samples. The original test
    LMDB is renamed by appending '_full' and a newtest  is created having the same
    name as the original one.
    """
    
    test_in_lmdb = args.test_lmdb + '_full'
    test_out_lmdb = args.test_lmdb
    
    assert os.path.exists(test_out_lmdb), "LMDB %s not found" % test_out_lmdb
    os.rename(test_out_lmdb, test_in_lmdb)
    
    pp_in = tools.pre_processing.PreProcessingInputLMDB(test_in_lmdb)
    pp_out = tools.pre_processing.PreProcessingOutputLMDB(test_out_lmdb)
    pp = tools.pre_processing.PreProcessingSubsample(pp_in, pp_out, 0.05)
    pp.run()    
    
def main_train():
    """
    After running :func:`examples.bsds500.main_train`, a network can be trained.
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

        net.conv1 = caffe.layers.Convolution(net.data, kernel_size = 3, num_output = 7, 
                                             weight_filler = dict(type = 'xavier'))
        net.bn1 = caffe.layers.BatchNorm(net.conv1)
        net.relu1 = caffe.layers.ReLU(net.bn1, in_place = True)
        net.conv2 = caffe.layers.Convolution(net.relu1, kernel_size = 3, num_output = 21, 
                                             weight_filler = dict(type = 'xavier'))
        net.bn2 = caffe.layers.BatchNorm(net.conv2)
        net.relu2 = caffe.layers.ReLU(net.bn2, in_place = True)
        net.conv3 = caffe.layers.Convolution(net.relu2, kernel_size = 3, num_output = 7,
                                             weight_filler = dict(type = 'xavier'))
        net.bn3 = caffe.layers.BatchNorm(net.conv3)
        net.relu3 = caffe.layers.ReLU(net.bn3, in_place = True)
        net.score = caffe.layers.InnerProduct(net.relu3, num_output = 1, 
                                              weight_filler = dict(type = 'xavier'))
        net.loss =  caffe.layers.SigmoidCrossEntropyLoss(net.score, net.labels)
        
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
    
    assert os.path.exists(args.train_lmdb), "LMDB %s does not exist" % args.train_lmdb
    assert os.path.exists(args.test_lmdb), "LMDB %s does not exist" % args.test_lmdb
    
    train_prototxt_path = args.working_directory + '/train.prototxt'
    test_prototxt_path = args.working_directory + '/test.prototxt'
    deploy_prototxt_path = args.working_directory + '/deploy.prototxt'
    
    with open(train_prototxt_path, 'w') as f:
        f.write(str(network(args.train_lmdb, 1024)))
    
    with open(test_prototxt_path, 'w') as f:
        f.write(str(network(args.test_lmdb, 5000)))
    
    tools.prototxt.train2deploy(train_prototxt_path, (1, 3, 7, 7), deploy_prototxt_path)


    prototxt_solver = args.working_directory + '/solver.prototxt'
    solver_prototxt = tools.solvers.SolverProtoTXT({
        'train_net': train_prototxt_path,
        'test_net': test_prototxt_path,
        'test_initialization': 'false', # no testing
        'test_iter': 0, # no testing
        'test_interval': 100000,
        'base_lr': 0.001,
        'lr_policy': 'step',
        'gamma': 0.01,
        'stepsize': 1000,
        'display': 100,
        'max_iter': 1000,
        'momentum': 0.95,
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

def main_resume():
    """
    Resume training; assumes training has been started using :func:`examples.bsds500.main_train`.
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
    
def main_detect():
    """
    Detect edges on a given image, after training a network using :func:`examples.bsds500.main_train`.
    """
    
    pass

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    if args.mode == 'extract':
        main_extract()
    if args.mode == 'subsample_test':
        main_subsample_test()
    elif args.mode == 'train':
        main_train()
    elif args.mode =='resume':
        main_resume()
    else:
        print('Invalid mode.')