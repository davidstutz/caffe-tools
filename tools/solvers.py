"""
:class:`tools.solvers.MonitoringSolver` is a wrapper for Caffe's solvers allowing
to register various kinds of callbacks.

:class:`tools.solvers.SolverProtoTXT` allows to easily read and write solver
configuration/prototxt files.

The remaining classes are callbacks that can be hooked into :class:`tools.solvers.MonitoringSolver`
allowing to monitor loss, error and gradients in a convenient way.

For detailed usage examples, see :mod:`examples`.
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import multiprocessing
import numpy
import time
import os

class SolverProtoTXT:
    """
    Utility class to create and manage solver prototxt files. Makes life
    with solver prototxt files easier and is the basis for the solvers
    defined below.
    """
    
    _defaults = {
        'test_initialization': 'false',
        'test_iter': 1000,
        'test_interval': 1000,
        'base_lr': 0.01,
        'lr_policy': 'step',
        'gamma': 0.1,
        'stepsize': 100000,
        'display': 20,
        'max_iter': 450000,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'snapshot': 10000,
        'solver_mode': 'GPU',
    }
    """ (dict) Default solver pareters. """
    
    def __init__(self, parameters = {}):
        """
        Constructor based on the solver parameters, if given. See 
        :func:`tools.solvers.SolverProtoTXT.set_parameters` for details.
        
        :param parameters: dictionary defining the parameters
        :type paramters: dict
        """
        
        self._parameters = {}
        """ ({}) Stores the parameters of the solver. """
        
        if parameters:
            self.set_parameters(parameters)
            """ (dict) Solver parameters. """
        
    def get_parameters(self):
        """
        Get the parameters of the solver.
        
        :return: parameters as object
        :rtype: {string: mixed}
        """
        
        return self._parameters
    
    def set_parameters(self, parameters):
        """
        Set the parameters of the solver.
        
        .. code-block:: python
        
            net: "train_val.prototxt"
            test_iter: 1000
            test_interval: 1000
            base_lr: 0.01
            lr_policy: "step"
            gamma: 0.1
            stepsize: 100000
            display: 20
            max_iter: 450000
            momentum: 0.9
            weight_decay: 0.0005
            snapshot: 10000
            snapshot_prefix: "caffenet_train"
            solver_mode: GPU
        
        Note that this class does not support specifying train_net or test_net
        as it assumes the corresponding phases to be specified in the net prototxt 
        files as follows:
        
        .. code-block:: python
        
            layer {
              name: "data"
              type: "Data"
              top: "data"
              top: "label"
              include {
                phase: TRAIN
              }
              data_param {
                source: "train_lmdb"
                batch_size: 256
                backend: LMDB
              }
            }
            layer {
              name: "data"
              type: "Data"
              top: "data"
              top: "label"
              top: "label"
              include {
                phase: TEST
              }
              data_param {
                source: "val_lmdb"
                batch_size: 50
                backend: LMDB
              }
            }
        """
        
        assert ('train_net' in parameters and 'test_net' in parameters) or 'net' in parameters, \
                "the 'train_net' and 'test_net' keys or the 'net' key needs to be specified"
        assert 'snapshot_prefix' in parameters, "the 'snapshot_prefix' key needs to be specified"
        
        for key in self._defaults.keys():
            if not key in parameters:
                parameters[key] = self._defaults[key]
        
        self._parameters = parameters
        
    def write(self, prototxt_path):
        """
        Write the solver parameters to the given path.
        
        :param prototxt_path: path to the prototxt file to write
        :type prototxt_path: string
        """
        
        with open(prototxt_path, 'w') as f:
            for key in self._parameters.keys():
                value = self._parameters[key]
                
                if not 'test_net' in self._parameters and key[0:5] == 'test_':
                    continue
                
                if type(value) is str and key != 'test_initialization' and key != 'solver_mode':
                    f.write(key + ': "' + value + '"\n')
                else:
                    f.write(key + ': ' + str(value) + '\n')
    
    def read(self, prototxt_path):
        """
        Read the solver parameters from the given file.
        
        :param prototxt_path: path to the prototxt file to read
        :type prototxt_path: string
        """
        
        assert os.path.exists(prototxt_path), "could not find prototxt_path"
        
        with open(prototxt_path) as f:
            
            lines = f.read().split('\n')
            for line in lines:
                if line:
                    parts = line.split(':')
                
                    assert len(parts) == 2, "found invalid line in prototxt_path: %s" % line
                    
                    key = parts[0].strip()
                    value = parts[1].strip()                
                    
                    if value.find('"') >= 0 or key == 'solver_mode':
                        self._parameters[key] = str(value.replace('"', ''))
                    elif key == 'test_initialization':
                        self._parameters[key] = str(value)
                    elif key == 'base_lr' or key == 'lr_policy' or key == 'power' \
                            or key == 'momentum' or key == 'weight_decay' or key == 'gamma':
                            
                        self._parameters[key] = float(value)
                    else:
                        self._parameters[key] = int(value)
        
    def __str__(self):
        """
        Print the parameters.
        """
        
        output = ''
        for key, value in self._parameters:
            output += key + ': ' + str(value) + '\n'
        
        return output

class SnapshotCallback:
    """
     Callback to create snapshots for :class:`tools.solvers.MonitoringSolver`.
    """
    
    def write_snapshot(self, iteration, solver):
        """
        Create a snapshot.
        
        :param iteration: current iteration (may deviate from solver iteration)
        :type iteration: int
        :param solver: solver
        :type solver: caffe.Solver
        """
        
        solver.snapshot()
        print('Snapshot [' + str(iteration) + ']')

class LossCallback(object):
    """
    Collection of callbacks for :class:`tools.solvers.MonitoringSolver`
    to report loss.
    """
    
    def __init__(self, report_interval = 100):
        """
        Constructor.
        
        :param report_interval: interval to actually report the loss
        :type report_interval: int
        """
        
        self.report_interval = report_interval
        """ (int) The interval defining how often to report he loss. """
        
        self._iterations = []
        """ ([int]) Iterations. """        
        
        self._losses = []
        """ ([float]) Takes all losses. """
    
    def _get_loss(self, iteration, solver):
        """
        Gets the loss from the training net.
        
        :param iteration: current iteration
        :type iteration: int
        :param solver: solver
        :type solver: caffe.Solver
        """
        
        if len(self._iterations) == 0 or self._iterations[-1] != iteration:
            self._iterations.append(iteration)
            self._losses.append(solver.net.blobs['loss'].data)
    
    def report_loss(self, iteration, solver):
        """
        Report the training loss.
        
        :param iteration: current iteration (may deviate from solver iteration)
        :type iteration: int
        :param solver: solver
        :type solver: caffe.Solver
        """
        
        self._get_loss(iteration, solver)
        
        if iteration%self.report_interval == 0:
            print('Training loss [' + str(self._iterations[-1]) + ']: ' + str(self._losses[-1]))
    
class PlotLossCallback(LossCallback):
    """
    Plot loss callback for :class:`tools.solvers.MonitoringSolver`.
    """
    
    def __init__(self, report_interval = 100, save = '', show = False):
        """
        Constructor.
        
        :param report_interval: interval to actually report the loss
        :type report_interval: int
        :param save: path to file to save the plot as image, does not save the
            plot if save is empty
        :type save: string
        :param show: wether to show the plot
        :type show: bool
        """
        
        super(PlotLossCallback, self).__init__(report_interval)
        
        self.save = save
        """ (string) Where to save the plot as image if not empty. """
        
        self.show = show
        """ (bool) Whether to show the plot. """
        
        manager = multiprocessing.Manager()
        self._iterations = manager.list([])
        """ ([int]) Iterations. """        
        
        self._losses = manager.list([])
        """ ([float]) Takes all losses. """
        
        self._lock = manager.Lock()
        """ (multiprocessing.Lock) Lock for data access. """
        
        self._process = multiprocessing.Process(target = self._plot, args = (self._lock, self._iterations, self._losses))
        """ (multiprocessing.Process) Process for plotting. """          
        
        self._process.start()     
    
    def _get_loss(self, iteration, solver):
        """
        Gets the loss from the training net.
        
        :param iteration: current iteration
        :type iteration: int
        :param solver: solver
        :type solver: caffe.Solver
        """
        
        if len(self._iterations) == 0 or self._iterations[-1] != iteration:
            self._iterations.append(iteration)
            self._losses.append(solver.net.blobs['loss'].data)
    
    def report_loss(self, iteration, solver):
        """
        Report the training loss.
        
        :param iteration: current iteration (may deviate from solver iteration)
        :type iteration: int
        :param solver: solver
        :type solver: caffe.Solver
        """
        
        self._lock.acquire()
        self._get_loss(iteration, solver)
        self._lock.release()
        
        if iteration%self.report_interval == 0:
            print('Training loss [' + str(self._iterations[-1]) + ']: ' + str(self._losses[-1]))
        
    def plot_loss(self, iteration, solver):
        """
        Update the plotted loss.
        
        :param iteration: current iteration (may deviate from solver iteration)
        :type iteration: int
        :param solver: solver
        :type solver: caffe.Solver
        """
        print(iteration)
        self._lock.acquire()
        self._get_loss(iteration, solver)
        self._lock.release()
        
    def _plot(self, lock, iterations, losses):
        """
        Plot the current loss.
        
        :param lock: lock to access plotting data
        :type lock: multiprocessing.Lock()
        :param iterations: iteartions to plot
        :type iterations: [int]
        :param losses: loss to plot
        :type losses: [float]
        """
        
        if self.show:
            plt.ion()
        else:
            plt.ioff()
            
        train_error, = plt.plot([], [], label = 'Train Loss')
        plt.legend(loc = 'upper right')
        
        if self.show:
            plt.pause(0.001)
        
        length = 0
        while True:
            try:
                lock.acquire()
                if len(iterations) > length:
                    train_error.set_xdata(iterations)
                    train_error.set_ydata(losses)
                    
                    if self.show:
                        plt.pause(0.001)
                    
                    plt.gca().relim()
                    plt.gca().autoscale_view()
                    length = len(iterations)
                    
                    if len(self.save) > 0:
                        plt.savefig(self.save)
                        
                lock.release()
                    
                time.sleep(0.1)
            
            except RuntimeError as e:
                #print(e)
                pass
            except IOError as e:
                if e.errno == 32:
                    break
                
                #print(e)
                pass
                
class ErrorCallback(object):
    """
    Collection of callbacks for :class:`tools.solvers.MonitoringSolver`
    to report error.
    
    Uses both the test and train networks of the solver to determine the
    error. Needs that both 'test_net' and 'train_net' are defined in the
    solver prototxt file. These networks need to use different LMDBs.
    As pyCaffe does not offer an interface to access the solver parameters through
    the solver, the LMDB sizes need to be given as constructor paramters.
    """
    
    def __init__(self, count_errors, train_lmdb_size, test_lmdb_size, snapshot_prefix = ''):
        """
        Constructor. Expects the sizes of both train and test LMDBs to be given.
        
        :param count_errors: a function expecting as input the output of the score layer
            and the label of the corresponding image and returning the count of errors, 
            should also work with lists of scores and labels
        :type count_errors: function
        :param train_lmdb_size: size of train LMDB, 0 if training error should
            not be computed
        :type train_lmdb_size: int
        :param test_lmdb_size: size of test LMDB, 0 if test error should
            not be computed
        :type test_lmdb_size: int
        :param snapshot_prefix: snapshot prefix used in the solver configuration/prototxt
            used for early stopping
        :type snapshot_prefix: str
        """
        
        self._count_errors = count_errors
        """ (function) Function for counting errors. """
        
        self._train_lmdb_size = train_lmdb_size
        """ (int) Size of train LMDB. """
        
        self._test_lmdb_size = test_lmdb_size
        """ (int) Size of test LMDB. """             
        
        self._errors = {'train': [], 'test': []}
        """ ({'train': [float], 'test': [float]}) Takes all losses. """
        
        self._snapshot_prefix = snapshot_prefix
        """ (string) Snapshot prefix. """
        
        self._early_stopping = {'min_iteration': 0, 'min_test_error': -1, 'min_train_error': -1}
        """ ({'min_iteration': int, 'mint_test_error': float, 'min_train_error': float}) Statistics for early stopping. """
        
        self._iterations = {'train': [], 'test': []}
        """ ({'train': [int], 'test': [int]}) Iterations. """    
        
    def _copy_weights(self, net_from, net_to):
        """
        Copy weights between networks.
        
        :param net_from: network to copy weights from
        :type net_from: caffe.Net
        :param net_to: network to copy weights to
        :type net_to: caffe.Net
        """
        
        # http://stackoverflow.com/questions/38511503/how-to-compute-test-validation-loss-in-pycaffe
        params = net_from.params.keys()
        for pr in params:
            net_to.params[pr][1] = net_from.params[pr][1]
            net_to.params[pr][0] = net_from.params[pr][0]
            
    def _get_batch_size(self, net):
        """
        Get the batch size used in the network.
        
        :param net: network
        :type net: caffe.Net
        """
        
        return net.blobs['data'].data.shape[0]
        
    def _compute_errors(self, type, iteration, net):
        """
        Compute the error.
        
        :param type: 'train' or 'test'
        :type type: string
        :param iteration: current iteration
        :type iteration: int
        :param net: network
        :type net: caffe.Net
        """
        
        size = self._test_lmdb_size
        if type == 'train':
            size = self._train_lmdb_size
        if size > 0: # only compute error if size is given
            if len(self._errors[type]) == 0 or self._iterations[type][-1] != iteration:
                
                size = self._test_lmdb_size
                if type == 'train':
                    size = self._train_lmdb_size
                    
                batch_size = self._get_batch_size(net)
                number_batches = size//batch_size
                
                errors = 0
                for b in range(number_batches):
                    net.forward()
                    scores = net.blobs['score'].data
                    labels = []
                    
                    if 'labels' in net.blobs:
                        labels = net.blobs['labels'].data
                    
                    if b == number_batches - 1:
                        errors += self._count_errors(scores[0:size%batch_size], labels[0:min(len(labels), size%batch_size)])
                    else:
                        errors += self._count_errors(scores, labels)
                    
                    print('Computing errors ... ' + str(b/float(number_batches)) + '%\r', end = '') # http://stackoverflow.com/questions/4897359/output-to-the-same-line-overwriting-previous-output-python-2-5
                    
                self._iterations[type].append(iteration)
                self._errors[type].append(errors/float(size))
    
    def report_error(self, iteration, solver):
        """
        Report the error.
        
        :param iteration: current iteration (may deviate from solver iteration)
        :type iteration: int
        :param solver: solver
        :type solver: caffe.Solver
        """
        
        self._copy_weights(solver.net, solver.test_nets[0])
        self._compute_errors('train', iteration, solver.net)
        self._compute_errors('test', iteration, solver.test_nets[0])
        
        if len(self._iterations['train']) > 0:
            print('Training error [' + str(self._iterations['train'][-1]) + ']: ' + str(self._errors['train'][-1]))
        if len(self._iterations['test']) > 0:
            print('Testing error [' + str(self._iterations['test'][-1]) + ']: ' + str(self._errors['test'][-1]))
    
    def stop_early(self, iteration, solver):
        """
        Creates a separate snapshot containing the model with minimum test error.
        Does not stop solving, though.
        
        Assumes that the error is always positive or zero.
        
        It may also be wise to put this callback on a different "schedule" compared
        to the snapshot callback.
        
        :param iteration: current iteration (may deviate from solver iteration)
        :type iteration: int
        :param solver: solver
        :type solver: caffe.Solver
        """
        
        assert self._snapshot_prefix, "snapshot prefix not provided"
        
        self._copy_weights(solver.net, solver.test_nets[0])
        self._compute_errors('train', iteration, solver.net)
        self._compute_errors('test', iteration, solver.test_nets[0])
        
        if len(self._iterations['train']) > 0 and len(self._iterations['test']) == len(self._iterations['train']):
            if self._errors['test'][-1] < self._early_stopping['min_test_error'] or self._early_stopping['min_test_error'] < 0:
                self._early_stopping['min_test_error'] = self._errors['test'][-1]
                self._early_stopping['min_train_error'] = self._errors['train'][-1]
                self._early_stopping['min_iteration'] = self._iterations['test'][-1]
                
                solver.snapshot()
                
                caffemodel = self._snapshot_prefix + '_iter_' + str(self._iterations['test'][-1]) + '.caffemodel'
                solverstate = self._snapshot_prefix + '_iter_' + str(self._iterations['test'][-1]) + '.solverstate'
                
                assert os.path.exists(caffemodel), "%s not found" % caffemodel
                assert os.path.exists(solverstate), "%s not found" % solverstate
                
                early_stopping_caffemodel = self._snapshot_prefix + '_early_stopping.caffemodel'
                early_stopping_solverstate = self._snapshot_prefix + '_early_stopping.solverstate'
                
                if os.path.exists(early_stopping_caffemodel):
                    os.remove(early_stopping_caffemodel)
                
                if os.path.exists(early_stopping_solverstate):
                    os.remove(early_stopping_solverstate)
                    
                os.rename(caffemodel, early_stopping_caffemodel)
                os.rename(solverstate, early_stopping_solverstate)
                
                print('Early stopping [' + str(self._iterations['train'][-1]) + ']: ' + str(self._errors['train'][-1]) + ' / ' + str(self._errors['test'][-1]) + ' from [' + str(self._early_stopping['min_iteration']) + ']')
                
class PlotErrorCallback(ErrorCallback):
    """
    Callbacks for :clas:`tools.solvers.MonitoringSolver` for plotting the error 
    in a separate thread. Also allows to save screenshots of the plots now and then.
    """
    
    def __init__(self, count_errors, train_lmdb_size, test_lmdb_size, snapshot_prefix = '', save = '', show = False):
        """
        Constructor. Expects the sizes of both train and test LMDBs to be given.
        
        :param count_errors: a function expecting as input the output of the score layer
            and the label of the corresponding image and returning the count of errors, 
            should also work with lists of scores and labels
        :type count_errors: function
        :param train_lmdb_size: size of train LMDB
        :type train_lmdb_size: int
        :param test_lmdb_size: size of test LMDB
        :type test_lmdb_size: int
        :param snapshot_prefix: snapshot prefix used in the solver configuration/prototxt
            used for early stopping
        :type snapshot_prefix: str
        :param save: path to file to save the plot as image, does not save the
            plot if save is empty
        :type save: string
        :param show: wether to show the plot
        :type show: bool
        """
        
        super(PlotErrorCallback, self).__init__(count_errors, train_lmdb_size, test_lmdb_size, snapshot_prefix)
        
        self.save = save
        """ (string) Where to save the plot as image if not empty. """
        
        self.show = show
        """ (bool) Whether to show the plot. """
        
        manager = multiprocessing.Manager()
        self._errors = {'train': manager.list([]), 'test': manager.list([])}
        """ ({'train': [float], 'test': [float]}) Takes all losses. """
        
        self._iterations = {'train': manager.list([]), 'test': manager.list([])}
        """ ({'train': [int], 'test': [int]}) Iterations. """    
        
        self._lock = manager.Lock()
        """ (multiprocessing.Lock) Underlying lock for _errors and _iterations. """
        
        self._process = multiprocessing.Process(target = self._plot, args = (self._lock, self._iterations, self._errors))
        """ (multiprocessing.Process) Process for plotting. """          
        
        self._process.start()     
    
    def report_error(self, iteration, solver):
        """
        Report the error.
        
        :param iteration: current iteration (may deviate from solver iteration)
        :type iteration: int
        :param solver: solver
        :type solver: caffe.Solver
        """
        
        self._lock.acquire()
        self._copy_weights(solver.net, solver.test_nets[0])
        self._compute_errors('train', iteration, solver.net)
        self._compute_errors('test', iteration, solver.test_nets[0])
        self._lock.release()
        
        if len(self._iterations['train']) > 0:
            print('Training error [' + str(self._iterations['train'][-1]) + ']: ' + str(self._errors['train'][-1]))
        if len(self._iterations['test']) > 0:
            print('Testing error [' + str(self._iterations['test'][-1]) + ']: ' + str(self._errors['test'][-1]))
            
    def plot_error(self, iteration, solver):
        """
        Update the plotted error.
        
        :param iteration: current iteration (may deviate from solver iteration)
        :type iteration: int
        :param solver: solver
        :type solver: caffe.Solver
        """
        
        self._lock.acquire()
        self._copy_weights(solver.net, solver.test_nets[0])
        self._compute_errors('train', iteration, solver.net)
        self._compute_errors('test', iteration, solver.test_nets[0])
        self._lock.release()

    def _plot(self, lock, iterations, errors):
        """
        Plot the current errors.
        
        :param lock: lock to access plotting data
        :type lock: multiprocessing.Lock()
        :param iterations: iteartions to plot
        :type iterations: [int]
        :param errors: errors to plot
        :type errors: [float]
        """
        
        if self.show:
            plt.ion()
        else:
            plt.ioff()
        
        train_error, = plt.plot([], [], label = 'Train Error')
        test_error, = plt.plot([], [], label = 'Test Error')
        plt.legend(loc = 'upper right')
        
        if self.show:
            plt.pause(0.001)
        
        length_train = 0
        length_test = 0
        while True:
            try:
                lock.acquire()
                
                if len(iterations['train']) > length_train:
                    train_error.set_xdata(iterations['train'])
                    train_error.set_ydata(errors['train'])
                    
                    if self.show:
                        plt.pause(0.001)
                        
                    plt.gca().relim()
                    plt.gca().autoscale_view()
                    length_train = len(iterations['train'])
                    
                    if len(self.save) > 0:
                        plt.savefig(self.save)
                    
                lock.release()
                lock.acquire()
                
                if len(iterations['test']) > length_test:
                    test_error.set_xdata(iterations['test'])
                    test_error.set_ydata(errors['test'])
                    
                    if self.show:
                        plt.pause(0.001)
                        
                    plt.gca().relim()
                    plt.gca().autoscale_view()
                    length_test = len(iterations['test'])
                    
                    if len(self.save) > 0:
                        plt.savefig(self.save)
                    
                lock.release()
                
                time.sleep(0.1)
                
            except RuntimeError as e:
                #print(e)
                pass
            except IOError as e:
                if e.errno == 32:
                    break
                
                #print(e)
                pass

class GradientCallback(object):
    """
    Callback for reporting the gradient in :class:`tools.solvers.MonitoringSolver`.
    """

    def __init__(self, report_interval = 100):
        """
        Constructor.
        
        :param report_interval: interval to actually report the loss
        :type report_interval: int
        """
        
        self._report_interval = report_interval
        """ (int) The inverval to actually report the gradients. """
        
        self._gradients = {}
        """ ({string: [float]}) Gradients of each iteration. """
        
        self._iterations = []
        """ ([int]) Iterations. """
        
    def _get_gradient_magnitude(self, iteration, net):
        """
        Computes the overall gradient magnitude across all layers.
        
        :param iteration: current iteration (may deviate from solver iteration)
        :type iteration: int
        :param net: network
        :type net: caffe.Net
        """
        
        if len(self._iterations) == 0 or self._iterations[-1] != iteration:
            params = net.params.keys()
            
            j = 0
            for i in range(len(net.layers)):
                if len(net.layers[i].blobs) > 0:
                    gradient = numpy.sum(numpy.multiply(net.layers[i].blobs[0].diff, net.layers[i].blobs[0].diff)) \
                            + numpy.sum(numpy.multiply(net.layers[i].blobs[1].diff, net.layers[i].blobs[1].diff))
                    
                    key = params[j]
                    if not key in self._gradients.keys():
                        self._gradients[key] = [gradient]
                    else:
                        # append doesn't work here, kind of weird bug with multiprocessing ...
                        self._gradients[key] = self._gradients[key] + [gradient]
                    
                    j += 1
            
            self._iterations.append(iteration)
            
    def report_gradient(self, iteration, solver):
        """
        Report the gradient for all layers.
        
        :param iteration: current iteration (may deviate from solver iteration)
        :type iteration: int
        :param solver: solver
        :type solver: caffe.Solver
        """
        
        self._get_gradient_magnitude(iteration, solver.net)
        
        if iteration%self._report_interval == 0:
            output = ''
            for key in self._gradients.keys():
                if len(self._gradients[key]) > 0:
                    output += str(key) + ' ' + str(self._gradients[key][-1]) + ' | '
            
            print('Gradients [' + str(self._iterations[-1]) + ']: ' + output[0:-3])

class PlotGradientCallback(GradientCallback):
    """
    Callbacks for :clas:`tools.solvers.MonitoringSolver` for plotting the error 
    in a separate thread. Also allows to save screenshots of the plots now and then.
    """
    
    def __init__(self, report_interval = 100, save = '', show = False):
        """
        Constructor. Expects the sizes of both train and test LMDBs to be given.
        
        :param report_interval: interval to actually report the loss
        :type report_interval: int
        :param save: path to file to save the plot as image, does not save the
            plot if save is empty
        :type save: string
        :param show: wether to show the plot
        :type show: bool
        """
        
        super(PlotGradientCallback, self).__init__(report_interval)
        
        self.save = save
        """ (string) Where to save the plot as image if not empty. """
        
        self.show = show
        """ (bool) Whether to show the plot. """
        
        manager = multiprocessing.Manager()
        self._gradients = manager.dict({})
        """ ([{}]) Takes all losses. """
        
        self._iterations = manager.list([])
        """ ([int]) Iterations. """    
        
        self._lock = manager.Lock()
        """ (multiprocessing.Lock) Underlying lock for _errors and _iterations. """
        
        self._process = multiprocessing.Process(target = self._plot, args = (self._lock, self._iterations, self._gradients))
        """ (multiprocessing.Process) Process for plotting. """          
        
        self._process.start()     
    
    def report_gradient(self, iteration, solver):
        """
        Report the gradient for all layers.
        
        :param iteration: current iteration (may deviate from solver iteration)
        :type iteration: int
        :param solver: solver
        :type solver: caffe.Solver
        """
        
        self._lock.acquire()
        self._get_gradient_magnitude(iteration, solver.net)
        self._lock.release()
        
        if iteration%self._report_interval == 0:
            output = ''
            for key in self._gradients.keys():
                if len(self._gradients[key]) > 0:
                    output += str(key) + ' ' + str(self._gradients[key][-1]) + ' | '
            
            print('Gradients [' + str(self._iterations[-1]) + ']: ' + output[0:-3])
    
    def plot_gradient(self, iteration, solver):
        """
        Update the plotted error.
        
        :param iteration: current iteration (may deviate from solver iteration)
        :type iteration: int
        :param solver: solver
        :type solver: caffe.Solver
        """
        
        self._lock.acquire()
        self._get_gradient_magnitude(iteration, solver.net)
        self._lock.release()

    def _plot(self, lock, iterations, gradients):
        """
        Plot the current errors.
        
        :param lock: lock to access plotting data
        :type lock: multiprocessing.Lock()
        :param iterations: iteartions to plot
        :type iterations: [int]
        :param gradients: gradients to plot
        :type gradients: [{}]
        """
        
        if self.show:
            plt.ion()
        else:
            plt.ioff()
        
        length = 0
        while length <= 0:
            lock.acquire()
            length = len(iterations)
            lock.release()
        
        lock.acquire()
        plots = {}
        for key in gradients.keys():
            plots[key], = plt.plot([], [], label = key)
            
        plt.legend(loc = 'upper right')
        lock.release()
        
        if self.show:
            plt.pause(0.001)
        
        while True:
            try:
                lock.acquire()
                
                xy = numpy.array([])
                for key in gradients.keys():
                    plots[key].set_xdata(iterations)
                    plots[key].set_ydata(gradients[key])
                    
                    if xy.size == 0:
                        xy = numpy.vstack(plots[key].get_data()).T
                    else:
                        xy = numpy.vstack((xy, numpy.vstack(plots[key].get_data()).T))
                        
                plt.gca().dataLim.update_from_data_xy(xy, ignore = False)
                plt.gca().autoscale_view()

                if self.show:
                    plt.pause(0.001)                
                
                if len(self.save) > 0:
                    plt.savefig(self.save)
                    
                lock.release()
                
                time.sleep(0.1)
                
            except RuntimeError as e:
                #print(e)
                pass
            except IOError as e:
                if e.errno == 32:
                    break
                
                #print(e)
                pass

class MonitoringSolver:
    """
    Solver wrapper to enhance monitoring capabilities.
    """
    
    def __init__(self, solver, start = 0):
        """
        Constructor, set underlying solver.
        
        :param solver: underlying solver
        :type solver: caffe.Solver
        :param start: start iteration (for resuming training)
        :type start: int
        """
        
        self._solver = solver
        """ (caffe.Solver) Underlying caffe solver. """
        
        self._callbacks = []
        """ ([{'callback': function, 'interval': int}]) Callbacks, see :func:`solvers.MonitoringSolver.register_callback`. """
        
        self._iteration = start
        """ (int) Current iteration. """
        
    def register_callback(self, callbacks):
        """
        Registers a callback that is called every now and then according
        to the specified interval.
        
        :param callbacks: list of callbacks to register, each callback has a function
            corresponding to the actual callback and an interval defining how
            often the callback is called during solving
        :type callbacks: [{'callback': function, 'interval': int}]
        """
        
        for callback in callbacks:
            assert 'callback' in callback, "each callback needs to be a dictionary defining 'callback', 'object' and 'interval'"
            assert 'object' in callback, "each callback needs to be a dictionary defining 'callback', 'object' and 'interval'"
            assert 'interval' in callback, "each callback needs to be a dictionary defining 'callback', 'object' and 'interval'"
        
        self._callbacks = callbacks                
    
    def solve(self, iterations, step = 1, wait = False):
        """
        Run solver using individual solver steps. Monitoring can be
        done at each iteration as specified by the corresponding options.
        
        :param iterations: number of iterations to solve
        :type iterations: int
        :param step: solver steps done at once
        :type step: int
        """
        
        for iteration in range(iterations):
            for callback in self._callbacks:
                if self._iteration%callback['interval'] == 0:
                    callback['callback'](callback['object'], self._iteration, self._solver)
                    
            self._solver.step(step)
            self._iteration += step
        
        for callback in self._callbacks:
            if self._iteration%callback['interval'] == 0:
                callback['callback'](callback['object'], self._iteration, self._solver)
        
        if wait:
            raw_input('Enter to continue ... ')