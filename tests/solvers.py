"""
Test solver capabilities in :mod:`tools.solvers`.
"""

import unittest
import random
import shutil
import numpy
import os

os.environ['GLOG_minloglevel'] = '3'
import caffe
import tools.solvers
import tools.lmdb_io

caffe.set_device(0)
caffe.set_mode_gpu()

class TestSolvers(unittest.TestCase):
    """
    Test solver capabilities in :mod:`tools.solvers`.
    """
    
    def _clean(self):
        if os.path.exists(self._train_lmdb_path):
            shutil.rmtree(self._train_lmdb_path)
        
        if os.path.exists(self._test_lmdb_path):
            shutil.rmtree(self._test_lmdb_path)
    
    def setUp(self):
        """
        Set up paths to LMDBs.
        """
        
        self._train_lmdb_path = 'tests/train_lmdb'
        self._test_lmdb_path = 'tests/test_lmdb'
        
        self._clean()
    
    def test_linear(self):
        """
        Test :class:`tools.solvers.MonitoringSolver` using a toy example
        learning a hyperplane.
        """
        
        N = 20000
        N_train = int(0.8*N)
        D = 20
        model = numpy.random.rand(D)*2 - 1
        
        images = []
        labels = []
        
        for n in range(N):            
            image = (numpy.random.rand(D, 1, 1)*2 - 1).astype(numpy.float)
            label = numpy.sign(numpy.dot(numpy.transpose(model), image[:, 0, 0])).astype(int)
            
            images.append(image)
            labels.append(label)
        
        train_lmdb = tools.lmdb_io.LMDB(self._train_lmdb_path)
        train_lmdb.write(images[0:N_train], labels[0:N_train])
        
        test_lmdb = tools.lmdb_io.LMDB(self._test_lmdb_path)
        test_lmdb.write(images[N_train:N], labels[N_train:N])
        
        def network(lmdb_path, batch_size):
            net = caffe.NetSpec()
            
            net.data, net.labels = caffe.layers.Data(batch_size = batch_size, backend = caffe.params.Data.LMDB, 
                                                    source = lmdb_path, ntop = 2)
            net.fc2 = caffe.layers.InnerProduct(net.data, num_output = 1,
                                                  bias_filler = dict(type = 'constant', value = 0.0),
                                                  weight_filler = dict(type = 'xavier', std = 0.1))
            net.score = caffe.layers.TanH(net.fc2)
            net.loss = caffe.layers.EuclideanLoss(net.score, net.labels)
            
            return net.to_proto()
    
        train_prototxt_path = 'tests/train.prototxt'
        test_prototxt_path = 'tests/test.prototxt'
        batch_size = 64
        
        with open(train_prototxt_path, 'w') as f:
            f.write(str(network(self._train_lmdb_path, batch_size)))
        
        with open(test_prototxt_path, 'w') as f:
            f.write(str(network(self._test_lmdb_path, batch_size)))
        
        solver_prototxt_path = 'tests/solver.prototxt'
        solver_prototxt = tools.solvers.SolverProtoTXT({
            'train_net': train_prototxt_path,
            'test_net': test_prototxt_path,
            'test_initialization': 'false', # no testing
            'test_iter': 0, # no testing
            'test_interval': 100000,
            'base_lr': 0.01,
            'lr_policy': 'step',
            'gamma': 0.01,
            'stepsize': 1000,
            'display': 100,
            'max_iter': 1000,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'snapshot': 0, # only at the end
            'snapshot_prefix': 'tests/snapshot_',
            'solver_mode': 'CPU'
        })
        
        solver_prototxt.write(solver_prototxt_path)
        solver = caffe.SGDSolver(solver_prototxt_path)
        
        def count_errors(scores, labels):
            return numpy.sum(numpy.sign(scores).reshape(scores.shape[0]) != labels)        
        
        plot_loss = tools.solvers.PlotLossCallback(100, 'tests/linear_loss.png', False)
        report_loss_callback = {
            'callback': tools.solvers.PlotLossCallback.report_loss,
            'object': plot_loss,
            'interval': 1,
        }
        
        plot_error = tools.solvers.PlotErrorCallback(count_errors, N_train, N - N_train, 'tests/linear_error.png', False)
        report_error_callback = {
            'callback': tools.solvers.PlotErrorCallback.report_error,
            'object': plot_error,
            'interval': 100,
        }
        
        report_gradient_callback = {
            'callback': tools.solvers.GradientCallback.report_gradient,
            'object': tools.solvers.GradientCallback(),
            'interval': 100,
        }            
        
        monitoring_solver = tools.solvers.MonitoringSolver(solver)
        monitoring_solver.register_callback([report_loss_callback,
                                             report_error_callback,
                                             report_gradient_callback])
        monitoring_solver.solve(5*(N//batch_size))
        
    def test_convex(self):
        """
        Test :class:`tools.solvers.MonitoringSolver` using a toy example
        learning a convex set.
        """
        
        N = 20000
        N_train = int(0.8*N)
        D = 20
        
        M = 20
        models = []
        for m in range(M):
            models.append(numpy.random.rand(D)*2 - 1)
        
        images = []
        labels = []
        
        for n in range(N):            
            image = (numpy.random.rand(D, 1, 1)*2 - 1).astype(numpy.float)
            
            label = 1
            for m in range(M):
                if numpy.sign(numpy.dot(numpy.transpose(models[m]), image[:, 0, 0])).astype(int) < 0:
                    label = -1
                    
            images.append(image)
            labels.append(label)
        
        train_lmdb = tools.lmdb_io.LMDB(self._train_lmdb_path)
        train_lmdb.write(images[0:N_train], labels[0:N_train])
        
        test_lmdb = tools.lmdb_io.LMDB(self._test_lmdb_path)
        test_lmdb.write(images[N_train:N], labels[N_train:N])
        
        def network(lmdb_path, batch_size):
            net = caffe.NetSpec()
            
            net.data, net.labels = caffe.layers.Data(batch_size = batch_size, backend = caffe.params.Data.LMDB, 
                                                    source = lmdb_path, ntop = 2)
            net.fc1 = caffe.layers.InnerProduct(net.data, num_output = 3*D, 
                                                bias_filler = dict(type = 'constant', value = 0.0),
                                                weight_filler = dict(type = 'xavier'))
            net.tanh1 = caffe.layers.TanH(net.fc1, in_place = False)
            net.fc2 = caffe.layers.InnerProduct(net.tanh1, num_output = 1,
                                                  bias_filler = dict(type = 'constant', value = 0.0),
                                                  weight_filler = dict(type = 'xavier', std = 0.1))
            net.score = caffe.layers.TanH(net.fc2)
            net.loss = caffe.layers.EuclideanLoss(net.score, net.labels)
            
            return net.to_proto()
    
        train_prototxt_path = 'tests/train.prototxt'
        test_prototxt_path = 'tests/test.prototxt'
        batch_size = 64
        
        with open(train_prototxt_path, 'w') as f:
            f.write(str(network(self._train_lmdb_path, batch_size)))
        
        with open(test_prototxt_path, 'w') as f:
            f.write(str(network(self._test_lmdb_path, batch_size)))
        
        solver_prototxt_path = 'tests/solver.prototxt'
        solver_prototxt = tools.solvers.SolverProtoTXT({
            'train_net': train_prototxt_path,
            'test_net': test_prototxt_path,
            'test_initialization': 'false', # no testing
            'test_iter': 0, # no testing
            'test_interval': 100000,
            'base_lr': 0.01,
            'lr_policy': 'step',
            'gamma': 0.01,
            'stepsize': 1000,
            'display': 100,
            'max_iter': 1000,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'snapshot': 0, # only at the end
            'snapshot_prefix': 'tests/snapshot_',
            'solver_mode': 'CPU'
        })
        
        solver_prototxt.write(solver_prototxt_path)
        solver = caffe.SGDSolver(solver_prototxt_path)
        
        def count_errors(scores, labels):
            return numpy.sum(numpy.sign(scores).reshape(scores.shape[0]) != labels)        
        
        plot_loss = tools.solvers.PlotLossCallback(100, 'tests/convex_loss.png', False)
        report_loss_callback = {
            'callback': tools.solvers.PlotLossCallback.report_loss,
            'object': plot_loss,
            'interval': 1,
        }
        
        plot_error = tools.solvers.PlotErrorCallback(count_errors, N_train, N - N_train, 'tests/convex_error.png', False)
        report_error_callback = {
            'callback': tools.solvers.PlotErrorCallback.report_error,
            'object': plot_error,
            'interval': 100,
        }
        
        report_gradient_callback = {
            'callback': tools.solvers.GradientCallback.report_gradient,
            'object': tools.solvers.GradientCallback(),
            'interval': 100,
        }            
        
        monitoring_solver = tools.solvers.MonitoringSolver(solver)
        monitoring_solver.register_callback([report_loss_callback,
                                             report_error_callback,
                                             report_gradient_callback])
        monitoring_solver.solve(5*(N//batch_size))
    
    def test_non_convex(self):
        """
        Test :class:`tools.solvers.MonitoringSolver` using a toy example
        learning a non-convex set.
        """
        
        N = 20000
        N_train = int(0.8*N)
        D = 20
        
        M = 10
        models = []
        for m in range(M):
            models.append(numpy.random.rand(D)*2 - 1)
        
        conjunctions = []
        for m in range(M):
            conjunctions.append(random.randint(0, 1))
        
        images = []
        labels = []
        
        for n in range(N):            
            image = (numpy.random.rand(D, 1, 1)*2 - 1).astype(numpy.float)
            images.append(image)
            
            label = True
            for m in range(M):
                l = numpy.sign(numpy.dot(numpy.transpose(models[m]), image[:, 0, 0])).astype(int)
                
                if conjunctions[m] == 1:
                    if l < 0:
                        label = label and False
                    else:
                        label = label and True
                else:
                    if l < 0:
                        label = label or False
                    else:
                        label = label or True
            
            if label:
                labels.append(1)
            else:
                labels.append(-1)
        
        train_lmdb = tools.lmdb_io.LMDB(self._train_lmdb_path)
        train_lmdb.write(images[0:N_train], labels[0:N_train])
        
        test_lmdb = tools.lmdb_io.LMDB(self._test_lmdb_path)
        test_lmdb.write(images[N_train:N], labels[N_train:N])
        
        def network(lmdb_path, batch_size):
            net = caffe.NetSpec()
            
            net.data, net.labels = caffe.layers.Data(batch_size = batch_size, backend = caffe.params.Data.LMDB, 
                                                    source = lmdb_path, ntop = 2)
            net.fc1 = caffe.layers.InnerProduct(net.data, num_output = 6*D, 
                                                bias_filler = dict(type = 'gaussian', std = 0.1),
                                                weight_filler = dict(type = 'xavier'))
            net.tanh1 = caffe.layers.TanH(net.fc1, in_place = False)
            net.drop1 = caffe.layers.Dropout(net.tanh1)
            net.fc2 = caffe.layers.InnerProduct(net.drop1, num_output = 3*D, 
                                                bias_filler = dict(type = 'gaussian', std = 0.1),
                                                weight_filler = dict(type = 'xavier'))
            net.tanh2 = caffe.layers.TanH(net.fc2, in_place = False)
            net.fc3 = caffe.layers.InnerProduct(net.tanh2, num_output = 1,
                                                  bias_filler = dict(type = 'constant', value = 0.0),
                                                  weight_filler = dict(type = 'xavier', std = 0.1))
            net.score = caffe.layers.TanH(net.fc3)
            net.loss = caffe.layers.EuclideanLoss(net.score, net.labels)
            
            return net.to_proto()
    
        train_prototxt_path = 'tests/train.prototxt'
        test_prototxt_path = 'tests/test.prototxt'
        batch_size = 128
        
        with open(train_prototxt_path, 'w') as f:
            f.write(str(network(self._train_lmdb_path, batch_size)))
        
        with open(test_prototxt_path, 'w') as f:
            f.write(str(network(self._test_lmdb_path, batch_size)))
        
        solver_prototxt_path = 'tests/solver.prototxt'
        solver_prototxt = tools.solvers.SolverProtoTXT({
            'train_net': train_prototxt_path,
            'test_net': test_prototxt_path,
            'test_initialization': 'false', # no testing
            'test_iter': 0, # no testing
            'test_interval': 100000,
            'base_lr': 0.01,
            'lr_policy': 'step',
            'gamma': 0.01,
            'stepsize': 1000,
            'display': 100,
            'max_iter': 1000,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'snapshot': 0, # only at the end
            'snapshot_prefix': 'tests/snapshot_',
            'solver_mode': 'CPU'
        })
        
        solver_prototxt.write(solver_prototxt_path)
        solver = caffe.SGDSolver(solver_prototxt_path)
        
        def count_errors(scores, labels):
            return numpy.sum(numpy.sign(scores).reshape(scores.shape[0]) != labels)        
        
        plot_loss = tools.solvers.PlotLossCallback(100, 'tests/nonconvex_loss.png', False)
        report_loss_callback = {
            'callback': tools.solvers.PlotLossCallback.report_loss,
            'object': plot_loss,
            'interval': 1,
        }
        
        plot_error = tools.solvers.PlotErrorCallback(count_errors, N_train, N - N_train, 'tests/nonconvex_error.png', False)
        report_error_callback = {
            'callback': tools.solvers.PlotErrorCallback.report_error,
            'object': plot_error,
            'interval': 100,
        }
        
        report_gradient_callback = {
            'callback': tools.solvers.GradientCallback.report_gradient,
            'object': tools.solvers.GradientCallback(),
            'interval': 100,
        }            
        
        monitoring_solver = tools.solvers.MonitoringSolver(solver)
        monitoring_solver.register_callback([report_loss_callback,
                                             report_error_callback,
                                             report_gradient_callback])
        monitoring_solver.solve(10*(N//batch_size), 1)    
    
    def test_autoencoder(self):
        """
        Test :class:`tools.solvers.MonitoringSolver` for auto encoder.
        """
        
        N = 20000        
        N_train = int(0.8*N)
        D = 20
        
        images = []
        for n in range(N):            
            image = (numpy.random.rand(D, 1, 1)*2 - 1).astype(numpy.float)
            images.append(image)
        
        train_lmdb = tools.lmdb_io.LMDB(self._train_lmdb_path)
        train_lmdb.write(images[0:N_train])
        
        test_lmdb = tools.lmdb_io.LMDB(self._test_lmdb_path)
        test_lmdb.write(images[N_train:N])
        
        def network(lmdb_path, batch_size):
            net = caffe.NetSpec()
            
            net.data = caffe.layers.Data(batch_size = batch_size, backend = caffe.params.Data.LMDB, 
                                                    source = lmdb_path, ntop = 1)
            net.labels = caffe.layers.Data(batch_size = batch_size, backend = caffe.params.Data.LMDB,
                                                    source = lmdb_path, ntop = 1)
            net.fc1 = caffe.layers.InnerProduct(net.data, num_output = 3*D, 
                                                bias_filler = dict(type = 'gaussian', std = 0.1),
                                                weight_filler = dict(type = 'xavier'))
            net.score = caffe.layers.InnerProduct(net.fc1, num_output = D,
                                                  bias_filler = dict(type = 'gaussian', std = 0.1),
                                                  weight_filler = dict(type = 'xavier'))
            net.loss = caffe.layers.EuclideanLoss(net.score, net.labels)
            
            return net.to_proto()
    
        train_prototxt_path = 'tests/train.prototxt'
        test_prototxt_path = 'tests/test.prototxt'
        batch_size = 128
        
        with open(train_prototxt_path, 'w') as f:
            f.write(str(network(self._train_lmdb_path, batch_size)))
        
        with open(test_prototxt_path, 'w') as f:
            f.write(str(network(self._test_lmdb_path, batch_size)))
        
        solver_prototxt_path = 'tests/solver.prototxt'
        solver_prototxt = tools.solvers.SolverProtoTXT({
            'train_net': train_prototxt_path,
            'test_net': test_prototxt_path,
            'test_initialization': 'false', # no testing
            'test_iter': 0, # no testing
            'test_interval': 100000,
            'base_lr': 0.01,
            'lr_policy': 'step',
            'gamma': 0.01,
            'stepsize': 1000,
            'display': 100,
            'max_iter': 1000,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'snapshot': 0, # only at the end
            'snapshot_prefix': 'tests/snapshot_',
            'solver_mode': 'CPU'
        })
        
        solver_prototxt.write(solver_prototxt_path)
        solver = caffe.SGDSolver(solver_prototxt_path)
        
        def calc_errors(scores, labels):
            labels = labels.reshape((labels.shape[0], labels.shape[2]))
            return numpy.sum(numpy.multiply(scores - labels, scores - labels))        
        
        plot_loss = tools.solvers.PlotLossCallback(100, 'tests/autoencoder_loss.png', False)
        report_loss_callback = {
            'callback': tools.solvers.PlotLossCallback.report_loss,
            'object': plot_loss,
            'interval': 1,
        }
        
        plot_error = tools.solvers.PlotErrorCallback(calc_errors, N_train, N - N_train, 'tests/autoencoder_error.png', False)
        report_error_callback = {
            'callback': tools.solvers.PlotErrorCallback.report_error,
            'object': plot_error,
            'interval': 100,
        }
        
        report_gradient_callback = {
            'callback': tools.solvers.GradientCallback.report_gradient,
            'object': tools.solvers.GradientCallback(),
            'interval': 100,
        }            
        
        monitoring_solver = tools.solvers.MonitoringSolver(solver)
        monitoring_solver.register_callback([report_loss_callback,
                                             report_error_callback,
                                             report_gradient_callback])
        monitoring_solver.solve(10*(N//batch_size), 1)    
    
    def test_autoencoder_manhatten(self):
        """
        Test :class:`tools.solvers.MonitoringSolver` for auto encoder.
        """
        
        N = 20000        
        N_train = int(0.8*N)
        D = 20
        
        images = []
        for n in range(N):            
            image = (numpy.random.rand(D, 1, 1)*2 - 1).astype(numpy.float)
            images.append(image)
        
        train_lmdb = tools.lmdb_io.LMDB(self._train_lmdb_path)
        train_lmdb.write(images[0:N_train])
        
        test_lmdb = tools.lmdb_io.LMDB(self._test_lmdb_path)
        test_lmdb.write(images[N_train:N])
        
        def network(lmdb_path, batch_size):
            net = caffe.NetSpec()
            
            net.data = caffe.layers.Data(batch_size = batch_size, backend = caffe.params.Data.LMDB, 
                                                    source = lmdb_path, ntop = 1)
            net.labels = caffe.layers.Data(batch_size = batch_size, backend = caffe.params.Data.LMDB,
                                                    source = lmdb_path, ntop = 1)
            net.fc1 = caffe.layers.InnerProduct(net.data, num_output = 3*D, 
                                                bias_filler = dict(type = 'gaussian', std = 0.1),
                                                weight_filler = dict(type = 'xavier'))
            net.score = caffe.layers.InnerProduct(net.fc1, num_output = D,
                                                  bias_filler = dict(type = 'gaussian', std = 0.1),
                                                  weight_filler = dict(type = 'xavier'))
            net.loss = caffe.layers.Python(net.score, net.labels, python_param = dict(module = 'tools.layers', layer = 'ManhattenLoss'))
            
            return net.to_proto()
    
        train_prototxt_path = 'tests/train.prototxt'
        test_prototxt_path = 'tests/test.prototxt'
        batch_size = 128
        
        with open(train_prototxt_path, 'w') as f:
            f.write('force_backward: true\n')
            f.write(str(network(self._train_lmdb_path, batch_size)))
        
        with open(test_prototxt_path, 'w') as f:
            f.write('force_backward: true\n')
            f.write(str(network(self._test_lmdb_path, batch_size)))
        
        solver_prototxt_path = 'tests/solver.prototxt'
        solver_prototxt = tools.solvers.SolverProtoTXT({
            'train_net': train_prototxt_path,
            'test_net': test_prototxt_path,
            'test_initialization': 'false', # no testing
            'test_iter': 0, # no testing
            'test_interval': 100000,
            'base_lr': 0.01,
            'lr_policy': 'step',
            'gamma': 0.01,
            'stepsize': 1000,
            'display': 100,
            'max_iter': 1000,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'snapshot': 0, # only at the end
            'snapshot_prefix': 'tests/snapshot_',
            'solver_mode': 'CPU'
        })
        
        solver_prototxt.write(solver_prototxt_path)
        solver = caffe.SGDSolver(solver_prototxt_path)
        
        def calc_errors(scores, labels):
            labels = labels.reshape((labels.shape[0], labels.shape[2]))
            return numpy.sum(numpy.multiply(scores - labels, scores - labels))        
        
        plot_loss = tools.solvers.PlotLossCallback(100, 'tests/autoencoder_manhatten_loss.png', False)
        report_loss_callback = {
            'callback': tools.solvers.PlotLossCallback.report_loss,
            'object': plot_loss,
            'interval': 1,
        }
        
        plot_error = tools.solvers.PlotErrorCallback(calc_errors, N_train, N - N_train, 'tests/autoencoder_manhatten_error.png', False)
        report_error_callback = {
            'callback': tools.solvers.PlotErrorCallback.report_error,
            'object': plot_error,
            'interval': 100,
        }         
        
        plot_gradient = tools.solvers.PlotGradientCallback('tests/autoencoder_manhatten_gradient.png', False)
        report_gradient_callback = {
            'callback': tools.solvers.PlotGradientCallback.report_gradient,
            'object': plot_gradient,
            'interval': 100,
        }
        
        monitoring_solver = tools.solvers.MonitoringSolver(solver)
        monitoring_solver.register_callback([report_loss_callback,
                                             report_error_callback,
                                             report_gradient_callback])
        monitoring_solver.solve(10*(N//batch_size), 1)
        
    def test_mnist(self):
        """
        Test :class:`tools.solvers.MonitoringSolver` on MNIST. Make sure to copy
        the MNIST LMDbs in the `tests` directory.
        """
        
        def network(lmdb_path, batch_size):
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
            
        train_lmdb_path = 'tests/mnist_train_lmdb'
        test_lmdb_path = 'tests/mnist_test_lmdb'
        train_prototxt_path = 'tests/train.prototxt'
        test_prototxt_path = 'tests/test.prototxt'
        batch_size = 64
        
        with open(train_prototxt_path, 'w') as f:
            f.write(str(network(train_lmdb_path, batch_size)))
        
        with open(test_prototxt_path, 'w') as f:
            f.write(str(network(test_lmdb_path, batch_size)))
        
        solver_prototxt_path = 'tests/solver.prototxt'
        solver_prototxt = tools.solvers.SolverProtoTXT({
            'train_net': train_prototxt_path,
            'test_net': test_prototxt_path,
            'test_initialization': 'true', # no testing
            'test_iter': 100, # no testing
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
            'snapshot_prefix': 'tests/snapshot_',
            'solver_mode': 'CPU'
        })
        
        solver_prototxt.write(solver_prototxt_path)
        solver = caffe.SGDSolver(solver_prototxt_path)
        
        def count_errors(scores, labels):
            return numpy.sum(numpy.argmax(scores, axis = 1) != labels)        
        
        plot_loss = tools.solvers.PlotLossCallback(100, 'tests/mnist_loss.png', False)
        report_loss_callback = {
            'callback': tools.solvers.PlotLossCallback.report_loss,
            'object': plot_loss,
            'interval': 1,
        }
        
        plot_error = tools.solvers.PlotErrorCallback(count_errors, 60000, 10000, 'tests/mnist_error.png', False)
        report_error_callback = {
            'callback': tools.solvers.PlotErrorCallback.report_error,
            'object': plot_error,
            'interval': 100,
        }
        
        report_gradient_callback = {
            'callback': tools.solvers.GradientCallback.report_gradient,
            'object': tools.solvers.GradientCallback(),
            'interval': 100,
        }            
        
        monitoring_solver = tools.solvers.MonitoringSolver(solver)
        monitoring_solver.register_callback([report_loss_callback,
                                             report_error_callback,
                                             report_gradient_callback])
        monitoring_solver.solve(5*(60000//batch_size))
        
    def test_mnist_cnn(self):
        """
        Test :class:`tools.solvers.MonitoringSolver` on MNIST. Make sure to copy
        the MNIST LMDBs in the `tests` directory.
        """
        
        def network(lmdb_path, batch_size):
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
            net.fc2 = caffe.layers.InnerProduct(net.relu1, num_output = 10, 
                                                  weight_filler = dict(type = 'xavier'))
            net.score = caffe.layers.Softmax(net.fc2)
            net.loss =  caffe.layers.MultinomialLogisticLoss(net.score, net.labels)
            
            return net.to_proto()
            
        train_lmdb_path = 'tests/mnist_train_lmdb'
        test_lmdb_path = 'tests/mnist_test_lmdb'
        train_prototxt_path = 'tests/train.prototxt'
        test_prototxt_path = 'tests/test.prototxt'
        batch_size = 64
        
        with open(train_prototxt_path, 'w') as f:
            f.write(str(network(train_lmdb_path, batch_size)))
        
        with open(test_prototxt_path, 'w') as f:
            f.write(str(network(test_lmdb_path, batch_size)))
        
        solver_prototxt_path = 'tests/solver.prototxt'
        solver_prototxt = tools.solvers.SolverProtoTXT({
            'train_net': train_prototxt_path,
            'test_net': test_prototxt_path,
            'test_initialization': 'true', # no testing
            'test_iter': 100, # no testing
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
            'snapshot_prefix': 'tests/snapshot_',
            'solver_mode': 'CPU'
        })
        
        solver_prototxt.write(solver_prototxt_path)
        solver = caffe.SGDSolver(solver_prototxt_path)
        
        def count_errors(scores, labels):
            return numpy.sum(numpy.argmax(scores, axis = 1) != labels)        
        
        plot_loss = tools.solvers.PlotLossCallback(100, 'tests/mnist_cnn_loss.png', False)
        report_loss_callback = {
            'callback': tools.solvers.PlotLossCallback.report_loss,
            'object': plot_loss,
            'interval': 1,
        }
        
        plot_error = tools.solvers.PlotErrorCallback(count_errors, 60000, 10000, 'tests/mnist_cnn_error.png', False)
        report_error_callback = {
            'callback': tools.solvers.PlotErrorCallback.report_error,
            'object': plot_error,
            'interval': 100,
        }
        
        report_gradient_callback = {
            'callback': tools.solvers.GradientCallback.report_gradient,
            'object': tools.solvers.GradientCallback(),
            'interval': 100,
        }            
        
        monitoring_solver = tools.solvers.MonitoringSolver(solver)
        monitoring_solver.register_callback([report_loss_callback,
                                             report_error_callback,
                                             report_gradient_callback])
        monitoring_solver.solve(5*(60000//batch_size))
        
    def test_python_layer(self):
        """
        Test :class:`tools.solvers.MonitoringSolver` using a toy example
        learning a hyperplane.
        """
        
        N = 20000
        N_train = int(0.8*N)
        D = 20
        model = numpy.random.rand(D)*2 - 1
        
        images = []
        labels = []
        
        for n in range(N):            
            image = (numpy.random.rand(D, 1, 1)*2 - 1).astype(numpy.float)
            label = numpy.sign(numpy.dot(numpy.transpose(model), image[:, 0, 0])).astype(int)
            
            images.append(image)
            labels.append(label)
        
        train_lmdb = tools.lmdb_io.LMDB(self._train_lmdb_path)
        train_lmdb.write(images[0:N_train], labels[0:N_train])
        
        test_lmdb = tools.lmdb_io.LMDB(self._test_lmdb_path)
        test_lmdb.write(images[N_train:N], labels[N_train:N])
        
        def network(lmdb_path, batch_size):
            net = caffe.NetSpec()
            
            net.data, net.labels = caffe.layers.Data(batch_size = batch_size, backend = caffe.params.Data.LMDB, 
                                                    source = lmdb_path, ntop = 2)
            net.test1 = caffe.layers.Python(net.data, python_param = dict(module = 'tools.layers', layer = 'TestLayer'))
            net.fc1 = caffe.layers.InnerProduct(net.test1, num_output = 1,
                                                  bias_filler = dict(type = 'constant', value = 0.0),
                                                  weight_filler = dict(type = 'constant', std = 0.1))
            net.score = caffe.layers.TanH(net.fc1)
            net.loss = caffe.layers.EuclideanLoss(net.score, net.labels)
            #net.loss = caffe.layers.Python(net.score, net.labels, python_param = dict(module = 'tools.layers', layer = 'EuclideanLossLayer'))
            
            return net.to_proto()
    
        train_prototxt_path = 'tests/train.prototxt'
        test_prototxt_path = 'tests/test.prototxt'
        batch_size = 64
        
        with open(train_prototxt_path, 'w') as f:
            f.write('force_backward: true\n')
            f.write(str(network(self._train_lmdb_path, batch_size)))
        
        with open(test_prototxt_path, 'w') as f:
            f.write('force_backward: true\n')
            f.write(str(network(self._test_lmdb_path, batch_size)))
        
        solver_prototxt_path = 'tests/solver.prototxt'
        solver_prototxt = tools.solvers.SolverProtoTXT({
            'train_net': train_prototxt_path,
            'test_net': test_prototxt_path,
            'test_initialization': 'false', # no testing
            'test_iter': 0, # no testing
            'test_interval': 100000,
            'base_lr': 0.01,
            'lr_policy': 'step',
            'gamma': 0.01,
            'stepsize': 1000,
            'display': 100,
            'max_iter': 1000,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'snapshot': 0, # only at the end
            'snapshot_prefix': 'tests/snapshot_',
            'solver_mode': 'CPU'
        })
        
        solver_prototxt.write(solver_prototxt_path)
        solver = caffe.SGDSolver(solver_prototxt_path)
        
        def count_errors(scores, labels):
            return numpy.sum(numpy.sign(scores).reshape(scores.shape[0]) != labels)        
        
        plot_loss = tools.solvers.PlotLossCallback(100, 'tests/python_layer_loss.png', False)
        report_loss_callback = {
            'callback': tools.solvers.PlotLossCallback.report_loss,
            'object': plot_loss,
            'interval': 1,
        }
        
        plot_error = tools.solvers.PlotErrorCallback(count_errors, N_train, N - N_train, 'tests/python_layer_error.png', False)
        report_error_callback = {
            'callback': tools.solvers.PlotErrorCallback.report_error,
            'object': plot_error,
            'interval': 100,
        }
        
        report_gradient_callback = {
            'callback': tools.solvers.GradientCallback.report_gradient,
            'object': tools.solvers.GradientCallback(),
            'interval': 100,
        }            
        
        monitoring_solver = tools.solvers.MonitoringSolver(solver)
        monitoring_solver.register_callback([report_loss_callback,
                                             report_error_callback,
                                             report_gradient_callback])
        monitoring_solver.solve(5*(N//batch_size))
    
    def tearDown(self):
        """
        Delete temporary LDMBs.
        """
        
        self._clean()

if __name__ == '__main__':
    unittest.main()