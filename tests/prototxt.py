"""
Tests for :mod:`tools.prototxt`.
"""

import tools.prototxt
import unittest
import caffe

class TestPrototxt(unittest.TestCase):
    """
    Tests for :mod:`tools.prototxt`.
    """
    
    def test_train2deploy(self):
        """
        Test train to deploy conversion.
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
            
        train_prototxt_path = 'tests/train.prototxt'
        deploy_prototxt_path = 'tests/deploy.prototxt'
        
        with open(train_prototxt_path, 'w') as f:
            f.write(str(network('tests/train_lmdb', 128)))
        
        tools.prototxt.train2deploy(train_prototxt_path, (128, 3, 28, 28), deploy_prototxt_path)
        
if __name__ == '__main__':
    unittest.main()