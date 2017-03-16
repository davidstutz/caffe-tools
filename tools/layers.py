"""
Python layers.
"""

import caffe
import numpy
import random
import tools.data_augmentation

class TestLayer(caffe.Layer):
    """
    A test layer meant for testing purposes which actually does nothing.
    Note, however, to use the force_backward: true option in the net specification
    to enable the backward pass in layers without parameters.
    """

    def setup(self, bottom, top):
        """
        Checks the correct number of bottom inputs.
        
        :param bottom: bottom inputs
        :type bottom: [numpy.ndarray]
        :param top: top outputs
        :type top: [numpy.ndarray]
        """
        
        pass

    def reshape(self, bottom, top):
        """
        Make sure all involved blobs have the right dimension.
        
        :param bottom: bottom inputs
        :type bottom: caffe._caffe.RawBlobVec
        :param top: top outputs
        :type top: caffe._caffe.RawBlobVec
        """
        
        top[0].reshape(bottom[0].data.shape[0], bottom[0].data.shape[1], bottom[0].data.shape[2], bottom[0].data.shape[3])
        
    def forward(self, bottom, top):
        """
        Forward propagation.
        
        :param bottom: bottom inputs
        :type bottom: caffe._caffe.RawBlobVec
        :param top: top outputs
        :type top: caffe._caffe.RawBlobVec
        """
        
        top[0].data[...] = bottom[0].data

    def backward(self, top, propagate_down, bottom):
        """
        Backward pass.
        
        :param bottom: bottom inputs
        :type bottom: caffe._caffe.RawBlobVec
        :param propagate_down:
        :type propagate_down:
        :param top: top outputs
        :type top: caffe._caffe.RawBlobVec
        """
             
        bottom[0].diff[...] = top[0].diff[...]

class DataAugmentationDoubleLabelsLayer(caffe.Layer):
    """
    All data augmentation labels double or quadruple the number of samples per
    batch. This layer is the base layer to double or quadruple the 
    labels accordingly.
    """
        
    def setup(self, bottom, top):
        """
        Checks the correct number of bottom inputs.
        
        :param bottom: bottom inputs
        :type bottom: [numpy.ndarray]
        :param top: top outputs
        :type top: [numpy.ndarray]
        """
        
        self._k = 2

    def reshape(self, bottom, top):
        """
        Make sure all involved blobs have the right dimension.
        
        :param bottom: bottom inputs
        :type bottom: caffe._caffe.RawBlobVec
        :param top: top outputs
        :type top: caffe._caffe.RawBlobVec
        """
        
        if len(bottom[0].shape) == 4:
            top[0].reshape(self._k*bottom[0].data.shape[0], bottom[0].data.shape[1], bottom[0].data.shape[2], bottom[0].data.shape[3])
        elif len(bottom[0].shape) == 3:
            top[0].reshape(self._k*bottom[0].data.shape[0], bottom[0].data.shape[1], bottom[0].data.shape[2])
        elif len(bottom[0].shape) == 2:
            top[0].reshape(self._k*bottom[0].data.shape[0], bottom[0].data.shape[1])
        else:
            top[0].reshape(self._k*bottom[0].data.shape[0])
        
    def forward(self, bottom, top):
        """
        Forward propagation.
        
        :param bottom: bottom inputs
        :type bottom: caffe._caffe.RawBlobVec
        :param top: top outputs
        :type top: caffe._caffe.RawBlobVec
        """
        
        batch_size = bottom[0].data.shape[0]
        if len(bottom[0].shape) == 4:
            top[0].data[0:batch_size, :, :, :] = bottom[0].data
            
            for i in range(self._k - 1):
                top[0].data[(i + 1)*batch_size:(i + 2)*batch_size, :, :, :] = bottom[0].data
        elif len(bottom[0].shape) == 3:
            top[0].data[0:batch_size, :, :] = bottom[0].data
            
            for i in range(self._k - 1):
                top[0].data[(i + 1)*batch_size:(i + 2)*batch_size, :, :] = bottom[0].data
        elif len(bottom[0].shape) == 2:
            top[0].data[0:batch_size, :] = bottom[0].data
            
            for i in range(self._k - 1):
                top[0].data[(i + 1)*batch_size:(i + 2)*batch_size, :] = bottom[0].data
        else:
            top[0].data[0:batch_size] = bottom[0].data
            
            for i in range(self._k - 1):
                top[0].data[(i + 1)*batch_size:(i + 2)*batch_size] = bottom[0].data
            
    def backward(self, top, propagate_down, bottom):
        """
        Backward pass.
        
        :param bottom: bottom inputs
        :type bottom: caffe._caffe.RawBlobVec
        :param propagate_down:
        :type propagate_down:
        :param top: top outputs
        :type top: caffe._caffe.RawBlobVec
        """
             
        pass
    
class DataAugmentationMultiplicativeGaussianNoiseLayer(caffe.Layer):
    """
    Multiplicative Gaussian noise.
    """
    
    def setup(self, bottom, top):
        """
        Checks the correct number of bottom inputs.
        
        :param bottom: bottom inputs
        :type bottom: [numpy.ndarray]
        :param top: top outputs
        :type top: [numpy.ndarray]
        """
        
        pass

    def reshape(self, bottom, top):
        """
        Make sure all involved blobs have the right dimension.
        
        :param bottom: bottom inputs
        :type bottom: caffe._caffe.RawBlobVec
        :param top: top outputs
        :type top: caffe._caffe.RawBlobVec
        """
        
        top[0].reshape(2*bottom[0].data.shape[0], bottom[0].data.shape[1], bottom[0].data.shape[2], bottom[0].data.shape[3])
        
    def forward(self, bottom, top):
        """
        Forward propagation.
        
        :param bottom: bottom inputs
        :type bottom: caffe._caffe.RawBlobVec
        :param top: top outputs
        :type top: caffe._caffe.RawBlobVec
        """
        
        batch_size = bottom[0].data.shape[0]
        top[0].data[0:batch_size, :, :, :] = bottom[0].data
        top[0].data[batch_size:2*batch_size, :, :, :] = tools.data_augmentation.multiplicative_gaussian_noise(bottom[0].data)
        
    def backward(self, top, propagate_down, bottom):
        """
        Backward pass.
        
        :param bottom: bottom inputs
        :type bottom: caffe._caffe.RawBlobVec
        :param propagate_down:
        :type propagate_down:
        :param top: top outputs
        :type top: caffe._caffe.RawBlobVec
        """
             
        pass

class DataAugmentationAdditiveGaussianNoiseLayer(caffe.Layer):
    """
    Additive Gaussian noise.
    """
    
    def setup(self, bottom, top):
        """
        Checks the correct number of bottom inputs.
        
        :param bottom: bottom inputs
        :type bottom: [numpy.ndarray]
        :param top: top outputs
        :type top: [numpy.ndarray]
        """
        
        pass

    def reshape(self, bottom, top):
        """
        Make sure all involved blobs have the right dimension.
        
        :param bottom: bottom inputs
        :type bottom: caffe._caffe.RawBlobVec
        :param top: top outputs
        :type top: caffe._caffe.RawBlobVec
        """
        
        top[0].reshape(2*bottom[0].data.shape[0], bottom[0].data.shape[1], bottom[0].data.shape[2], bottom[0].data.shape[3])
        
    def forward(self, bottom, top):
        """
        Forward propagation.
        
        :param bottom: bottom inputs
        :type bottom: caffe._caffe.RawBlobVec
        :param top: top outputs
        :type top: caffe._caffe.RawBlobVec
        """
        
        batch_size = bottom[0].data.shape[0]
        top[0].data[0:batch_size, :, :, :] = bottom[0].data
        top[0].data[batch_size:2*batch_size, :, :, :] = tools.data_augmentation.additive_gaussian_noise(bottom[0].data)
        
    def backward(self, top, propagate_down, bottom):
        """
        Backward pass.
        
        :param bottom: bottom inputs
        :type bottom: caffe._caffe.RawBlobVec
        :param propagate_down:
        :type propagate_down:
        :param top: top outputs
        :type top: caffe._caffe.RawBlobVec
        """
             
        pass

class DataAugmentationQuadrupleCropsLayer(caffe.Layer):
    """
    Quadruple the data with random crops. Note that this reduces the size of the input
    by (per default) 4 pixels in each dimension.
    """
    
    def setup(self, bottom, top):
        """
        Checks the correct number of bottom inputs.
        
        :param bottom: bottom inputs
        :type bottom: [numpy.ndarray]
        :param top: top outputs
        :type top: [numpy.ndarray]
        """
        
        pass

    def reshape(self, bottom, top):
        """
        Make sure all involved blobs have the right dimension.
        
        :param bottom: bottom inputs
        :type bottom: caffe._caffe.RawBlobVec
        :param top: top outputs
        :type top: caffe._caffe.RawBlobVec
        """
        
        top[0].reshape(2*bottom[0].data.shape[0], bottom[0].data.shape[1], bottom[0].data.shape[2], bottom[0].data.shape[3])
        
    def forward(self, bottom, top):
        """
        Forward propagation.
        
        :param bottom: bottom inputs
        :type bottom: caffe._caffe.RawBlobVec
        :param top: top outputs
        :type top: caffe._caffe.RawBlobVec
        """
        
        batch_size = bottom[0].data.shape[0]
        crop_left = random.randint(0, 4)
        crop_top = random.randint(0, 4)
        top[0].data[0:batch_size, :, :, :] = tools.data_augmentation.crop(bottom[0].data, (crop_left, crop_top, 4 - crop_left, 4 - crop_top))
        
        crop_left = random.randint(0, 4)
        crop_top = random.randint(0, 4)
        top[0].data[batch_size:2*batch_size, :, :, :] = tools.data_augmentation.crop(bottom[0].data, (crop_left, crop_top, 4 - crop_left, 4 - crop_top))
        
        crop_left = random.randint(0, 4)
        crop_top = random.randint(0, 4)
        top[0].data[2*batch_size:3*batch_size, :, :, :] = tools.data_augmentation.crop(bottom[0].data, (crop_left, crop_top, 4 - crop_left, 4 - crop_top))
        
        crop_left = random.randint(0, 4)
        crop_top = random.randint(0, 4)
        top[0].data[3*batch_size:4*batch_size, :, :, :] = tools.data_augmentation.crop(bottom[0].data, (crop_left, crop_top, 4 - crop_left, 4 - crop_top))
        
    def backward(self, top, propagate_down, bottom):
        """
        Backward pass.
        
        :param bottom: bottom inputs
        :type bottom: caffe._caffe.RawBlobVec
        :param propagate_down:
        :type propagate_down:
        :param top: top outputs
        :type top: caffe._caffe.RawBlobVec
        """
             
        pass
    
class ManhattenLoss(caffe.Layer):
    """
    Compute the Manhatten Loss.
    """
    
    def setup(self, bottom, top):
        """
        Checks the correct number of bottom inputs.
        
        :param bottom: bottom inputs
        :type bottom: [numpy.ndarray]
        :param top: top outputs
        :type top: [numpy.ndarray]
        """
            
        if len(bottom) != 2:
            raise Exception('Need two bottom inputs for Manhatten distance.')
        
    def reshape(self, bottom, top):
        """
        Make sure all involved blobs have the right dimension.
        
        :param bottom: bottom inputs
        :type bottom: caffe._caffe.RawBlobVec
        :param top: top outputs
        :type top: caffe._caffe.RawBlobVec
        """
        
        # Check bottom dimensions.
        if bottom[0].count != bottom[1].count:
            raise Exception('Inputs of both bottom inputs have to match.')
        
        # Set shape of diff to input shape.
        self.diff = numpy.zeros_like(bottom[0].data, dtype = numpy.float32)
        
        # Set output dimensions:            
        top[0].reshape(1)
    
    def forward(self, bottom, top):
        """
        Forward propagation, i.e. compute the Manhatten loss.
        
        :param bottom: bottom inputs
        :type bottom: caffe._caffe.RawBlobVec
        :param top: top outputs
        :type top: caffe._caffe.RawBlobVec
        """
        
        scores = bottom[0].data # network output
        labels = bottom[1].data.reshape(scores.shape) # labels
        
        self.diff[...] = (-1)*(scores < labels).astype(int) \
                + (scores > labels).astype(int)
        
        top[0].data[0] = numpy.sum(numpy.abs(scores - labels)) / bottom[0].num
    
    def backward(self, top, propagate_down, bottom):
        """
        Backward pass.
        
        :param bottom: bottom inputs
        :type bottom: caffe._caffe.RawBlobVec
        :param propagate_down:
        :type propagate_down:
        :param top: top outputs
        :type top: caffe._caffe.RawBlobVec
        """
        
        for i in range(2):
            if not propagate_down[i]:
                continue
            
            if i == 0:
                sign = 1
            else:
                sign = -1
            
            # also see the discussion at http://davidstutz.de/pycaffe-tools-examples-and-resources/
            bottom[i].diff[...] = (sign * self.diff * top[0].diff[0] / bottom[i].num).reshape(bottom[i].diff.shape)
