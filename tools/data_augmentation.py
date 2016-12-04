"""
Data augmentation methods. These methods are used in :mod:`tools.layers` and
assume that the data is shaped according to Caffe (i.e. batch, height, width, channels)
and some data augmentation methods assume float data.
"""

import cv2
import numpy

def multiplicative_gaussian_noise(images, std = 0.05):
    """
    Multiply with Gaussian noise.
    
    :param images: images (or data) in Caffe format (batch_size, height, width, channels)
    :type images: numpy.ndarray
    :param std: standard deviation of Gaussian
    :type std: float
    :return: images (or data) with multiplicative Gaussian noise
    :rtype: numpy.ndarray
    """
    
    assert images.ndim == 4
    assert images.dtype == numpy.float32
    
    return numpy.multiply(images, numpy.random.randn(images.shape[0], images.shape[1], images.shape[2], images.shape[3])*std + 1)
    
def additive_gaussian_noise(images, std = 0.05):
    """
    Add Gaussian noise to the images.
    
    :param images: images (or data) in Caffe format (batch_size, height, width, channels)
    :type images: numpy.ndarray
    :param std: standard deviation of Gaussian
    :type std: float
    :return: images (or data) with additive Gaussian noise
    :rtype: numpy.ndarray
    """
    
    assert images.ndim == 4
    assert images.dtype == numpy.float32
    
    return images + numpy.random.randn(images.shape[0], images.shape[1], images.shape[2], images.shape[3])*std

def crop(images, crop):
    """
    Crop the images along all dimensions.
    
    :param images: images (or data) in Caffe format (batch_size, height, width, channels)
    :type images: numpy.ndarray
    :param crop: cropy in (crop_left, crop_top, crop_right, crop_bottom)
    :type crop: (int, int, int, int)
    :return: images (or data) cropped
    :rtype: numpy.ndarray
    """
    
    assert images.ndim == 4
    assert images.dtype == numpy.float32
    assert len(crop) == 4
    assert crop[0] >= 0 and crop[1] >= 0 and crop[2] >= 0 and crop[3] >= 0
    assert crop[0] + crop[2] <= images.shape[2]
    assert crop[1] + crop[3] <= images.shape[1]
    
    return images[:, crop[1]:images.shape[1] - crop[3], crop[0]:images.shape[2] - crop[2], :]

def flip(images):
    """
    Flip the images horizontally.
    
    :param images: images (or data) in Caffe format (batch_size, height, width, channels)
    :type images: numpy.ndarray
    :return: images (or data) flipped horizontally
    :rtype: numpy.ndarray
    """
    
    pass

def drop_color_gaussian(images, channel, mean = 0.5, std = 0.05):
    """
    Drop the specified color channel and replace by Gaussian noise with given mean
    and standard deviation.
    
    :param images: images (or data) in Caffe format (batch_size, height, width, channels)
    :type images: numpy.ndarray
    :param channel: channel to drop
    :type channel: int
    :param mean: mean of Gaussian noise
    :type mean: float
    :param std: standard deviation of Gaussian noise
    :type std: float
    :return: images (or data) dropped channel
    :rtype: numpy.ndarray
    """
    
    assert images.ndim == 4
    assert images.dtype == numpy.float32
    assert images.shape[3] == 3
    
    channels = []
    for i in range(images.shape[3]):
        if i == channel:
            channels.append(numpy.random.randn(images.shape[0], images.shape[1], images.shape[3], 1))
        else:
            channels.append(images[:, :, :, i].reshape(images.shape[0], images.shape[1], images.shape[3], 1))
    
    return numpy.concatenate(tuple(channels), axis = 3)

def scaling_artifacts(image, factor = .5, interpolation = cv2.INTER_LINEAR):
    """
    Introduce scaling articacts by downscaling to the given factor and upscaling again.
    
    :param images: images (or data) in Caffe format (batch_size, height, width, channels)
    :type images: numpy.ndarray
    :param factor: factor to downsample by
    :type factor: float
    :param interpolation: interpolation to use, see OpenCV documentation for resize
    :type interpolation: int
    :return: images (or data) with scaling artifacts
    :rtype: numpy.ndarray
    """
    
    pass
    
def contrast(images, exponent):
    """
    Apply contrast transformation.
    
    :param images: images (or data) in Caffe format (batch_size, height, width, channels)
    :type images: numpy.ndarray
    :param exponent: exponentfor contrast transformation
    :type exponent: float
    :return: images (or data) with contrast normalization
    :rtype: numpy.ndarray
    """
    
    return numpy.power(images, exponent)