"""
Tests for reading and writing LMDBs using :class:`tools.lmdb.LMDB`.

In order to run the tests, the MNIST LMDB created by Caffe is created.
Change to the caffe root directory and run ``./data/mnist/get_mnist.sh`` 
as well as ``./examples/mnist/create_mnist.sh``. Then copy 
``examples/mnist/mnist_test_lmdb`` to this folder.

Do the same for Cifar10!
"""

import tools.lmdb_io
import unittest
import shutil
import numpy
import cv2
import os

class TestLMDB(unittest.TestCase):
    """
    Tests for :class:`tools.lmdb.LMDB`.
    """
    
    def test_keys_mnist(self):
        """
        Tests reading the keys from the MNIST LMDB.
        """
        
        lmdb_path = 'tests/mnist_test_lmdb'
        lmdb = tools.lmdb_io.LMDB(lmdb_path)
        
        i = 0
        for key in lmdb.keys():
            self.assertEqual(key, '{:08}'.format(i))
            i += 1
    
    def test_read_mnist(self):
        """
        Tests reading from the MNIST LMDB.
        """
        
        lmdb_path = 'tests/mnist_test_lmdb'
        lmdb = tools.lmdb_io.LMDB(lmdb_path)
        
        keys = lmdb.keys(5)
        for key in keys:
            image, label, key = lmdb.read(key)
            
            image_path = 'tests/mnist_test/' + key + '.png'
            assert os.path.exists(image_path)            
            
            image = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    self.assertEqual(image[i, j], image[i, j])
    
    def test_keys_cifar(self):
        """
        Tests reading the keys from the cifar10 LMDB.
        """
        
        lmdb_path = 'tests/cifar10_test_lmdb'
        lmdb = tools.lmdb_io.LMDB(lmdb_path)
        
        i = 0
        for key in lmdb.keys():
            self.assertEqual(key, '{:05}'.format(i))
            i += 1
            
    def test_read_cifar(self):
        """
        Tests reading from the cifar10 LMDB.
        """
        
        lmdb_path = 'tests/cifar10_test_lmdb'
        lmdb = tools.lmdb_io.LMDB(lmdb_path)
        
        keys = lmdb.keys(5)
        for key in keys:
            image, label, key = lmdb.read(key)
            
            image_path = 'tests/cifar10_test/' + key + '.png'
            assert os.path.exists(image_path)
            
            image = cv2.imread(image_path)
            
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    for c in range(image.shape[2]):
                        self.assertEqual(image[i, j, c], image[i, j, c])
        
    def test_write_read_random(self):
        """
        Tests writing and reading on sample images.
        """
        
        lmdb_path = 'tests/test_lmdb'
        lmdb = tools.lmdb_io.LMDB(lmdb_path)
        
        write_images = [(numpy.random.rand(10, 10, 3)*255).astype(numpy.uint8)]*10
        write_labels = [0]*10
        
        lmdb.write(write_images, write_labels)
        read_images, read_labels, read_keys = lmdb.read()
        
        for n in range(10):
            for i in range(10):
                for j in range(10):
                    for c in range(3):
                        self.assertEqual(write_images[n][i, j, c], read_images[n][i, j, c])
            
            self.assertEqual(write_labels[n], read_labels[n])
        
        if os.path.exists(lmdb_path):
            shutil.rmtree(lmdb_path)
    
    def test_write_read_random_float(self):
        """
        Tests writing and reading on sample images.
        """
        
        lmdb_path = 'tests/test_lmdb'
        lmdb = tools.lmdb_io.LMDB(lmdb_path)
        
        write_images = [numpy.random.rand(10, 10, 3).astype(numpy.float)]*10
        write_labels = [0]*10
        
        lmdb.write(write_images, write_labels)
        read_images, read_labels, read_keys = lmdb.read()
        
        for n in range(10):
            for i in range(10):
                for j in range(10):
                    for c in range(3):
                        self.assertAlmostEqual(write_images[n][i, j, c], read_images[n][i, j, c])
            
            self.assertEqual(write_labels[n], read_labels[n])
        
        if os.path.exists(lmdb_path):
            shutil.rmtree(lmdb_path)
        
if __name__ == '__main__':
    unittest.main()