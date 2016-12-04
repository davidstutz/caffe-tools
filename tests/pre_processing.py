"""
Tests for :mod:`tools.pre_processing`.
"""

import tools.pre_processing
import tools.lmdb_io
import unittest
import random
import shutil
import numpy
import cv2
import os

class TestPreProcessing(unittest.TestCase):
    """
    Tests for :mod:`tools.pre_processing`.
    """
    
    def test_lmdb_input(self):
        """
        Test LMDB input to pre processing.
        """
        
        N = 27
        H = 10
        W = 10
        C = 3
        
        images = []
        labels = []
        
        for n in range(N):
            image = (numpy.random.rand(H, W, C)*255).astype(numpy.uint8)
            label = random.randint(0, 1000)
            
            images.append(image)
            labels.append(label)
        
        
        lmdb_path = 'tests/test_lmdb'
        if os.path.exists(lmdb_path):
            shutil.rmtree(lmdb_path)
            
        lmdb = tools.lmdb_io.LMDB(lmdb_path)
        lmdb.write(images, labels)
        
        pp_in = tools.pre_processing.PreProcessingInputLMDB(lmdb_path)
        self.assertEqual(pp_in.count(), 27)
        
        n = 0
        for i in range(3):
            read_images, read_labels = pp_in.read(10)
            self.assertEqual(len(read_images), len(read_labels))
            
            for j in range(len(read_images)):
                for ii in range(H):
                    for jj in range(W):
                        for cc in range(C):
                            self.assertEqual(images[n][ii, jj, cc], read_images[j][ii, jj, cc])
                
                self.assertEqual(labels[n], read_labels[j])
                n += 1
        
        self.assertTrue(pp_in.end())
    
    def test_files_input(self):
        """
        Test LMDB input to pre processing.
        """
        
        N = 27
        H = 10
        W = 10
        C = 3
        
        files = []
        images = []
        labels = []
        
        pos_path = 'tests/test_pos'
        neg_path = 'tests/test_neg'
        
        if os.path.exists(pos_path):
            shutil.rmtree(pos_path)
        
        if os.path.exists(neg_path):
            shutil.rmtree(neg_path)
            
        os.mkdir(pos_path)
        os.mkdir(neg_path)
        
        for n in range(N):
            image = (numpy.random.rand(H, W, C)*255).astype(numpy.uint8)
            
            label = random.randint(0, 1)
            path = neg_path + '/' + str(n) + '.png'
            
            if label == 1:
                path = pos_path + '/' + str(n) + '.png'   
            
            files.append(path)
            cv2.imwrite(path, image)
            
            images.append(image)
            labels.append(label)

        pp_in = tools.pre_processing.PreProcessingInputFiles(files, labels)
        self.assertEqual(pp_in.count(), 27)
        
        n = 0
        for i in range(3):
            read_images, read_labels = pp_in.read(10)
            self.assertEqual(len(read_images), len(read_labels))
            
            for j in range(len(read_images)):
                for ii in range(H):
                    for jj in range(W):
                        for cc in range(C):
                            self.assertEqual(images[n][ii, jj, cc], read_images[j][ii, jj, cc])
                
                self.assertEqual(labels[n], read_labels[j])
                n += 1
        
        self.assertTrue(pp_in.end())
    
    def test_csv_input(self):
        """
        Test CSV input.
        """
        
        csv_file = 'tests/csv.csv'
        if os.path.exists(csv_file):
            os.unlink(csv_file)
        
        lmdb_path = 'tests/test_lmdb'
        if os.path.exists(lmdb_path):
            shutil.rmtree(lmdb_path)

        images = []
        labels = []
        
        with open(csv_file, 'w') as f:
            for n in range(27):
                image = [random.random(), random.random(), random.random()]
                
                label = 0
                if random.random() > 0.5:
                    label = 1
                
                if label == 0:
                    f.write(str(image[0]) + ',' + str(image[1]) + ',' + str(image[2]) + ',eins\n')
                else:
                    f.write(str(image[0]) + ',' + str(image[1]) + ',' + str(image[2]) + ',zwei\n')
                
                images.append(image)
                labels.append(label)
        
        pp_in = tools.pre_processing.PreProcessingInputCSV(csv_file, ',', 3, {'eins': 0, 'zwei': 1})
        
        self.assertEqual(pp_in.count(), 27)
        
        n = 0
        for t in range(3):
            read_images, read_labels = pp_in.read(10)
            self.assertEqual(len(read_images), len(read_labels))
            
            for i in range(len(read_images)):
                for ii in range(3):
                    self.assertAlmostEqual(images[n][ii], read_images[i][ii, 0, 0])
                
                self.assertEqual(labels[n], read_labels[i])
                n += 1
        
        self.assertTrue(pp_in.end())
        
    def test_pre_processing(self):
        """
        Test Pre-Processing.
        """
        
        N = 27
        H = 10
        W = 10
        C = 3
        
        files = []
        images = []
        labels = []
        
        pos_path = 'tests/test_pos'
        neg_path = 'tests/test_neg'
        lmdb_path = 'tests/test_lmdb'
        
        if os.path.exists(pos_path):
            shutil.rmtree(pos_path)
        
        if os.path.exists(neg_path):
            shutil.rmtree(neg_path)
        
        if os.path.exists(lmdb_path):
            shutil.rmtree(lmdb_path)
        
        os.mkdir(pos_path)
        os.mkdir(neg_path)
        
        for n in range(N):
            image = (numpy.random.rand(H, W, C)*255).astype(numpy.uint8)
            
            label = random.randint(0, 1)
            path = neg_path + '/' + str(n) + '.png'
            
            if label == 1:
                path = pos_path + '/' + str(n) + '.png'   
            
            files.append(path)
            cv2.imwrite(path, image)
            
            images.append(image)
            labels.append(label)

        pp_in = tools.pre_processing.PreProcessingInputFiles(files, labels)
        pp_out = tools.pre_processing.PreProcessingOutputLMDB(lmdb_path)
        pp = tools.pre_processing.PreProcessing(pp_in, pp_out, 10)
        
        pp.run()
        
        lmdb = tools.lmdb_io.LMDB(lmdb_path)
        read_images, read_labels, read_keys = lmdb.read()
        self.assertEqual(len(images), len(labels))
        
        n = 0
        for n in range(N):
            for i in range(H):
                for j in range(W):
                    for c in range(C):
                        self.assertEqual(images[n][i, j, c], read_images[n][i, j, c])
            
            self.assertEqual(labels[n], read_labels[n])
            n += 1
        
    def test_pre_processing_csv(self):
        """
        Test CSV pre processing.
        """
        
        csv_file = 'tests/csv.csv'
        if os.path.exists(csv_file):
            os.unlink(csv_file)
        
        lmdb_path = 'tests/test_lmdb'
        if os.path.exists(lmdb_path):
            shutil.rmtree(lmdb_path)

        images = []
        labels = []
        
        with open(csv_file, 'w') as f:
            for n in range(100):
                image = [random.random(), random.random(), random.random()]
                
                label = 0
                if random.random() > 0.5:
                    label = 1
                
                if label == 0:
                    f.write(str(image[0]) + ',' + str(image[1]) + ',' + str(image[2]) + ',eins\n')
                else:
                    f.write(str(image[0]) + ',' + str(image[1]) + ',' + str(image[2]) + ',zwei\n')
                
                images.append(image)
                labels.append(label)
        
        pp_in = tools.pre_processing.PreProcessingInputCSV(csv_file, ',', 3, {'eins': 0, 'zwei': 1})
        pp_out = tools.pre_processing.PreProcessingOutputLMDB(lmdb_path)
        pp = tools.pre_processing.PreProcessing(pp_in, pp_out, 10)
        
        pp.run()
        
        lmdb = tools.lmdb_io.LMDB(lmdb_path)
        read_images, read_labels, read_keys = lmdb.read()
        
        self.assertEqual(len(images), 100)
        self.assertEqual(len(images), len(labels))
        
        for n in range(100):
            for i in range(3):
                self.assertAlmostEqual(images[n][i], read_images[n][i, 0, 0])
            
            self.assertEqual(labels[n], read_labels[n])
            
if __name__ == '__main__':
    unittest.main()