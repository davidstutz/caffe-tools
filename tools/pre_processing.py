"""
Pre-processing for Caffe.
"""

import os
import cv2
import csv
import numpy
import random
import tools.lmdb_io

class PreProcessing(object):
    """
    Pre-processing utilities to normalize data and compute mean as well as
    standard deviation.
    """
    
    def __init__(self, source, dest, batch_size = 100000):
        """
        Constructor, needs a :class:`tools.pre_processing.PreProcessingInput` as
        input source and a :class:`tools.pre_processing.PreProcessingOutput` as
        output destination.
        
        :param source: input source
        :type source: (tools.pre_processing.PreProcessingInput)
        :param dest: output destination
        :type dest: (tools.pre_processing.PreProcessingOutput)
        :param batch_size: batch size in which to process the images
        :type batch_size: int
        """
        
        self._source = source
        """ (tools.pre_processing.PreProcessingInput) Input source. """
        
        self._dest = dest
        """ (tools.pre_processing.PreProcessingOutput) Output source. """
        
        self._batch_size = batch_size
        """ (int) Batch size to process images in. """
    
    def run(self):
        """
        Run pre-processing, in this case simply writes from source to destination.
        """ 
        
        while not self._source.end():
            images, labels = self._source.read(self._batch_size)
            self._dest.write(images, labels)

class PreProcessingNormalize(PreProcessing):
    """
    Normalize the data to lie in [0, 1] by dividing by a fixed given value.
    """
    
    def __init__(self, source, dest, normalizer = 255., batch_size = 100000):
        """
        Constructor, needs a :class:`tools.pre_processing.PreProcessingInput` as
        input source and a :class:`tools.pre_processing.PreProcessingOutput` as
        output destination.
        
        :param source: input source
        :type source: (tools.pre_processing.PreProcessingInput)
        :param dest: output destination
        :type dest: (tools.pre_processing.PreProcessingOutput)
        :param normalizer: value to normalize by
        :type normalizer: float
        :param batch_size: batch size in which to process the images
        :type batch_size: int
        """
        
        super(PreProcessingNormalize, self).__init__(source, dest, batch_size)
        
        self._normalizer = normalizer
        """ (float) The value to normalize by. """
        
    def run(self):
        """
        Run pre-procesisng, this will normalize the images by the overall mean.
        
        :return: mean of normalized data
        :rtype: numpy.ndarray
        """
        
        while not self._source.end():
            images, labels = self._source.read(self._batch_size)

            normalized = []
            for i in range(len(images)):
                normalized_image = images[i]/float(self._normalizer)
                normalized.append(normalized_image)
            
            self._dest.write(normalized, labels)
        
class PreProcessingSplit:
    """
    Pre processing utilities to split the data into training and test set
    or training, validaiton and test sets.
    """
    
    def __init__(self, source, dests, split = (0.9, 0.1), batch_size = 100000):
        """
        Constructor, needs a :class:`tools.pre_processing.PreProcessingInput` as
        input source and a :class:`tools.pre_processing.PreProcessingOutput` as
        output destination.
        
        :param source: input source
        :type source: (tools.pre_processing.PreProcessingInput)
        :param dests: output destinations for training/validation/test sets
        :type dests: ((tools.pre_processing.PreProcessingOutput, tools.pre_processing.PreProcessingOutput, tools.pre_processing.PreProcessingOutput))
        :param batch_size: batch size in which to process the images
        :type batch_size: int
        :param split: the train/validation/test split, a tuple of probabilities
            summing to one, either 3 probabilities or two (without validation set)
        :type split: (int, int, int) or (int, int)
        """
        
        assert len(split) == 2 or len(split) == 3, "split should contain 2 or 3 probabilities"
        
        # http://stackoverflow.com/questions/5595425/what-is-the-best-way-to-compare-floats-for-almost-equality-in-python
        def isclose(a, b, relative_tolerance = 1e-09, absolute_tolerance = 1e-06):
            return abs(a - b) <= max(relative_tolerance * max(abs(a), abs(b)), absolute_tolerance)
        
        if len(split) == 2:
            assert isclose(split[0] + split[1], 1.0), "the probabilities do not sum to 1"
        if len(split) == 3:
            assert isclose(split[0] + split[1] + split[2], 1.0), "the porbabilities do not sum to 1"
        
        assert len(split) == len(dests), "number of destinations does not fit number of split probabilities"
        
        self._source = source
        """ (tools.pre_processing.PreProcessingInput) Input source. """
        
        self._dests = dests
        """ (tools.pre_processing.PreProcessingOutput,tools.pre_processing.PreProcessingOutput,tools.pre_processing.PreProcessingOutput) Output source. """
        
        self._split = split
        """ ((float, float) or (float, float, float)) split in training/validation/test sets. """
        
        self._batch_size = batch_size
        """ (int) Batch size to process images in. """
        
    def run(self):
        """
        Run pre-processing, decide to output in training/validation/test sets.
        """
        
        while not self._source.end():
            images, labels = self._source.read(self._batch_size)
            
            training_images = []
            validation_images = []
            test_images = []
            
            training_labels = []
            validation_labels = []
            test_labels = []
            
            for n in range(len(images)):
                r = random.random()
                if r < self._split[0]:
                    training_images.append(images[n])
                    if len(labels) > 0:
                        training_labels.append(labels[n])
                        
                else:
                    if len(self._split) == 2:
                        test_images.append(images[n])
                        if len(labels) > 0:
                            test_labels.append(labels[n])
                            
                    else:
                        if r >= self._split[0] + self._split[1]:
                            test_images.append(images[n])
                            if len(labels) > 0:
                                test_labels.append(labels[n])
                                
                        else:
                            validation_images.append(images[n])
                            if len(labels) > 0:
                                validation_labels.append(labels[n])
            
            self._dests[0].write(training_images, training_labels)
            
            if len(self._split) == 2:
                self._dests[1].write(test_images, test_labels)
            else:
                self._dests[1].write(validation_images, validation_labels)
                self._dests[2].write(test_images, test_labels)  

class PreProcessingSubsample(PreProcessing):
    """
    Pre processing utilities to shuffle the data. This is a simple approach of
    shuffling that only works if the batch size is larger than the dataset size!
    """
    
    def __init__(self, source, dest, p = 0.5, batch_size = 100000):
        """
        Constructor, needs a :class:`tools.pre_processing.PreProcessingInput` as
        input source and a :class:`tools.pre_processing.PreProcessingOutput` as
        output destination.
        
        :param source: input source
        :type source: (tools.pre_processing.PreProcessingInput)
        :param dest: output destination
        :type dest: (tools.pre_processing.PreProcessingOutput)
        :param p: the probability of taking a sample
        :type p: float
        :param batch_size: batch size in which to process the images
        :type batch_size: int
        """
        
        super(PreProcessingSubsample, self).__init__(source, dest, batch_size)
        
        self._p = p
        """ (float) Probability of taking a sample. """
        
    def run(self):
        """
        Run pre-processing, i.e. shuffle the data.
        """
        
        while not self._source.end():
                
            read_images, read_labels = self._source.read(self._batch_size)
            indices = numpy.random.choice(len(read_images), len(read_images))
            
            write_images = []
            write_labels = []                
            
            for index in indices:
                r = random.random()
                if r < self._p:
                    write_images.append(read_images[index])
                    if len(read_labels) > 0:
                        write_labels.append(read_labels[index])
            
            self._dest.write(write_images, write_labels)

class PreProcessingShuffle(PreProcessing):
    """
    Pre processing utilities to shuffle the data. This is a simple approach of
    shuffling that only works if the batch size is larger than the dataset size!
    """
    
    def __init__(self, source, dest, iterations = 10, batch_size = 100000):
        """
        Constructor, needs a :class:`tools.pre_processing.PreProcessingInput` as
        input source and a :class:`tools.pre_processing.PreProcessingOutput` as
        output destination.
        
        :param source: input source
        :type source: (tools.pre_processing.PreProcessingInput)
        :param dest: output destination
        :type dest: (tools.pre_processing.PreProcessingOutput)
        :param batch_size: batch size in which to process the images
        :type batch_size: int
        """
        
        self._source = source
        """ (tools.pre_processing.PreProcessingInput) Input source. """
        
        self._dest = dest
        """ (tools.pre_processing.PreProcessingOutput) Output source. """
        
        self._batch_size = batch_size
        """ (int) Batch size to process images in. """
    
    def run(self):
        """
        Run pre-processing, i.e. shuffle the data.
        """
        
        while not self._source.end():
                
            read_images, read_labels = self._source.read(self._batch_size)
            indices = numpy.random.choice(len(read_images), len(read_images))
            
            write_images = []
            write_labels = []                
            
            for index in indices:
                write_images.append(read_images[index])
                if len(read_labels) > 0:
                    write_labels.append(read_labels[index])
            
            self._dest.write(write_images, write_labels)
        
class PreProcessingInput:
    """
    Provides the input data for :class:`tools.pre_processing.PreProcessing`.    
    """
    
    def reset(self):
        """
        Reset reading to start from the beginning.
        """
        
        raise NotImplementedError("Should have been implemented!")
    
    def read(self, n):
        """
        Read data in batches.
        
        :param n: number of images to read
        :type n: int
        :return: images and optionally labels as lists
        :rtype: ([numpy.ndarray], [float])
        """
        
        raise NotImplementedError("Should have been implemented!")
    
    def count(self):
        """
        Return the count of imags.
        
        :return: count
        :rtype: int        
        """
        
        raise NotImplementedError("Should have been implemented!")
    
    def end(self):
        """
        Whether the end has been reached.
        
        :return: true if the end has been reached or overstepped
        :rtype: bool
        """
        
        raise NotImplementedError("Should have been implemented!")
    
class PreProcessingInputLMDB(PreProcessingInput):
    """
    Provides the input data for :class:`tools.pre_processing.PreProcessing`
    from an LMDB.
    """
    
    def __init__(self, lmdb_path):
        """
        Constructor, provide path to LMDB.
        
        :param lmdb_path: path to LMDB
        :type lmdb_path: string
        """
        
        self._lmdb = tools.lmdb_io.LMDB(lmdb_path)
        """ (tools.lmdb_io.LMDB) Underlying LMDB. """
        
        self._keys = self._lmdb.keys()
        """ ([string]) Keys of elements stored in the LMDB. """
        
        self._pointer = 0
        """ (int) Current index to start reading. """
    
    def reset(self):
        """
        Reset reading to start from the beginning.
        """
        
        self._pointer = 0
        
    def read(self, n):
        """
        Read data in batches.
        
        :param n: number of images to read
        :type n: int
        :return: images and optionally labels as lists
        :rtype: ([numpy.ndarray], [float])
        """
        
        images = []
        labels = []
        keys = self._keys[self._pointer: min(self._pointer + n, len(self._keys))]
        
        for key in keys:
            image, label, key = self._lmdb.read_single(key)
            
            images.append(image)
            labels.append(label)
        
        self._pointer += n
        
        return images, labels
    
    def count(self):
        """
        Return the count of imags.
        
        :return: count
        :rtype: int        
        """
        
        return len(self._keys)
        
    def end(self):
        """
        Whether the end has been reached.
        
        :return: true if the end has been reached or overstepped
        :rtype: bool
        """
        
        return self._pointer >= len(self._keys)

class PreProcessingInputFiles(PreProcessingInput):
    """
    Provide input data for :class:`tools.pre_processing.PreProcessing` based on a list
    of file paths.    
    """
    
    def __init__(self, files, labels = []):
        """
        Constructor, provide list of files and optional list of labels.
        
        :param files: file paths
        :type files: [string]
        :param labels: labels
        :type labels: [float]
        """
        
        assert len(files) > 0, "files is empty"
        
        self._files = files
        """ ([string]) File paths. """
        
        if len(labels) > 0:
            assert len(labels) == len(files), "if labels are provided there needs to be a label for each file"
            
        self._labels = labels
        """ ([float]) Labels. """
        
        self._pointer = 0
        """ (int) Current index to start reading. """
    
    def reset(self):
        """
        Reset reading to start from the beginning.
        """
        
        self._pointer = 0
        
    def read(self, n):
        """
        Read data in batches.
        
        :param n: number of images to read
        :type n: int
        :return: images and optionally labels as lists
        :rtype: ([numpy.ndarray], [float])
        """
        
        images = []
        labels = []
        files = self._files[self._pointer: min(self._pointer + n, len(self._files))]
        
        labels = []
        if len(self._labels) > 0:
            labels = self._labels[self._pointer: min(self._pointer + n, len(self._labels))]
        
        for i in range(len(files)):
            assert os.path.exists(files[i]), "file %s not found" % files[i]
            
            image = cv2.imread(files[i])
            images.append(image)
        
        self._pointer += n
        
        return images, labels
    
    def count(self):
        """
        Return the count of imags.
        
        :return: count
        :rtype: int        
        """
        
        return len(self._files)    
    
    def end(self):
        """
        Whether the end has been reached.
        
        :return: true if the end has been reached or overstepped
        :rtype: bool
        """
        
        return self._pointer >= len(self._files)

class PreProcessingInputCSV:
    """
    Allows :class:`tools.pre_processing.PreProcessing` to take input from a 
    CSV file.
    """
    
    def __init__(self, csv_file, delimiter = ',', label_column = -1, label_column_mapping = {}):
        """
        Constructor.
        
        :param csv_file: path to the csv file to use
        :type csv_file: string
        :param csv_delimiter: delimited used between cells
        :type csv_delimiter: string
        :param label_column: the label column index, or -1 if label is not
            provided in the CSV file
        :type label_column: int
        :param label_column_mapping: the mapping from categoric labels to label
            indices if the labels provided in the CSV file are category names,
            or an empty object if the labels are already saved as integers
        :type label_column_mapping: {string: int}
        """
        
        assert os.path.exists(csv_file), "the CSV file could not be found"
        self._csv_file = csv_file
        """ (string) CSV file path. """
        
        self._delimiter = delimiter
        """ (string) Delimiter between cells for CSV file. """
        
        self._label_column = label_column
        """ (int) The column to use for labels, -1 if label is not present. """
        
        self._label_column_mapping = label_column_mapping
        """ ({string: intr}) The mapping from categoric label names to label indices if required. """
        
        self._columns = -1
        """ (int) Number of columns. """
        
        self._rows = 0
        """ (int) Number of rows. """
        
        self._pointer = 0
        """ (int) Pointer to current row. """
        
        with open(self._csv_file) as f:
            self._rows = 0
            for cells in csv.reader(f, delimiter = self._delimiter):
                cells = [cell.strip() for cell in cells if len(cell.strip()) > 0]
                  
                if self._columns < 0:
                    self._columns = len(cells)
                
                if len(cells) > 0:
                    assert self._columns == len(cells), "CSV file does not contain a consistent number of columns"
                    self._rows += 1
                    
    def reset(self):
        """
        Reset reading to start from the beginning.
        """
        
        self._pointer = 0
        
    def read(self, n):
        """
        Read data in batches.
        
        :param n: number of images to read
        :type n: int
        :return: images and optionally labels as lists
        :rtype: ([numpy.ndarray], [float])
        """
        
        images = []
        labels = []
        
        with open(self._csv_file) as f:
            row = 0
            for cells in csv.reader(f, delimiter = self._delimiter):
                if row == self._pointer and n > 0:
                    cells = [cell.strip() for cell in cells if len(cell.strip()) > 0]
                    
                    if len(cells) > 0:
                        assert self._columns == len(cells), "CSV file does not contain a consistent number of columns"
                        
                        if self._label_column < 0:
                            cells = [float(cell) for cell in cells]
                        else:
                            label = cells[self._label_column]
                            cells = cells[0:self._label_column] + cells[self._label_column + 1:]
                            
                            if len(self._label_column_mapping) > 0:
                                assert label in self._label_column_mapping, "label %s not found in label_column_mapping" % label
                                label = int(self._label_column_mapping[label])
                        
                            labels.append(label)
                        images.append(numpy.array(cells).reshape(len(cells), 1, 1).astype(float))
                    
                    self._pointer += 1
                    n -= 1
                
                row += 1
        
        return images, labels
        
    def count(self):
        """
        Return the count of imags.
        
        :return: count
        :rtype: int        
        """
        
        return self._rows   
    
    def end(self):
        """
        Whether the end has been reached.
        
        :return: true if the end has been reached or overstepped
        :rtype: bool
        """
        
        return self._pointer >= self._rows

class PreProcessingOutput:
    """
    Allows :class:`tools.pre_processing.PreProcessing` to write its output.
    """
    
    def write(self, images, labels = []):
        """
        Write the images and the given labels as output.
        
        :param images: list of images as numpy.ndarray
        :type images: [numpy.ndarray]
        """
    
        raise NotImplementedError("Should have been implemented!")

class PreProcessingOutputLMDB(PreProcessingOutput):
    """
    Allows :class:`tools.pre_processing.PreProcessing` to write its output
    to an LMDB.
    """
    
    def __init__(self, lmdb_path):
        """
        Constructor, provide path to LMDB.
        
        :param lmdb_path: path to LMDB
        :type lmdb_path: string
        """
        
        self._lmdb = tools.lmdb_io.LMDB(lmdb_path)
        """ (tools.lmdb_io.LMDB) Underlying LMDB. """
        
    def write(self, images, labels = []):
        """
        Write the images and the given labels as output.
        
        :param images: list of images as numpy.ndarray
        :type images: [numpy.ndarray]
        """
        
        self._lmdb.write(images, labels)