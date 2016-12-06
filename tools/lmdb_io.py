"""
Module for comfortably reading and writing LMDBs.
"""

import lmdb
import numpy
import re

import caffe

def version_compare(version_a, version_b):
    """
    Compare two versions given as strings, taken from `here`_.
    
    .. _here: http://stackoverflow.com/questions/1714027/version-number-comparison
    
    :param version_a: version a
    :type version_a: string
    :param version_b: version b
    :type version_b: string
    :return: 0 if versions are equivalent, < 0 if version_a is lower than version_b
        , > 0 if version_b is lower than version_b
    """
    def normalize(v):
        return [int(x) for x in re.sub(r'(\.0+)*$','', v).split(".")]
        
    return cmp(normalize(version_a), normalize(version_b))
    
def to_key(i):
    """
    Transform the given id integer to the key used by :class:`lmdb_io.LMDB`.
    
    :param i: integer id
    :type i: int
    :return: string key
    :rtype: string
    """
    
    return '{:08}'.format(i)

class LMDB:
    """
    Utility class to read and write LMDBs. The code is based on the `LMDB documentation`_,
    as well as `this blog post`_.
    
    .. _LMDB documentation: https://lmdb.readthedocs.io/en/release/
    .. _this blog post: http://deepdish.io/2015/04/28/creating-lmdb-in-python/
    """
    
    def __init__(self, lmdb_path):
        """
        Constructor, given LMDB path.
        
        :param lmdb_path: path to LMDB
        :type lmdb_path: string
        """
        
        self._lmdb_path = lmdb_path
        """ (string) The path to the LMDB to read or write. """
        
        self._write_pointer = 0
        """ (int) Pointer for writing and appending. """
        
    def read(self, key = ''):
        """
        Read a single element or the whole LMDB depending on whether 'key'
        is specified. Essentially a prox for :func:`lmdb.LMDB.read_single`
        and :func:`lmdb.LMDB.read_all`.
        
        :param key: key as 8-digit string of the entry to read
        :type key: string
        :return: data and labels from the LMDB as associate dictionaries, where
            the key as string is the dictionary key and the value the numpy.ndarray
            for the data and the label for the labels
        :rtype: ({string: numpy.ndarray}, {string: float})
        """
        
        if not key:
            return self.read_all();
        else:
            return self.read_single(key);
        
    def read_single(self, key):
        """
        Read a single element according to the given key. Note that data in an
        LMDB is organized using string keys, which are eight-digit numbers
        when using this class to write and read LMDBs.

        :param key: the key to read
        :type key: string
        :return: image, label and corresponding key
        :rtype: (numpy.ndarray, int, string)
        """
        
        image = False
        label = False
        env = lmdb.open(self._lmdb_path, readonly = True)
        
        with env.begin() as transaction:
            raw = transaction.get(key)
            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(raw)
            
            label = datum.label
            if datum.data:
                image = numpy.fromstring(datum.data, dtype = numpy.uint8).reshape(datum.channels, datum.height, datum.width).transpose(1, 2, 0)
            else:
                image = numpy.array(datum.float_data).astype(numpy.float).reshape(datum.channels, datum.height, datum.width).transpose(1, 2, 0)
                
        return image, label, key
        
    def read_all(self):
        """
        Read the whole LMDB. The method will return the data and labels (if
        applicable) as dictionary which is indexed by the eight-digit numbers
        stored as strings.

        :return: images, labels and corresponding keys
        :rtype: ([numpy.ndarray], [int], [string])
        """
        
        images = []
        labels = []
        keys = []
        env = lmdb.open(self._lmdb_path, readonly = True)
        
        with env.begin() as transaction:
            cursor = transaction.cursor();
            
            for key, raw in cursor:
                datum = caffe.proto.caffe_pb2.Datum()
                datum.ParseFromString(raw)
                
                label = datum.label
                
                if datum.data:
                    image = numpy.fromstring(datum.data, dtype = numpy.uint8).reshape(datum.channels, datum.height, datum.width).transpose(1, 2, 0)
                else:
                    image = numpy.array(datum.float_data).astype(numpy.float).reshape(datum.channels, datum.height, datum.width).transpose(1, 2, 0)
                
                images.append(image)
                labels.append(label)
                keys.append(key)
        
        return images, labels, keys
    
    def count(self):
        """
        Get the number of elements in the LMDB.
        
        :return: count of elements
        :rtype: int
        """
        
        env = lmdb.open(self._lmdb_path)
        with env.begin() as transaction:
            return transaction.stat()['entries']
        
    def keys(self, n = 0):
        """
        Get the first n (or all) keys of the LMDB
        
        :param n: number of keys to get, 0 to get all keys
        :type n: int
        :return: list of keys
        :rtype: [string]
        """
        
        keys = []
        env = lmdb.open(self._lmdb_path, readonly = True)
        
        with env.begin() as transaction:
            cursor = transaction.cursor()
            
            i = 0
            for key, value in cursor:
                
                if i >= n and n > 0:
                    break;
                
                keys.append(key)
                i += 1
        
        return keys
    
    def write(self, images, labels = []):
        """
        Write a single image or multiple images and the corresponding label(s).
        The imags are expected to be two-dimensional NumPy arrays with
        multiple channels (if applicable).
        
        :param images: input images as list of numpy.ndarray with height x width x channels
        :type images: [numpy.ndarray]
        :param labels: corresponding labels (if applicable) as list
        :type labels: [float]
        :return: list of keys corresponding to the written images
        :rtype: [string]
        """
        
        if len(labels) > 0:
            assert len(images) == len(labels)
        
        keys = []
        env = lmdb.open(self._lmdb_path, map_size = max(1099511627776, len(images)*images[0].nbytes))
        
        with env.begin(write = True) as transaction:
            for i in range(len(images)):
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = images[i].shape[2]
                datum.height = images[i].shape[0]
                datum.width = images[i].shape[1]
                
                assert version_compare(numpy.version.version, '1.9') < 0, "installed numpy is 1.9 or higher, change .tostring() to .tobytes()"
                assert images[i].dtype == numpy.uint8 or images[i].dtype == numpy.float, "currently only numpy.uint8 and numpy.float images are supported"
                
                if images[i].dtype == numpy.uint8:
                    datum.data = images[i].transpose(2, 0, 1).tostring()
                else:
                    datum.float_data.extend(images[i].transpose(2, 0, 1).flat)
                    
                if len(labels) > 0:
                    datum.label = labels[i]
                
                key = to_key(self._write_pointer)
                keys.append(key)
                
                transaction.put(key.encode('ascii'), datum.SerializeToString());
                self._write_pointer += 1
        
        return keys
