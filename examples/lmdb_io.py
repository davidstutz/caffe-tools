"""
Examples for reading LMDBs.

.. argparse::
   :ref: examples.lmdb_io.get_parser
   :prog: lmdb_io
"""

import os
import cv2
import argparse

# To silence Caffe! Must be added before importing Caffe or modules which
# are importing Caffe.
os.environ['GLOG_minloglevel'] = '0'
import tools.lmdb_io

def get_parser():
    """
    Get the parser.
    
    :return: parser
    :rtype: argparse.ArgumentParser
    """
    
    parser = argparse.ArgumentParser(description = 'Read LMDBs.')
    parser.add_argument('mode', default = 'read')
    parser.add_argument('--lmdb', default = 'examples/cifar10/train_lmdb', type = str,
                       help = 'path to input LMDB')
    parser.add_argument('--output', default = 'examples/output', type = str,
                        help = 'output directory')
    parser.add_argument('--limit', default = 100, type = int,
                        help = 'limit the number of images to read')
                       
    return parser

def main_statistics():
    """
    Read and print the size of an LMDB.
    """
    
    lmdb = tools.lmdb_io.LMDB(args.lmdb)
    print(lmdb.count())

def main_read():
    """
    Read up to ``--limit`` images from the LMDB.
    """
    
    lmdb = tools.lmdb_io.LMDB(args.lmdb)
    keys = lmdb.keys()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    with open(args.output + '/labels.txt', 'w') as f:
        for n in range(min(len(keys), args.limit)):
            image, label, key = lmdb.read_single(keys[n])
            image_path = args.output + '/' + keys[n] + '.png'
            cv2.imwrite(image_path, image)
            f.write(image_path + ': ' + str(label) + '\n')

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    if args.mode == 'read':
        main_read()
    elif args.mode == 'statistics':
        main_statistics()
    else:
        print('Invalid mode.')