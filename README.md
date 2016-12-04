# Caffe-Tools

Tools and examples for pyCaffe, including:

* LMDB input and output and conversion from/to CSV and image files;
* monitoring the training process including error, loss and gradients;
* on-the-fly data augmentation;
* custom Python layers.

The data used for the examples can either be generated manually, see the documentation
or corresponding files in `examples`, or downloaded from [davidstutz/caffe-tools-data](https://github.com/davidstutz/caffe-tools-data).

Also see the corresponding blog articles at [davidstutz.de](http://davidstutz.de).

## Examples

The provided examples include:

* [MNIST](http://yann.lecun.com/exdb/mnist/): [examples/mnist.py](examples/mnist.py)
* [Iris](https://archive.ics.uci.edu/ml/datasets/Iris): [examples/iris.py](examples/iris.py)
* [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html): [examples/cifar10.py](examples/cifar10.py.py)
* [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html): [examples/bsds500.py](examples/bsds500.py)

Note that the BSDS500 example is **work in progress**! The corresponding data can 
be downloaded from [davidstutz/caffe-tools-data](https://github.com/davidstutz/caffe-tools-data).
See the instructions in the corresponding files for details.

## Resources

Some resources I found usefl while working with Caffe:

* Installation:
    * http://stackoverflow.com/questions/31395729/how-to-enable-multithreading-with-caffe/31396229
    * https://github.com/BVLC/caffe/wiki/Install-Caffe-on-EC2-from-scratch-(Ubuntu,-CUDA-7,-cuDNN-3)
    * https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide
    * https://gist.github.com/titipata/f0ef48ad2f0ebc07bcb9
    * https://github.com/asampat3090/caffe-ubuntu-14.04
    * https://github.com/mrgloom/Caffe-snippets
* GitHub Repositories for pyCaffe:
    * https://github.com/nitnelave/pycaffe_tutorial
    * https://github.com/pulkitag/pycaffe-utils
    * https://github.com/DeeperCS/pycaffe-mnist
    * https://github.com/swift-n-brutal/pycaffe_utils
    * https://github.com/jimgoo/caffe-oxford102
    * https://github.com/ruimashita/caffe-train
    * https://github.com/roseperrone/video-object-detection
    * https://github.com/pecarlat/caffeTools
    * https://github.com/donnemartin/data-science-ipython-notebooks
    * https://github.com/jay-mahadeokar/pynetbuilder
    * https://github.com/adilmoujahid/deeplearning-cats-dogs-tutorial
    * https://github.com/Franck-Dernoncourt/caffe_demos
    https://github.com/koosyong/caffestudy
* Issues:
    * https://github.com/BVLC/caffe/issues/3651 (solverstate)
    * https://github.com/BVLC/caffe/issues/1566
    * https://github.com/BVLC/caffe/pull/3082/files (snapshot)
    * https://github.com/BVLC/caffe/issues/1257 (net surgery on solver net)
    * https://github.com/BVLC/caffe/issues/409 (net diverges, loss = NaN)
    * https://github.com/BVLC/caffe/issues/1168 (pyCaffe example incldued)
    * https://github.com/BVLC/caffe/issues/462 (pyCaffe example incldued)
    * https://github.com/BVLC/caffe/issues/2684 (change batch size)
    * https://github.com/rbgirshick/py-faster-rcnn/issues/77 (load solverstate)
    * https://github.com/BVLC/caffe/issues/2116 (Caffe LMDB float data)
* LMDB:
    * https://lmdb.readthedocs.io/en/release/
    * http://research.beenfrog.com/code/2015/12/30/write-read-lmdb-example.html
    * http://deepdish.io/2015/04/28/creating-lmdb-in-python/
    * https://github.com/BVLC/caffe/issues/3959
* Tutorials/Blogs:
    * http://christopher5106.github.io/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html
    * http://www.alanzucconi.com/2016/05/25/generating-deep-dreams/#part2
    * http://adilmoujahid.com/posts/2016/06/introduction-deep-learning-python-caffe/
* Caffe Versions:
    * https://github.com/kevinlin311tw/caffe-augmentation (on-the-fly data augmentation)
    * https://github.com/ShaharKatz/Caffe-Data-Augmentation (data augmentation)
* 3D:
    * https://github.com/faustomilletari/3D-Caffe
    * https://github.com/wps712/caffe4video
* StackOverflow:
    * http://stackoverflow.com/questions/33905326/caffe-training-without-testing (training without testing)
    * http://stackoverflow.com/questions/38348801/caffe-hangs-after-printing-data-label (stuck at data -> label)
    * http://stackoverflow.com/questions/35529078/how-to-predict-in-pycaffe (predicting in pyCaffe)
    * http://stackoverflow.com/questions/35529078/how-to-predict-in-pycaffe/35572495#35572495 (testing from LMDB with transformer)
    * http://stackoverflow.com/questions/37642885/am-i-using-lmdb-incorrectly-it-says-environment-mapsize-limit-reached-after-0-i (LMDB mapsize)
    * http://stackoverflow.com/questions/31820976/lmdb-increase-map-size (LMDB mapsize)
    * http://stackoverflow.com/questions/34092606/how-to-get-the-dataset-size-of-a-caffe-net-in-python/34117558 (dataset size)
    * http://stackoverflow.com/questions/32379878/cheat-sheet-for-caffe-pycaffe (pyCaffe cheat sheet)
    * http://stackoverflow.com/questions/38511503/how-to-compute-test-validation-loss-in-pycaffe (copying weights to test net)
    * http://stackoverflow.com/questions/29788075/setting-glog-minloglevel-1-to-prevent-output-in-shell-from-caffe (slience GLOG logging in 
    * http://stackoverflow.com/questions/36108120/shuffle-data-in-lmdb-file
    * http://stackoverflow.com/questions/36459266/caffe-python-manual-sgd
* Layers:
    * http://installing-caffe-the-right-way.wikidot.com/start
    * https://github.com/NVIDIA/DIGITS/tree/master/examples/python-layer
    * https://github.com/BVLC/caffe/blob/master/examples/pycaffe/layers/pyloss.py
    * https://github.com/BVLC/caffe/blob/master/examples/pycaffe/layers/pascal_multilabel_datalayers.py
    * http://stackoverflow.com/questions/34549743/caffe-how-to-get-the-phase-of-a-python-layer/34588801#34588801
    * http://stackoverflow.com/questions/34996075/caffe-data-layer-example-step-by-step
    * https://github.com/BVLC/caffe/issues/4023
    * https://codegists.com/code/caffe-python-layer/
    * https://codedump.io/share/CiQmhfC63OD0/1/pycaffe-how-to-create-custom-weights-in-a-python-layer
    * http://stackoverflow.com/questions/34498527/pycaffe-how-to-create-custom-weights-in-a-python-layer
    * https://github.com/gcucurull/caffe-conf-matrix/blob/master/python_confmat.py | http://gcucurull.github.io/caffe/python/deep-learning/2016/06/29/caffe-confusion-matrix/

## Documentation

Installing and running Sphinx (also see [davidstutz/sphinx-example](https://github.com/davidstutz/sphinx-example) for details):

    $ sudo apt-get install python-sphinx
    $ sudo pip install sphinx
    $ cd docs
    $ make html

## License

Copyright (c) 2016 David Stutz All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* Neither the name of David Stutz nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
