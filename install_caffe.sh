# This script installs Caffe and pycaffe on Ubuntu 14.04 x64 or 14.10 x64. CPU only, multi-threaded Caffe.
# Usage: 
# 0. Set up here how many cores you want to use during the installation:
# By default Caffe will use all these cores.
NUMBER_OF_CORES=2
# 1. Execute this script, e.g. "bash compile_caffe_ubuntu_14.04.sh" (~30 to 60 minutes on a new Ubuntu).
# 2. Open a new shell (or run "source ~/.bash_profile"). You're done. You can try 
#    running "import caffe" from the Python interpreter to test.

#http://caffe.berkeleyvision.org/install_apt.html : (general install info: http://caffe.berkeleyvision.org/installation.html)
cd
sudo apt-get update
#sudo apt-get upgrade -y # If you are OK getting prompted
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y -q -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" # If you are OK with all defaults

sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev
sudo apt-get install -y --no-install-recommends libboost-all-dev
sudo apt-get install -y libatlas-base-dev 
sudo apt-get install -y python-dev 
sudo apt-get install -y python-pip git

# For Ubuntu 14.04
sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler 

git clone https://github.com/LMDB/lmdb.git 
cd lmdb/libraries/liblmdb
sudo make 
sudo make install

# More pre-requisites 
sudo apt-get install -y cmake unzip doxygen
sudo apt-get install -y protobuf-compiler
sudo apt-get install -y libffi-dev python-dev build-essential
sudo pip install lmdb
sudo pip install numpy
sudo apt-get install -y python-numpy
sudo apt-get install -y gfortran # required by scipy
sudo pip install scipy # required by scikit-image
sudo apt-get install -y python-scipy # in case pip failed
sudo apt-get install -y python-nose
sudo pip install scikit-image # to fix https://github.com/BVLC/caffe/issues/50

# Get caffe (http://caffe.berkeleyvision.org/installation.html#compilation)
cd
mkdir caffe
cd caffe
wget https://github.com/BVLC/caffe/archive/master.zip
unzip -o master.zip
cd caffe-master

# Prepare Python binding (pycaffe)
cd python
for req in $(cat requirements.txt); do sudo pip install $req; done
echo "export PYTHONPATH=$(pwd):$PYTHONPATH " >> ~/.bash_profile # to be able to call "import caffe" from Python after reboot
source ~/.bash_profile # Update shell 
cd ..

# Compile caffe and pycaffe
cp Makefile.config.example Makefile.config
sed -i '8s/.*/CPU_ONLY := 1/' Makefile.config # Line 8: CPU only
sudo apt-get install -y libopenblas-dev
sed -i '33s/.*/BLAS := open/' Makefile.config # Line 33: to use OpenBLAS
# Note that if one day the Makefile.config changes and these line numbers change, we're screwed
# Maybe it would be best to simply append those changes at the end of Makefile.config 
echo "export OPENBLAS_NUM_THREADS=($NUMBER_OF_CORES)" >> ~/.bash_profile 
mkdir build
cd build
cmake ..
cd ..
make all -j$NUMBER_OF_CORES # 4 is the number of parallel threads for compilation: typically equal to number of physical cores
make pycaffe -j$NUMBER_OF_CORES
make test
make runtest
#make matcaffe
make distribute

# Bonus for other work with pycaffe
sudo pip install pydot
sudo apt-get install -y graphviz
sudo pip install scikit-learn

# At the end, you need to run "source ~/.bash_profile" manually or start a new shell to be able to do 'python import caffe', 
# because one cannot source in a bash script. (http://stackoverflow.com/questions/
