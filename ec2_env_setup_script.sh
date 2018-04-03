# -y added to default yes




sudo apt-get update
sudo apt-get upgrade
Y
# Install the required developer tools and packages:
sudo apt-get install -y build-essential cmake pkg-config

# These packages allow you to load various image file formats such as JPEG, PNG, TIFF, etc.
sudo apt-get install -y libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev

# Install the GTK development library. This library is used to build Graphical User Interfaces (GUIs)
sudo apt-get install -y libgtk2.0-dev

# not neccesary if not computer vision is going to happen
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libatlas-base-dev gfortran

# I normally like the boost library to be installed.
# This will already install python 2.7
# If one does not want boost simply replace
# next two lines with
# sudo apt-get install python
sudo apt-get install -y libboost-all-dev

# get pip. This can also be done with:
# sudo apt-get install python-pip
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
sudo apt-get install -y python-tk



# install python things
sudo pip install --upgrade pip
sudo pip install numpy
sudo pip install scipy
sudo pip install scikit-learn
sudo pip install scikit-optimize
sudo pip install pandas
sudo pip install pandasql
sudo pip install matplotlib
sudo pip install awscli
sudo pip install jupyter
sudo pip install seaborn
sudo pip install python-dotenv
sudo pip install joblib
sudo pip install gensim
sudo pip install flask
sudo pip install autopep8
sudo pip install flake8
sudo pip install nose
sudo pip install Cython
sudo pip install xgboost
sudo pip install boto
sudo pip install boto3
sudo pip install feather-format
sudo pip install scikit-optimize
sudo pip install hyperopt
sudo pip install h5py
sudo pip install hyperopt
sudo pip install ipython
sudo pip install joblib
sudo pip install jupyter
sudo pip install Keras
sudo pip install lightfm
pip install git+https://github.com/coreylynch/pyFM
sudo pip install pymc3
sudo pip install pymc
sudo pip install requests
sudo pip install sklearn-pandas
sudo pip install theano
sudo pip install tensorflow
sudo apt install spark

sudo apt-get install -y tcl-dev tk-dev python-tk python3-tk
pip uninstall matplotlib
git clone https://github.com/matplotlib/matplotlib.git
cd matplotlib
#python setup.py install


# Install Java
# https://www.digitalocean.com/community/tutorials/how-to-install-java-on-ubuntu-12-04-with-apt-get
sudo apt-get update
sudo apt-get install default-jre
sudo apt-get install default-jdk
java -version



# aws ec2 run-instances --image-id ami-xxxxxxxx --count 1 --instance-type t1.micro --key-name MyKeyPair --security-groups my-sg
# aws ec2 run-instances --image-id ami-5ede8338 --count 1 --instance-type t2.micro --key-name ~/keys/dbs-test.pem

# chmod 400 /Users/davidbartram-shaw/keys/dm-dev.pem
# chmod 400 /Users/davidbartram-shaw/keys/dbs-test.pem
