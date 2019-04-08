
""" -----------------------------------------------------------------------------------------------------------
#
# Ubuntu Machines
#
#----------------------------------------------------------------------------------------------------------- """

"""
	A collection of notes for using ec2 with AWS
	Used for configuration of Ubuntu instances, SSH problems and GPU usage
"""


######################
# sublime on ec2
######################
sudo wget -O /usr/local/bin/rsub https://raw.github.com/aurora/rmate/master/rmate
sudo chmod +x /usr/local/bin/rsub

# CHANGE ON MAC
nano ~/.ssh/config
# PASTE IN
Hostname ec2-35-165-208-131.us-west-2.compute.amazonaws.com
RemoteForward 52698 127.0.0.1:52698
# CTRL+o saves
# CTRL+x exits


#SSH in
sudo wget -O /usr/local/bin/rsub https://raw.github.com/aurora/rmate/master/rmate
sudo chmod +x /usr/local/bin/rsub

#open file in subl
rsub filename.py


cd Notebook/code/spiders/image-spider/image-spider/spiders/

# UPDATE apt-get
sudo apt-get update



######################
# Permission denied new keys
######################

# this was forcing ssh to my prod instance
mv $HOME/.ssh/config $HOME/.ssh/config.bk

# Move back to sort out Sublime
mv $HOME/.ssh/config.bk $HOME/.ssh/config



######################
# GPU
######################

# Kill python process
sudo pkill python

#show python process
sudo pgrep python


#see GPU memory
nvidia-smi



######################
# Package problems
######################

# Graphviz
# Check if installed
dot -V

# Install it
sudo apt install graphviz


######################
# python setup 2&3
######################
# Fixes the multiple running of pythons
python2 -m pip install ipykernel
python2 -m ipykernel install --user

sudo python3 -m pip install ipykernel
sudo python3 -m ipykernel install --user



###########################
# jupyter Notebook setup
###########################
# run in ec2
ipython
from IPython.lib import passwd

passwd() #password
#Out[2]: 'sha1:42df52567bfa:1e71bc87c38b15efec3799f2b0b5e3e1c121feae'
# 'sha1:438ebcc4c512:8edea6d2becf9a0c7e7125965833ea4fd217d4c7'
# 'sha1:826246882b57:0957445ad9fca61e5c26ea8478f3af88f928da19'
# 'sha1:d7acf2d1c422:4d1f0dbef6b883f79231bf5a4f6cd84a15e77e25'

exit()

jupyter notebook --generate-config
y
mkdir certs
cd certs
sudo openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout mycert.pem -out mycert.pem

'''
Generating a 1024 bit RSA private key
..............................++++++
............++++++
writing new private key to 'mycert.pem'
-----
You are about to be asked to enter information that will be incorporated
into your certificate request.
What you are about to enter is what is called a Distinguished Name or a DN.
There are quite a few fields but you can leave some blank
For some fields there will be a default value,
If you enter '.', the field will be left blank.
-----
Country Name (2 letter code) [AU]:
State or Province Name (full name) [Some-State]:
Locality Name (eg, city) []:
Organization Name (eg, company) [Internet Widgits Pty Ltd]:
Organizational Unit Name (eg, section) []:
Common Name (e.g. server FQDN or YOUR name) []:
Email Address []:
'''

cd ~/.jupyter/
nano jupyter_notebook_config.py

#paste this in, [control+X] to exit
'''
c = get_config()
c.IPKernelApp.pylab = 'inline'
# add this for GCP
c.NotebookApp.allow_remote_access = True

c.NotebookApp.certfile = u'/home/ubuntu/certs/mycert.pem'
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False

# Your password below will be whatever you copied earlier
c.NotebookApp.password = 'sha1:d7acf2d1c422:4d1f0dbef6b883f79231bf5a4f6cd84a15e77e25'
c.NotebookApp.port = 8889
'''



cd

# Just accessable to a cerain part
mkdir Notebook
cd Notebook/

#nohup means it doesnt close down when exit
nohup jupyter notebook

nohup jupyter-notebook --open_browser=False

#https://193.109.116.16/32:8888/
https://35.199.147.134:8889/
https://35.199.147.217:8889/

#####################
# Install OPENCV
# UBUNTU
#####################
# from the guru https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/

''' Step #1: Install OpenCV dependencies on Ubuntu 16.04 '''
sudo apt-get update
sudo apt-get upgrade
# install developer tools:
sudo apt-get install build-essential cmake pkg-config
# facilitate the loading and decoding process for file formats from disk such as JPEG, PNG, TIFF
sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
# install packages used to process video streams and access frames from cameras
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
# highgui module (GUI for cv2.imshow ect) relies on the GTK library
sudo apt-get install libgtk-3-dev
# libraries that are used to optimize various functionalities inside OpenCV, such as matrix operations:
sudo apt-get install libatlas-base-dev gfortran
#  installing the Python development headers and libraries for both Python 2.7 and Python 3.5
sudo apt-get install python2.7-dev python3.5-dev

''' Step #2: Download the OpenCV source '''
cd ~
# most recent version of OpenCV is 3.1.0 , which we download a .zip  of and unarchive
# found here https://github.com/opencv/opencv
wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.1.0.zip
sudo apt install unzip
unzip opencv.zip
# we also need the opencv_contrib repository as well:
# contains SIFT and SURF on OpenCV 3+
wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.1.0.zip
unzip opencv_contrib.zip


'''Step #3: Setup your Python environment — Python 2.7 or Python 3 '''
# If setting up virtual environments
pip install numpy

'''Step #4: Configuring and compiling OpenCV on Ubuntu 16.04'''
cd ~/opencv-3.1.0/
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.1.0/modules \
    -D BUILD_EXAMPLES=ON ..



#####################
# Install OPENCV
# UBUNTU - Working
#####################
# https://www.learnopencv.com/install-opencv3-on-ubuntu/

'''Step 1: Update packages '''
sudo apt-get update
sudo apt-get upgrade

''' Step 2: Install OS libraries '''
sudo apt-get remove x264 libx264-dev #remove old versions
sudo apt-get install build-essential checkinstall cmake pkg-config yasm
sudo apt-get install git gfortran
sudo apt-get install libjpeg8-dev libjasper-dev libpng12-dev

# sudo apt-get install libtiff4-dev  # If you are using Ubuntu 14.04
sudo apt-get install libtiff5-dev  # If you are using Ubuntu 16.04

sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev
sudo apt-get install libxine2-dev libv4l-dev
sudo apt-get install libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev
sudo apt-get install qt5-default libgtk2.0-dev libtbb-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install libfaac-dev libmp3lame-dev libtheora-dev
sudo apt-get install libvorbis-dev libxvidcore-dev
sudo apt-get install libopencore-amrnb-dev libopencore-amrwb-dev
sudo apt-get install x264 v4l-utils

# Optional dependencies
sudo apt-get install libprotobuf-dev protobuf-compiler
sudo apt-get install libgoogle-glog-dev libgflags-dev
sudo apt-get install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen

'''Step 3: Install Python libraries '''
# sudo apt-get install python-dev python-pip python3-dev python3-pip
# sudo -H pip2 install -U pip numpy
# sudo -H pip3 install -U pip numpy
# pip install numpy scipy matplotlib scikit-image scikit-learn ipython

'''Step 4: Download OpenCV and OpenCV_contrib'''
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 3.3.1
cd ..

git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout 3.3.1
cd ..

''' Step 5: Compile and install OpenCV with contrib modules '''
cd opencv
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_QT=ON \
      -D WITH_OPENGL=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D BUILD_EXAMPLES=ON ..


''' Step 5.3: Compile and Install '''
# find out number of CPU cores in your machine
nproc
# substitute 4 by output of nproc
make -j1
sudo make install
sudo sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig



""" #-----------------------------------------------------------------------------------------------------------
#
# AWS Specific
#
#-----------------------------------------------------------------------------------------------------------"""

#check AWS IP
http://checkip.amazonaws.com/


######################
#AWS  Detaching a volume from ec2 and attaching to a new ec2
######################


# Volumes - detach volumes of required instance
Needs to be in the same area

# Ssh into new instance
sudo -i

#list block devices, tells you the device type, always look for second volume
#which drive it mounted
lsblk

# attach any volume into mnt
mount /dev/xvdf1 /mnt

#check permissions
s -al /mnt/home/

#
ls -al /mnt/home/ubuntu/

# replacing old key with new key
cp /home/ubuntu/.ssh/authorized_keys /mnt/home/ubuntu/.ssh/authorized_keys

#check diff
diff /mnt/etc/ssh/sshd_config  /etc/ssh/sshd_config


# Now detach from temp instance and attach volume to old ec2
/dev/sda1


# always create a backup, take a snapshot
Actions->image->create_image

#INcrease Size
 Stop instance

Increase is just the amount of compute




######################
## AWS EC2 notes
######################

# Show hidden
ls -la /home/ubuntu

#copy
sudo cp -R /home/ubuntu/extended-vol/home/ubuntu/Notebook ~/home/ubuntu

sudo cp -R /home/ubuntu/extended-vol/home/ubuntu/  ~/home/ubuntu/extended-vol/home/ubuntu/

aws s3 cp -R /home/ubuntu/extended-vol/home/ubuntu/ s3://wunderman-datascience/instance-backup/


# Copy folder FROM S3 to current folder
aws s3 cp s3://wunderman-datascience/microsoft-content-audit/code/pylonlab/ ./ --recursive

#Unmount vol
sudo umount ~/extended-vol


#Original instance
vol-09e1a8f46cb13bdcc
Instance id:  i-041f57cb1ab8611b9

#remove trash
rm -rf ~/.local/share/Trash/*

# show size of trash
du -hs ~/.local/share/Trash



######################
# Adding Volume to EC2
######################
#http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/recognize-expanded-volume-linux.html


#Find volume for EC2
EC2 Console > Select instance > Scroll down through Description tab > click Root Device > Click the volume ID

#Modify Volume
navigate to Volumes from your EC2 Console > select the volume associated with your instance (vol-09e1a8f46cb13bdcc) > Modify > leave the Volume Type the same but increase the Size > Modify

#SSH into instance
#Check its added. Should see more vol
lsblk

# OUTPUT
# ubuntu@ip-172-31-34-182:~$ lsblk12:11:20
# NAME MAJ:MIN RM SIZE RO TYPE MOUNTPOINT
# xvda 202:0 0 16G 0 disk
# └─xvda1 202:1 0 8G 0 part /

# Now update root
sudo growpart /dev/xvda 1
sudo resize2fs /dev/xvda1






#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# solve by deleting the old host key
# sometimes caused by changing the IP of cluster (I.e when assigning Elastic IP)
ssh-keygen -R ec2-35-165-208-131.us-west-2.compute.amazonaws.com


# Checking an instance
ping ec2-54-244-49-121.us-west-2.compute.amazonaws.com



######################
## AMIs
######################

# However please be aware that any time an AMI is taken..instance will get rebooted...this is recommended for preventing any data inconsistencies that might cause boot issues..
# If you created ami from instance..and launch it...it will create same exact copy..while creating an AMI by default reboot occurs...this is recommended...it is not recommended to check "no reboot option"
