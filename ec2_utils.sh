
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

#Unmount vol
sudo umount ~/extended-vol


#Original instance 
vol-09e1a8f46cb13bdcc
Instance id:  i-041f57cb1ab8611b9

#remove trash
rm -rf ~/.local/share/Trash/* 




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






#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# solve by deleting the old host key
# sometimes caused by changing the IP of cluster (I.e when assigning Elastic IP)
ssh-keygen -R ec2-35-165-208-131.us-west-2.compute.amazonaws.com                  


# Checking an instance
ping ec2-54-244-49-121.us-west-2.compute.amazonaws.com








