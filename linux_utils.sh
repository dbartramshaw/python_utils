
""" -----------------------------------------------------------------------------------------------------------
#
# Linux code for Python/Machine config 
#
#----------------------------------------------------------------------------------------------------------- """



######################
# Core Linux
######################

# Show all (even hidded)
ls -a

# Show all (show details)
ls -l

#open current location in finder
open . 

# count files in folder
# ls:list , -1: only one entry per line , |: pipe output onto... , wc:"wordcount"   , -l: count lines.
ls -1 | wc -l

# run jupyter on certain port
jupyter notebook --port=8889
jupyter notebook list
jupyter notebook stop 8888




######################
# Processes
######################

# Show processes
top

# Find processes
sudo pgrep python

# kill processes
sudo pkill python [certain number process]

# free space
df




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
# Screen - Ubuntu Ec2
######################

#start screen 
screen -S dbs_screen

#list screens
screen -ls

#open certain screen 
screen -r myprog

#quit screens
screen -X -S screen_name  quit

# quit all screens
pkill screen


######################
# Terminal Install python 
######################
# problems with hydrogen not saving in packages but in .npm
apm install hydrogen  




######################
# GIT
######################


#Check git details
git config --list 
git config user.name
git config user.email


# config set
git config --global user.name "Bartram-Shaw, David"
git config --global user.email "david.bartram-shaw@account.com"



# Created new repo using - go into the directory you wish to create
echo "# repo Name" >> README.md
git init
git add README.md
git add -A
git commit -m "Repo initiation"
git remote add origin https://david.bartram-shaw-loc-goes-here.git
git push -u origin master



# Add/Update files to GIT
git init
git add -A
git push
git commit -m "Notes added"
git push origin master


#disconnect from git repo
rm -rf .git





######################
# python setup 2&3
######################
# Fixes the multiple running of pythons
python2 -m pip install ipykernel
python2 -m ipykernel install --user

sudo python3 -m pip install ipykernel
sudo python3 -m ipykernel install --user

