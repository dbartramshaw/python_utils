
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
jupyter notebook stop 8888Well


# execute .sh
# Give execute permission to your script:
chmod +x /path/to/yourscript.sh

# And to run your script:
/path/to/yourscript.sh

#Since . refers to the current directory: if yourscript.sh is in the current directory, you can simplify this to:
./yourscript.sh


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

#start screen escreen
screen -S dbs_screen

#list screens
screen -ls

# detach form screen
Ctrl + A then Ctrl + D

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


######################
# # Fecked up pip
######################
# run the brew doctor - follow instructions

# Will still run
python3 -m pip install --upgrade pip 
python -m pip install --upgrade pip 

# wont run
pip install numpy

# Change of ownership -  dint work
sudo chown -R bartramshawd pip

#reset the PATH variable
#Create a version of bash_profile (Save this as my_profile) 

# #------------------------------------
# export PATH=/usr/local/bin:$PATH

# #--- Terminal en colores
# export TERM=xterm-color
# export CLICOLOR=1
# export LSCOLORS=ExgxdxdxfxdxdxfxfxExEx

# #—— Alias
# alias rm='rm -i'
# # (Adding this final Alias makes terminal check everytime you remove something)


cp my_profile .bash_profile # Homebrew

# copy this to a .bash_profile file
cp my_profile .bash_profile

# Make this the source
source .bash_profile

# run the brew doctor - follow instructions
brew postinstall python
brew postinstall python3
sudo easy_install pip






######################
# Space
######################
# du command: Estimate file space usage.
# -h : Print sizes in human readable format (e.g., 10MB).
# -S : Do not include size of subdirectories.
# -s : Display only a total for each argument.
# sort command : sort lines of text files.
# -r : Reverse the result of comparisons.
# -h : Compare human readable numbers (e.g., 2K, 1G).
# head : Output the first part of files

# To display the largest folders/files including the sub-directories, run:
du -Sh | sort -rh | head -5

# top 50 files
find -type f -exec du -Sh {} + | sort -rh | head -n 50




