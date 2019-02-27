###############################
# Setting up Github
###############################


# Set your username
$ git config --global user.name "Your Name Here"

# Set your email address
$ git config --global user.email "your_name@domain.com"

# Check you've got keychain installed
git credential-osxkeychain
# you should see this - Usage: git credential-osxkeychain <get|store|erase>

# Point the terminal to the directory that would contain SSH keys for your user account.
$ cd ~/.ssh

# Ensure that you are in your ~/.ssh folder
$ cd ~/.ssh

# Create a new ssh key using the provided email. The email you use in this step should match the one you entered when you created your Github account
$ ssh-keygen -t rsa -C "your_email@domain.com"

# The below command will copy your newly generated key to your computer's clipboard.
$ pbcopy < ~/.ssh/id_rsa.pub

# Now we’ll add your key to Github:
# 	Visit your account settings.
# 	Click Add SSH key.
# 	Enter a descriptive title for the computer you’re currently on, e.g. “Work iMac” into the Title field.
# 	Paste your key into the Key field (it has already been copied to your clipboard).
# 	Click Add Key.
# 	Enter your Github password.

# Attempts to connect to Github using your SSH key.
# Don't change the address shown below
$ ssh -T git@github.com

# You may see the following warning:
The authenticity of host 'github.com (207.97.227.239)'
cant be established.
RSA key fingerprint is 16:27:ac:a5:76:28:2d:36:63:1b:56:4d:eb:df:a6:48.
Are you sure you want to continue connecting (yes/no)?

# Type yes and press return
# You may have to enter your recently selected passphrase.

# You should then see:
Hi username! Youve successfully authenticated,
but GitHub does not provide shell access.

# Very useful
http://burnedpixel.com/blog/setting-up-git-and-github-on-your-mac/



###############################
# Github best pratcise
###############################
# useful resource https://realpython.com/python-git-github-intro/

#adding .gitignore for __pycache__(python 3) or .pyc files (python 3)
# create text file called .gitignore
"""
# .gitignore
__pycache__
venv
env
.pytest_cache
.coverage
"""
git add .gitignore

#this should also contain your virual environment folders
# https://realpython.com/python-virtual-environments-a-primer/

# Python 2:
$ virtualenv env

# Python 3
$ python3 -m venv env


"""
## AVOID STORING OUTPUT FILES
Git does not store a full copy of each version of each file you commit. 
Rather, it uses a complicated algorithm based on the differences between 
subsequent versions of a file to greatly reduce the amount of storage it needs. 
Binary files (like JPGs or MP3 files) don’t really have good diff tools, 
so Git will frequently just need to store the entire file each time it is committed.
"""

## PREVIOUS VERSIONS
#look at the git log to see changes made and where to move SHA
git log 


#you can the revert back to previous SHA (This changes the HEAD) - this will change your filesystem but not loose most recent changes
git checkout 946b99bfe1641102d39f95616ceaab5c3dc960f9

#get back to master (Points HEAD back to master)
git checkout master 



##CREATE BRANCH (for project tasks not yet complete - when complete we push to master)
git checkout -b my_new_feature


# compare differences in branch and master (branching - https://learngitbranching.js.org/)
git show-branch my_new_feature master

#merge with master (you can also use git cherry-pick to move single files)
git merge my_new_feature

