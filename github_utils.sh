###############################
# Setting up Github
###############################


# Set your username
$ git config --global user.name "Your Name Here"

# Set your email address
$ git config --global user.email "your_name@domain.com"

# Check you've got keychain installed
Usage: git credential-osxkeychain <get|store|erase>

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
Hi username! You've successfully authenticated,
but GitHub does not provide shell access.

# Very useful
http://burnedpixel.com/blog/setting-up-git-and-github-on-your-mac/