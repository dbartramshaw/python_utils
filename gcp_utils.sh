## Download  
$ wget https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz
$ tar -zxvf google-cloud-sdk.tar.gz
## INSTALL  
$ bash google-cloud-sdk/install.sh

# Login to gcloud to allow access
gcloud auth login
gcloud auth --project=apache-cluster #specific project login

# Set config for a project
gcloud config set project YOUR-PROJECT-ID-HER

# ssh from terminal
gcloud compute --project "project-name" ssh --zone "us-west1-b" "instance-name"

#login as root
sudo -s

# copy script from storage
gsutil cp -p gs://bucket-name/file-name.sh file-name.sh

# make executable
chmod +x file-name.sh

# run shell file
run ./file-name.sh


# default startup commands
sudo vim /etc/rc.local 

# add this to file
nohup /usr/local/bin/jupyter notebook

# Make sure the jupyter path is right
which jupyter
# /usr/local/bin/jupyter

# copy from s3 to gs (Needs creds set up)
gsutil cp -R s3://bucketname gs://bucketname

# If you have a lot of objects, run with the -m flag to perform the copy in parallel with multiple threads:
sutil -m cp -R s3://bucketname gs://bucketname

# split files
split -l 5000000 od_extract_20181130.csv  

# Needs
gsutil cp -R s3://wunderman-datascience/instance-backups/ci-prod-instance-backup-20-11-2018/nohup.out gs://dbs-loom/config/test_transfer.sh 

