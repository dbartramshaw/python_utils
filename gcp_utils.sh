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