#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Data loading utils in python

    Including
    ------------
    * JSON
    * boto
    * csv
"""



"""##################################
# boto
##################################"""
import boto3

#S3 connection
s3 = boto3.client('s3',
                   aws_access_key_id='XXX',
                   aws_secret_access_key='XXX',
                   region_name='eu-west-1'
                   )


##################################
#Import from s3
##################################
obj = s3.get_object(Bucket='bucket-name', Key = 'path/path/file.csv')
csv_string = obj['Body'].read().decode('utf-8')
df = pd.read_csv(StringIO(csv_string))
dbs_file_test = df[0:10]


##################################
# Upload to specific part of S3
##################################
BucketName = "bucket-name"
myfilename = "myfile_save.csv"
KeyFileName = "path/path/{fname}".format(fname=myfilename)


#From object (dataframe)
s3.put_object(Body=csv_buffer.getvalue(), Bucket=BucketName, Key=KeyFileName)


#From File
with open(myfilename) as f :
  object_data = f.read()
  client.put_object(Body=object_data, Bucket=BucketName, Key=KeyFileName)




"""##################################
# json
##################################"""
import json

# Save
with open('my_dict.json', 'w') as f:
    json.dump(my_dict, f)

# Load
with open('my_dict.json') as f:
    my_dict = json.load(f)


#Update Existing JSON
with open('output_test.json') as f:
    data = json.load(f)
data.update(a_dict)
with open('output_test.json', 'w') as f:
    json.dump(data, f)




############################
# Broken JSONs
############################
from itertools import islice
import json
data_path = 'Documents/folder/'

#Import the broken JSON. This is often when writing has been terminated
with open(data_path+'broken_json.json') as myfile:
    head = list(islice(myfile, 11987))

# the problem will be in the last row
n = len(head)

# reload but removing the last row
with open(data_path+'broken_json.json') as myfile:
    head = list(islice(myfile, n-1))

# Convert to json and resave
json_format_head = json.dumps(head2)
with open(data_path+'working_json.json', 'w') as f:
    json.dump(json_format_head, f)

# Test open
with open(data_path+'working_json') as f:
    my_dict = json.load(f)




"""############################
# Pickle
############################"""
import pickle
import pandas as pd

# Save a dictionary into a pickle file.
favorite_color = { "lion": "yellow", "kitty": "red" }
pickle.dump( favorite_color, open( "save.p", "wb" ) )


# Load the dictionary back from the pickle file.
import pickle
favorite_color = pickle.load( open( "save.p", "rb" ) )


# speed up your pickle access with cPickle
import cPickle as pickle

# Load
favorite_color = pickle.load( open( "save.p", "rb" ) )
with open('path/data_pol_fit.pkl', 'rb') as handle:
        data = pickle.load(handle)



"""############################
# Blob
############################"""
# file path of Unmetric data
file_path = "/file/path/name"
file_type = "xlsx"
fbfiles = glob.glob(file_path+"*."+file_type)
#remove open files ~$
fbfiles = [x for x in fbfiles if x[len(file_path):len(file_path)+2]!="~$"]
