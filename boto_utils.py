#!/usr/bin/env python
# -*- coding: utf-8 -*-

import boto3

##################################
#S3 connection
##################################
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
# Upload to a specific part of S3
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
