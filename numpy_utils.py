#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Numpy and general python utils

    Potentially can be sensibly split up at a later point
"""


############################
# Join dicts python2
############################
import numpy as np
from __future__ import division
import collections
from collections import defaultdict


def join_two_dicts(a,
                   b,
                   a_val_name='a_val_name',
                   b_val_name='b_val_name',
                   create_inner_dict = True):

        """
            Useful when both dicts dont have same values
            Use create_inner_dict = True to identify value origins

            Method: Left Join
            Primary key in a: All values must be in a.
        """

        if create_inner_dict == True:
            # create inner dicts
            a = {key:{a_val_name:val} for key,val in a.items()}
            b = {key:{b_val_name:val} for key,val in b.items() if key in a }

        dd = defaultdict(list)
        for d in (a, b): # you can list as many input dicts as you want here
            for key, value in d.iteritems():
                dd[key].append(value)
        return dd



############################
# reload in Python3
############################
import importlib
importlib.reload(module)



############################
# arrays
############################

#reshape to give 1 column
vec.reshape(1,100).shape # = (1,100)


# Reshape to
feature.shape #(286,) #286 rows
feature = feature.reshape(X.shape[0], 1)
feature.shape #(286, 1)


#append
np.append(data,new_col)


#argsort
x = np.array([1.48,1.41,0.0,0.1])
a = x.argsort()
print x[a] #, we will get array([ 0. , 0.1 , 1.41, 1.48]


# setdiff - difference in two arrays
np.setdiff1d(a, b)


# count non zero
tess_nums = [0,0,0,1]
np.count_nonzero(tess_nums)


# numpy where
np.where(tX != 0)[0]


# Lookup multiple values in list
import random
mylist = [random.choice(range(100)) for i in range(10)]
# find the indexs
find_idxs = [i for i,n in enumerate(mylist) if n > 33]
# find values (Make sure Numpy!!)
mylist=np.array(mylist)
mylist[find_idxs]



# Sampling on multiple datasets
ns=np.array(range(164))
size_train = int(round(164*0.66,0))
train_indx = random.sample(range(164), size_train)
test_index = ns[np.isin(ns,train_indx)==False]


#check
len(x_train[test_index])+len(x_train[train_indx])


# Just select first N cols of numpy
test_np = np.random.rand(10,7)
test_np[:,0:2]


# Delete the last item
test_np = np.delete(test_np, -1)


# array of a single value
m = np.zeros((3,3))
m += -1


# Index steps
import numpy as np
# start:stop:step
eg = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])
eg[1::4]


# Change format from mathematical to float
"{:.2f}".format(float("8.99284722486562e-02"))



############################
# Matrices
############################
#Dense column wise
full_dense = np.hstack((word_dense,likes_dense))

#Dense row wise
full_dense = np.vstack((word_dense,likes_dense))

#X = numpy.ndarray
np.asmatrix(X)

import scipy
full_sparse = scipy.sparse.hstack((word_features,likes_matrix))
full_sparse.shape



############################
# Convert an image to co-ordinates and values
############################
"""
    ------------------------------------------------
    Convert an image to co-ordinates and values
    ------------------------------------------------
    IN : Matrix or Image (height=y, width=x)
    OUT: x,y,values

"""
data = img.getdata()
width, height = img.size
pixelList = []
xy_cords =[]
values=[]
for i in range(height):
    for j in range(width):
        stride = (width*i) + j
        pixelList.append((j, i, data[stride]))
        xy_cords.append((j,i))
        values.append((data[stride]))




############################
# List Comprehensions
############################
[a if a else 2 for a in [0,1,0,3]]

flat_list = [item for sublist in l for item in sublist]




############################
# Dictionarys
############################


# create empty dict from keys 
val_dict = dict.fromkeys(['loc','serves','loc_link','serves_link','html'])


def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z


# MULTI DICTS
user_dict = {12: {'Category 1': {'att_1': 1, 'att_2': 'whatever'},
                  'Category 2': {'att_1': 23, 'att_2': 'another'}},
             15: {'Category 1': {'att_1': 10, 'att_2': 'foo'},
                  'Category 2': {'att_1': 30, 'att_2': 'bar'}}}


# Access first dict
user_dict[12]
for i in user_dict.keys():
    print(i)

# Access second dict
for i in user_dict[12].keys():
    print(i)

# Replicate multiple values from each dict
{(i,j): user_dict[i][j]
                           for i in user_dict.keys()
                           for j in user_dict[i].keys()}


# convert multi layer dict to DataFrame
import pandas as pd
pd.DataFrame.from_dict({(i,j): user_dict[i][j]
                           for i in user_dict.keys()
                           for j in user_dict[i].keys()},
                       orient='index')


def depth(d, level=1):
    """ # Find depth of dict """
    if not isinstance(d, dict) or not d:
        return level
    return max(depth(d[k], level + 1) for k in d)

example_dict= {u'address_components': [{u'long_name': u'0',
               u'short_name': u'0',
               u'types': [u'street_number']},
              {u'long_name': u'North Czech Lane',
               u'short_name': u'N Czech Ln',
               u'types': [u'route']},
              {u'long_name': u'Hessel',
               u'short_name': u'Hessel',
               u'types': [u'locality', u'political']},
              {u'long_name': u'Clark Township',
               u'short_name': u'Clark Township',
               u'types': [u'administrative_area_level_3', u'political']},
              {u'long_name': u'Mackinac County',
               u'short_name': u'Mackinac County',
               u'types': [u'administrative_area_level_2', u'political']},
              {u'long_name': u'Michigan',
               u'short_name': u'MI',
               u'types': [u'administrative_area_level_1', u'political']},
              {u'long_name': u'United States',
               u'short_name': u'US',
               u'types': [u'country', u'political']},
              {u'long_name': u'49745',
               u'short_name': u'49745',
               u'types': [u'postal_code']}],
             u'formatted_address': u'0 N Czech Ln, Hessel, MI 49745, USA',
             u'geometry': {u'bounds': {u'northeast': {u'lat': 46.0137765,
                u'lng': -84.46963450000001},
               u'southwest': {u'lat': 46.0137702, u'lng': -84.469651}},
              u'location': {u'lat': 46.0137702, u'lng': -84.469651},
              u'location_type': u'RANGE_INTERPOLATED',
              u'viewport': {u'northeast': {u'lat': 46.0151223302915,
                u'lng': -84.4682937697085},
               u'southwest': {u'lat': 46.0124243697085, u'lng': -84.47099173029152}}},
             u'partial_match': True,
             u'place_id': u'EiMwIE4gQ3plY2ggTG4sIEhlc3NlbCwgTUkgNDk3NDUsIFVTQQ',
             u'types': [u'street_address']}

# Run Example
depth(example_dict)



############################
# DefaultDicts
############################
from collections import defaultdict

#it means expect ints
def_dict = defaultdict(int)
your_list=['a','b','c']
for i in your_list:
    def_dict[i]=0


############################
# Numbers
############################


# Leading 0's i.e 003
str(3).zfill(3)



############################
# Zip
############################

# Zip two lists - into list of tuples
trainingData_test = list(zip(train_data,training_labels))
# Unzip into two lists
X, Y = zip(*trainingData_test)




############################
# Itertools
############################
import itertools

# ???
# # Simple overlap of all tuples (From lastfm_distance_metrics)
# list_of_sets = {'a': {0, 1, 2},'b':{0,4},'c':{2,3,4}}
# intersection_df = pd.DataFrame(columns=['artist_1','artist_2','count_of_overlap'])
# counter =0
# for i in list_of_sets.keys():
#     for j in list_of_sets.keys():
#         intersec= overlap(list_of_sets[i],list_of_sets[j])
#         empty_df = pd.DataFrame({'artist_1':i, 'artist_2':j, 'count_of_overlap':intersec}, index=[0])
#         intersection_df=intersection_df.append(empty_df)
#         counter+=1
# intersection_df.reset_index()


# Print all 4 number combinations
from itertools import combinations, starmap
number = [53, 64, 68, 71, 77, 82, 85]
print list(combinations(number, 4))


# Speed this up using combinations
s = [set([1, 2]), set([1, 3]), set([1, 2, 3]), set([2, 4])]
list(combinations(s,2))

# Combinations of two lists
import itertools
a = [[1,2,3],[4,5,6],[7,8,9,10]]
list(itertools.product(*a))




############################
# Time
############################
import time
from time import gmtime, strftime
print('Start: '+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
### DO SOMETHING
print('End: '+strftime("%Y-%m-%d %H:%M:%S", gmtime()))

import time
start = time.time()
### DO SOMETHING
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))



############################
# Dates
############################
#https://www.tutorialspoint.com/python/time_strptime.htm

import time
from datetime import datetime, timedelta
d = '2017-01-01'

# Convert string  datetime
new_d = datetime.strptime(d, '%Y-%m-%d')

# Convert datetime to UNIX timestamp
unixtime = time.mktime(new_d.timetuple())

# Convert UNIX timestamp to date

timeA = datetime.fromtimestamp(1492286980)
timeA+timedelta(hours=9)

# Current time
current_datetime = datetime.now()
current_datetime.strftime('%x %X')


# convert date to hourr
# adding 00 hours
a = datetime.fromordinal(required_wg.value.toordinal())+timedelta(hours=9)
#convert back to string
a.strftime("%Y-%m-%dT%H:%M:%SZ")



# full month to date
from datetime import datetime
str(dates_test[9]).replace(',','') #'September 21 2015'
new_d = datetime.strptime(str(dates_test[9]).replace(',',''), '%B %d %Y') #datetime.datetime(2015, 9, 21, 0, 0)
new_d.strftime('%x') #'09/21/15'


#get seconds from 00:00:00
def get_sec(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


def str_to_UNIX(str_date):
    #dt = datetime.strptime(str_date, '%Y-%m-%d %H:%M')
    dt = datetim e.strptime(str_date, '%Y-%m-%d')
    unixtime = time.mktime(dt.timetuple())
    return unixtime


def UNIX_to_str(UNIX_date):
    date = datetime.fromtimestamp(UNIX_date)
    #return date.strftime('%x %X') # '11/22/17 14:48:54'
    return date.strftime('%x')




############################
# Text
############################
sentence = 'this is a foo bar sentences and i want to ngramize it'

# Split list. Default by space
sentence.split()

# Unique list - sets are fast
set(sentence.split())

# List to np array
np.asarray(list_name)

# Convert list of unicode to str
clean_text = [x.encode('UTF8') for x in sentence.split()]

# right find
'/fhg/hd'.rfind('/')


from nltk import ngrams
def get_grams(sentence, n):
    """ # Get ngrams from sentance """
    ngrams_gen = ngrams(sentence.split(), n)
    return [' '.join(grams) for grams in ngrams_gen]

# Run Example
sentence = 'this is a foo bar sentences and i want to ngramize it foos'
get_grams(sentence, 2)

sentence.count('foo')


def count_occurrences_ngram(phrase, sentence):
    """ # count the occurences of any word or ngram """
    if len(phrase.split())==1:
        c = sentence.lower().split().count(phrase)
    if len(phrase.split())>1:
        c =  get_grams(sentence.lower(),len(phrase.split())).count(phrase)
    return c

# Run Example
test_sentence = "the cio and chief information officer doesnt really like python but the cio does like java"
count_occurrences_ngram('chief information officer',test_sentence)
count_occurrences_ngram('cio',test_sentence)



def count_occurrences_multi(phrases,sentence):
    """ # Muli phrases counted within one sentance """
    return sum([count_occurrences_ngram(word,sentence) for word in phrases])


# Run Example
job_titles = [['cio','chief information officer'],['cmo','chief marketing officer']]
count_occurrences_multi(job_titles[0],test_sentence)



def store_occurrences_ngram(phrases,sentance):
    """ # store the actual occurences """
    word_list = []
    for word in phrases:
        if count_occurrences_ngram(word,sentance) > 0:
            word_list.append((word ,count_occurrences_ngram(word,sentance)))
    return sorted(word_list, key=lambda x: x[1],reverse=True)

# Run Example
test_products = ['azure government trial','bing maps','surface','microsoft power bi','cortana intelligence suite']
test_text = 'microsoft power bi manufacturers cash data microsoft enterprise enterprise microsoft dynamics crm microsoft power bi surface microsoft power bi'
store_occurrences_ngram(test_products,test_text)



##########################
# Encoding
##########################
import binascii
## Initial byte string
s = b'hello'

## Encode as hex
import binascii
h = binascii.b2a_hex(s)
h
# b'68656c6c6f'

## Decode back to bytes
binascii.a2b_hex(h)
# b'hello'



##########################
# Dir paths
##########################
import os
example_path = '/Users/bartramshawd/Documents/datasets/kaggle_dogbreed_data/test/000621fb3cbb32d8935728e48679680e.jpg'
example_path.split(os.path.sep)


# walk()
import os
for root, dirs, files in os.walk(".", topdown=False):
                        #os.walk("./HRI/Sessions/") ## get subdir
   for name in files:
      print(os.path.join(root, name))
   for name in dirs:
      print(os.path.join(root, name))



##########################
# Optimise Code
##########################
import cProfile
cProfile.run('2 + 2')



##########################
# Missing variable
##########################
# Often used when pulling items
foo = 3
del foo

if 'foo' not in locals():
    var_ = 'foo'
    locals()['foo'] = 'unknown'
foo
