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
x = numpy.array([1.48,1.41,0.0,0.1])
a = x.argsort()
print x[a] #, we will get array([ 0. , 0.1 , 1.41, 1.48]


# setdiff - difference in two arrays
np.setdiff1d(a, b)

# count non zero
tess_nums = [0,0,0,1]
np.count_nonzero(tess_nums)

# numpy where
np.where(tX != 0)[0]




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
# List Comprehensions
############################
[a if a else 2 for a in [0,1,0,3]]

flat_list = [item for sublist in l for item in sublist]




############################
# Dictionarys
############################
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
# Zip
############################

# Zip two lists - into list of tuples
trainingData_test = list(zip(train_data,training_labels))
# Unzip into two lists
X, Y = zip(*trainingData_test)



############################
# TIMES
############################
from time import gmtime, strftime
print('Start: '+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
### DO SOMETHING
print('End: '+strftime("%Y-%m-%d %H:%M:%S", gmtime()))


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


############################
# Text functions
############################
from nltk import ngrams


def get_grams(sentence, n):
    """ # Get ngrams from sentance """
    ngrams_gen = ngrams(sentence.split(), n)
    return [' '.join(grams) for grams in ngrams_gen]

# Run Example
sentence = 'this is a foo bar sentences and i want to ngramize it'
get_grams(sentence, 2)



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
