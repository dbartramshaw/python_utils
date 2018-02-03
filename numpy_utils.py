#!/usr/bin/env python
# -*- coding: utf-8 -*-




############################
# package utils
############################
import numpy as np
from __future__ import division



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
# numpy notes
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



############################
# Matrices
############################
#Dense column wise
full_dense = np.hstack((word_dense,likes_dense))

#Dense row wise
full_dense = np.vstack((word_dense,likes_dense))

import scipy
full_sparse = scipy.sparse.hstack((word_features,likes_matrix))
full_sparse.shape



############################
# List Comprehensions
############################
[a if a else 2 for a in [0,1,0,3]]

flat_list = [item for sublist in l for item in sublist]
