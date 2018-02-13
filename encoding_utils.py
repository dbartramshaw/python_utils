#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

## RUN IT
# MultiColumnLabelEncoder(columns = cols_to_transform).fit_transform(offer_data_df)
# MultiColumnLabelEncoder().fit_transform(example_data.drop('weight',axis=1))


def balanced_sample_maker_limit(X, y, random_seed=None):
    """ return a balanced data set by oversampling minority class
        current version is developed on assumption that the positive
        class is the minority.

    Parameters:
    ===========
    X: {numpy.ndarrray}
    y: {numpy.ndarray}
    """
    uniq_levels = unique(y)
    uniq_counts = {level: sum(y == level) for level in uniq_levels}
    print '___Initial_balance___'
    print ''
    print uniq_counts
    limit = raw_input("Enter Sample size limit (ENTER None if no limit required):")

    if limit =='None':
        limit = 'None'
    else:
        limit =int(limit)

    if not random_seed is None:
        random.seed(random_seed)

    # find observation index of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level] ## pull back indexes for each value (0,1)
        groupby_levels[level] = obs_idx                              ## insert into dict as levels (0,1)

    # oversampling/limiting observations of negative label
    if limit == 'None' and uniq_counts[0]>uniq_counts[1]:
        print 'Sampling Rule: None & uniq_counts[0]>uniq_counts[1]'
        sample_size = uniq_counts[uniq_levels[0]]
        sample_inx = groupby_levels[uniq_levels[0]]

    elif limit =='None' and uniq_counts[1]>uniq_counts[0]:
        print 'Sampling Rule: None & uniq_counts[1]>uniq_counts[0]'
        sample_inx = random.choice(groupby_levels[uniq_levels[0]], size=uniq_counts[1], replace=True).tolist()
        sample_inx = np.sort(sample_inx)
        sample_size = len(sample_inx)

    elif limit>0:
        print 'Sampling Rule: Limited Volume'
        sample_inx = random.choice(groupby_levels[uniq_levels[0]], size=limit, replace=False).tolist()
        sample_inx = np.sort(sample_inx)
        sample_size = len(sample_inx)
    else:
        print 'Error'

    # oversampling/limiting observations of positive label
    if limit!='None' and limit>uniq_counts[1]:
        replace_y = True
    elif limit=='None' and uniq_counts[0]>uniq_counts[1]:
        replace_y = True
    else:
        replace_y = False

    pos_sample_idx = random.choice(groupby_levels[uniq_levels[1]], size=sample_size, replace=replace_y).tolist()  #randomly select obervations from 1' to match sample size
    balanced_copy_idx_new = np.append(sample_inx,pos_sample_idx)
    #random.shuffle(balanced_copy_idx)
    print ''
    print '___Balancing complete with a limit of '+str(limit)+'___'
    print '{0.0: '+str(len(sample_inx))+', 1.0: '+str(len(pos_sample_idx))+'}'
    return X[balanced_copy_idx_new, :], y[balanced_copy_idx_new]

## Balance Testing
X_new, Y_new = balanced_sample_maker_limit(X,Y)
