#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Pandas and general python utils
"""


import numpy as np
import pandas as pd

# Pandas options
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_colwidth', -1)



############################
# Column Formatting
############################

def p2f(x):
    '''
        Convert Percentage to decimal
        E.g. 46.30%	to 0.4630
    '''
    return float(x.strip('%'))/100

google_analytics['% Exit'] = google_analytics['% Exit'].apply(lambda x: p2f(x))
google_analytics['Bounce Rate'] = google_analytics['Bounce Rate'].apply(lambda x: p2f(x))



def get_sec(time_str):
    '''
        Get seconds from time
        E.g. 00:01:06 to 66
    '''
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

google_analytics['Avg. Time on Page'] = google_analytics['Avg. Time on Page'].apply(lambda x: get_sec(x)).astype(float)



############################
# dynamic filtering
############################
# Create a filter dynamically then filter a df
df.query('Cat == 0')

# better example using   widgets.SelectMultiple
str_cond =[]
for x in brand_.value:
    str_cond.append('('+x+'>0)')
brand_str_filter = ' & '.join(str_cond)
brand_str_filter # = '(adventuros>0) or (beyond>0) or (veterinarydiets>0)'
df.query(brand_str_filter)



############################
# dropping
############################

# drop multiple certain columns
df.drop(['columnheading1', 'columnheading2'], axis=1, inplace=True)




############################
# renaming
############################
df.rename(columns={'words_new': 'words', 'unique_words_new': 'unique_words'}, inplace=True)
