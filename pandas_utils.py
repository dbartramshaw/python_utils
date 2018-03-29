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
