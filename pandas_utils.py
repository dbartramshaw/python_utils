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
pd.set_option('display.max_columns', -1)

max_columns

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



# format % on column
df['var3'].map(lambda n: '{:,.2%}'.format(n))

# format % on series
series_example.map(lambda n: '{:,.2%}'.format(n))


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

# Drop duplicate columns
def duplicate_columns(df, return_dataframe = False, verbose = False):
    '''
        a function to detect and possibly remove duplicated columns for a pandas dataframe
    '''
    from pandas.core.common import array_equivalent
    # group columns by dtypes, only the columns of the same dtypes can be duplicate of each other
    groups = df.columns.to_series().groupby(df.dtypes).groups
    duplicated_columns = []

    for dtype, col_names in groups.items():
        column_values = df[col_names]
        num_columns = len(col_names)

        # find duplicated columns by checking pairs of columns, store first column name if duplicate exist
        for i in range(num_columns):
            column_i = column_values.iloc[:,i].values
            for j in range(i + 1, num_columns):
                column_j = column_values.iloc[:,j].values
                if array_equivalent(column_i, column_j):
                    if verbose:
                        print("column {} is a duplicate of column {}".format(col_names[i], col_names[j]))
                    duplicated_columns.append(col_names[i])
                    break
    if not return_dataframe:
        # return the column names of those duplicated exists
        return duplicated_columns
    else:
        # return a dataframe with duplicated columns dropped
        return df.drop(labels = duplicated_columns, axis = 1)



############################
# renaming
############################
df.rename(columns={'words_new': 'words', 'unique_words_new': 'unique_words'}, inplace=True)



############################
# MultiIndex - drop
############################
df.columns = df.columns.droplevel()



# Sort by index
df.sort_index(inplace=True)



############################
# lambda
############################
sample['PR'] = sample['PR'].apply(lambda x: np.nan if x < 90 else x)



############################
# Balance Sample - Random
############################
g = train_data.groupby('BOOKINGS_POST_YN')
train_data_balanced = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)).reset_index(drop=True)




############################
# add time delta (Time to first bookin in period)
############################
import datetime
pre_enddate = datetime.datetime.strptime('30-06-2017', "%d-%m-%Y").date()
post_enddate = datetime.datetime.strptime('30-06-2018', "%d-%m-%Y").date()

df_trans['time_to_first_tran'] = post_enddate - df_trans['FIRST_BOOKING_DT']                 #format = 349 days
df_trans['time_to_first_tran'] =  = [int(i.days) for i in df_trans['time_to_first_tran']]    #format = 349
