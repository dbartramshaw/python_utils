#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	Utils for text processing
"""

import numpy as np
import sklearn.feature_extraction.text as text
from sklearn import decomposition
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from bs4 import BeautifulSoup


def _removeNonAscii(s):
    return "".join(i for i in s if ord(i)<128)


def clean(strg):
    import re
    from nltk.corpus import stopwords
    """
    Clean string.
        - Removes all non alphabetical characters
        - converts to lower case
        - removes stopwords.words('english') from NLTK library
    ----------
    Parameters
    ----------
    strg : single string
    """
    s = re.sub("[^a-zA-Z,\s]", '', strg)
    s = s.lower()
    sw = stopwords.words('english')
    s = ' '.join([word for word in s.split() if word not in sw])
    return s


def clean_text_iter(strg):
	s = re.sub("[^a-zA-Z]"," ", strg)
	#patterns = ['.co.uk','.com','www.','.net','.org']
	#patterns = ','.join(patterns)
	#s = re.sub('\\b'+patterns+'\\b', ' ', s)
	s = re.sub(" +"," ", s)
	s = s.strip().lower()
	stopw = set(stopwords.words("english"))
	s = ' '.join([word for word in s.split() if word not in set(stopwords.words("english"))])
	return s



def lower_text_clean(strg):
	# Remove non-letters
	s = re.sub("[^a-zA-Z]"," ", strg)
	# Remove multiple white spaces and trailing white spaces
	s = re.sub(" +"," ", s)
	# Convert words to lower case
	s = s.strip().lower()
	return s


remove_format_indicators=['share','events','webcast','event','webinar','forum','microsoft','learn','read']
def remove_selected_words(s,remove_words):
    #remove_words = ['bahasa','Bahasa','january','february','march','april','may','june','july','august','september','october','november','december']
    s = ' '.join([word for word in s.split() if word not in remove_words])
    return s





##############################
# HTML Specific
##############################

def text_from_html(html_unicode):
  from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_unicode,"lxml")
    #Remove script
    for script in soup(["script", "style"]):
        script.extract()
    #remove footer area
    for script in soup.findAll("section", { "id" : ["footerArea","subFooterCTA","contactUs","follow","headerArea"] }):
        script.extract()
    #remove hover text
    for script in soup.findAll("a", { "class" : "screen-reader-text" }):
        script.extract()
    #remove hidden link text
    for script in soup.findAll("span", { "class" : "accessibility-hidden" }):
        script.extract()
    #clean text
    org_text = soup.get_text()
    text = org_text.replace("\n"," ")
    text = ' '.join(text.split())
    return text



##############################
# PARSE NOUNS
##############################

def nltk_noun_parser(corpus_name):
  ''' Word types retained = ['JJ','JJR','JJS','NN','NNP','NNS','NNPS']
      Using nltk.pos_tag'''
  noun_text= list()
  for j in range(0,len(corpus_name)):
      nouns = []
      texta = corpus_name[j]
      for word,pos in nltk.pos_tag(nltk.word_tokenize(str(texta))):
          if (pos=='NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
          #if (pos=='NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS' or pos == 'JJ' or pos == 'JJR' or pos == 'JJS'):
              nouns.append(word)
      stringText=" ".join(nouns)
      stringText = re.sub('[^a-zA-Z]', ' ',stringText)
      noun_text.append(stringText)
  return noun_text




##############################
# Word freq df utils
##############################

# reomove low freq words
def word_freq_per_doc(input_text): #, threshold=10):
  freq = CountVectorizer()
  freq_matrix =  freq.fit_transform(input_text)
  freq_feature_names = freq.get_feature_names()
  freq_df = pd.DataFrame({'doc_index':freq_matrix.nonzero()[0], 'doc_matrix_indices':freq_matrix.nonzero()[1], 'freq':freq_matrix.data})
  freq_df['phrase']=[freq_feature_names[x] for x in freq_df.doc_matrix_indices]
  return freq_df


def words2remove_entire_corpus(input_text, freq_threshold=10):
    freq_df = word_freq_per_doc(input_text)
    word_freq = pd.DataFrame(freq_df.groupby('phrase').sum().reset_index())
    remove_words = word_freq[word_freq.freq<freq_threshold].phrase.values
    return remove_words



def top_freq_words_list(clean_text , n=20):
    """ Generates a list of the top words for each input
        input: List of docs
        output: List of top n phrases & top n phrases with counts in []"""
    from sklearn.feature_extraction.text import CountVectorizer
    freq = CountVectorizer()
    freq_matrix =  freq.fit_transform(clean_text)
    freq_feature_names = freq.get_feature_names()

    freq_df = pd.DataFrame({'doc_index':freq_matrix.nonzero()[0], 'doc_matrix_indices':freq_matrix.nonzero()[1], 'freq':freq_matrix.data})
    freq_df['phrase']=[freq_feature_names[x] for x in freq_df.doc_matrix_indices]

    # Rank
    freq_df['rank']=freq_df.groupby('doc_index')['freq'].rank(ascending=False)
    freq_df['rank_dense']=freq_df.groupby('doc_index')['freq'].rank(ascending=False,method='dense')
    freq_df=freq_df.sort_values(['doc_index','rank'],ascending=True)
    freq_df=freq_df[freq_df['rank']<21]
    freq_df['new_phrase']=freq_df.phrase+'['+freq_df.freq.apply(str)+']'

    top_freq_words = freq_df.groupby('doc_index').agg({'phrase':lambda x:', '.join(x)}).reset_index()
    top_freq_words['new_phrase'] = freq_df.groupby('doc_index').agg({'new_phrase':lambda x:', '.join(x)}).reset_index()['new_phrase']
    return top_freq_words



def top_freq_words(clean_text,labels,ngram=(1,1)):
    """ Generates a df with word counts for each doc
        input: List of docs, choice of ngrams
        output: df with each doc, count, rank and dense rank of each word"
    """
    from sklearn.feature_extraction.text import CountVectorizer
    freq = CountVectorizer(ngram_range=ngram)
    freq_matrix =  freq.fit_transform(clean_text)
    freq_feature_names = freq.get_feature_names()

    #Create DF output
    freq_df = pd.DataFrame({'doc_index':freq_matrix.nonzero()[0], 'doc_matrix_indices':freq_matrix.nonzero()[1], 'freq':freq_matrix.data})
    freq_df['phrase']=[freq_feature_names[x] for x in freq_df.doc_matrix_indices]
    freq_df['label']=[labels[x] for x in freq_df.doc_index]
	#freq_df['label']=[final_article_text['group_name'].values[x] for x in freq_df.doc_index]

    # Rank
    freq_df['rank']=freq_df.groupby('doc_index')['freq'].rank(ascending=False)
    freq_df['rank_dense']=freq_df.groupby('doc_index')['freq'].rank(ascending=False,method='dense')
    return freq_df



def top_tfidf_words(clean_text,labels,ngram=(1,1)):
    """ Generates a df with tfidf scores  for each doc
        input: List of docs, choice of ngrams
        output: df with each doc, tfidf_score, rank and dense rank of each word"
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    tf = TfidfVectorizer(ngram_range=ngram)
    tfidf_matrix =  tf.fit_transform(clean_text)
    tfidf_feature_names = tf.get_feature_names()

    #Create DF output
    tfidf_df = pd.DataFrame({'doc_index':tfidf_matrix.nonzero()[0], 'doc_matrix_indices':tfidf_matrix.nonzero()[1], 'tfidf':tfidf_matrix.data})
    #pd.DataFrame({'doc_index':tfidf_matrix.nonzero()[0], 'doc_matrix_indices':tfidf_matrix.nonzero()[1], 'tfidf':tfidf_matrix.data})
    tfidf_df['phrase']=[tfidf_feature_names[x] for x in tfidf_df.doc_matrix_indices]
    tfidf_df['label']=[labels[x] for x in tfidf_df.doc_index]

    # Rank
    tfidf_df['rank']=tfidf_df.groupby('doc_index')['tfidf'].rank(ascending=False)
    tfidf_df['rank_dense']=tfidf_df.groupby('doc_index')['tfidf'].rank(ascending=False,method='dense')
    return tfidf_df
