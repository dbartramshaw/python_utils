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

def _removeNonAscii(s):
    s = ' '.join(s.split())
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



def clean_text_iter_rm(strg,remove_stopwords=False):
    text = remove_non_ascii(strg)
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


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

def cosine_similairy_df(matrixA,matrixB):
    import scipy.spatial as sp
    cosine_df = pd.DataFrame(1 - sp.distance.cdist(matrixA, matrixB, 'cosine'))
    cs_df_reshaped = pd.DataFrame(cosine_df.stack()).reset_index()
    cs_df_reshaped.columns=['doc_index','compared_doc_index','cosine_similarity']
    cs_df_reshaped['rank']=cs_df_reshaped.groupby('doc_index')['cosine_similarity'].rank(ascending=False)
    return cs_df_reshaped


class word_features_sim(object):
        """
            ----------------------------------------
            WORD FEATURE & SIMILARITY GENERATION
            ----------------------------------------
            Features are created on eitehr TFIDF or FREQ levels
            Outputs are dataframe formatted for visualisation & Dictionary form for processing

            Parameters:
    		-----------
    		input_text     :  np.array. text (Documents)
            vec_type       : 'tfidf' or 'freq'

    		Returns:
    		-----------
    		self: update of the class attributes

            self.word_features (tfidf matrix)
            self.feature_names
            self.top_words_per_doc
            self.top_similar_docs

        """

        def __init__(self
                    ,input_text
                    ,vec_type='tfidf'
                    ,_sublinear_tf=False
                    ,_ngram_range=(1,2)
                    ,_norm=None
                    ,_max_features=10000
                    ,_vocabulary=None
                    ,_stopwords='English'
                    ,_min_df=1
                    ,_max_df=1000
                    ):

            if vec_type=='tfidf':
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer(analyzer='word',stop_words='english',ngram_range=_ngram_range,max_features=_max_features,vocabulary=_vocabulary,min_df=_min_df,max_df=_max_df,sublinear_tf=_sublinear_tf,norm=_norm)

            if vec_type=='freq':
                from sklearn.feature_extraction.text import CountVectorizer
                vectorizer = CountVectorizer(analyzer='word',stop_words='english',ngram_range=_ngram_range,max_features=_max_features,vocabulary=_vocabulary,min_df=_min_df,max_df=_max_df)

            self.word_features = vectorizer.fit_transform(input_text)
            self.feature_names = vectorizer.get_feature_names()
            #print('-----------------------------------------')
            print('COMPLETE: '+vec_type+' Word Features Generated')

            # Store all word that appear in each doc and TFIDF score
            _df = pd.DataFrame({'doc_index':self.word_features.nonzero()[0], 'doc_matrix_indices':self.word_features.nonzero()[1], vec_type:self.word_features.data})
            _df['phrase']=[self.feature_names[x] for x in _df.doc_matrix_indices]
            _df = _df.sort_values(['doc_index',vec_type],ascending=[1,0])
            _df['rank']=_df.groupby('doc_index')[vec_type].rank(ascending=False)
            self._df = _df

        def top_words(self,n_top_words=20):
            # Store top words
            self.top_words_per_doc = self._df[self._df['rank']<=n_top_words]
            self.top_words_per_doc = self.top_words_per_doc.groupby('doc_index').agg({'phrase':lambda x:', '.join(x)}).reset_index()
            print('COMPLETE: Top words generated')

        def doc_similarity(self
                          ,labels=[]
                          ,n_similar_docs=20
                          ):
            """
                -----------
                Parameters:
        		-----------
                labels: list of labels that correspond to input_text_index
            """
            # Top similar between each doc
            cs_df_reshaped = cosine_similairy_df(self.word_features.todense() , self.word_features.todense())
            self.top_similar_docs = cs_df_reshaped[cs_df_reshaped['rank']<=n_similar_docs].sort_values(['doc_index','rank'],ascending=[1,1])
            if labels!=[]:
                self.top_similar_docs['label']         = [labels[x] for x in self.top_similar_docs.doc_index]
                self.top_similar_docs['compare_label'] = [labels[x] for x in self.top_similar_docs.compared_doc_index]

            print('COMPLETE: Similarity computed')
            #print('-----------------------------------------')


###############################################################
# COMPUTE SIMILARITY WITH DOCS
###############################################################
class industry_doc_similarity(object):
        """
            Inputs:
            -----------
            gold_labels_df: The ground truth labels from sitemap


            Returns:
            -----------
            cs_df: similarity of all urls to each indutry
            cs_max_similarity: max_similarity indutry only
        """

        def __init__(self
                    ,url_text=None
                    ,url_lables=None
                    ,industry_text=None
                    ,industry_labels=None
                    ,exclude_none_class = True
                    ,gold_labels_df = None
                    ,test_col = 'industry_predicted'
                    ,gold_col = 'industries_name'
                    ):
            #global url_tf,industry_tf,cs_df,cs_max_similarity,cs_df_check,url_tf_df,industry_tf_df

            #gold_labels_df = None
            print('----- URL LEVEL -----')
            self.url_tf = word_features_sim(url_text,vec_type='tfidf',_min_df=2,_max_df=500,_norm='l2')
            self.url_tf.top_words(n_top_words=20)
            self.url_tf.doc_similarity(labels=url_lables,n_similar_docs=len(url_text))
            self.url_tf_df = pd.DataFrame(self.url_tf.word_features.todense(),index=url_lables,columns=self.url_tf.feature_names)

            print('----- INDUSTRY LEVEL -----')
            # Ensure the same vocab is used from root: _vocabulary=url_tf.feature_names
            self.industry_tf = word_features_sim(industry_text,vec_type='tfidf',_vocabulary=self.url_tf.feature_names,_min_df=2,_max_df=6,_norm='l2')
            self.industry_tf.top_words(n_top_words=20)
            self.industry_tf.doc_similarity(labels=industry_labels,n_similar_docs=len(industry_text))
            self.industry_tf_df = pd.DataFrame(self.industry_tf.word_features.todense(),index=industry_labels,columns=self.industry_tf.feature_names)


            print('----- SIMILARITY CALC -----')
            self.cs_df = cosine_similairy_df(self.url_tf.word_features.todense(),self.industry_tf.word_features.todense())
            self.cs_df.columns = ['page_index','industry_index','cosine_similarity','rank']
            self.cs_df['url']=[url_lables[x] for x in self.cs_df.page_index]
            self.cs_df[test_col]=[industry_labels[x] for x in self.cs_df.industry_index]

            self.cs_df['top_quintile']= (self.cs_df['cosine_similarity']>=self.cs_df.cosine_similarity.quantile(0.9)).astype(int)
            self.cs_df['top2_quintile']= (self.cs_df['cosine_similarity']>=self.cs_df.cosine_similarity.quantile(0.8)).astype(int)
            self.cs_df = self.cs_df.sort_values(['page_index','rank'],ascending=[1,1])
            print('COMPLETE: Similarity Calculation')

            if exclude_none_class == True:
                self.cs_df_no_other=self.cs_df[self.cs_df[test_col]!='None'].reset_index()
                self.cs_max_similarity = self.cs_df_no_other.loc[self.cs_df_no_other.groupby(['page_index'])['rank'].idxmin()]
                print('COMPLETE: Classification - "None" category removed')
                #print(len(self.cs_df_no_other))
            else:
                self.cs_max_similarity = self.cs_df.loc[self.cs_df.groupby(['page_index'])['rank'].idxmin()]
                print('COMPLETE: Classification - No removal')
                #print(len(self.cs_df))


            if gold_labels_df is not None:
                top_similarity_class = pd.merge(self.cs_max_similarity,gold_labels_df, on='url', how='left')

                # Only measure for those with an industry tag
                top_similarity_class=top_similarity_class[top_similarity_class['industries_tag_yn']==True]
                top_similarity_class['correct_yn']= top_similarity_class[test_col]==top_similarity_class[gold_col]
                tested_items = top_similarity_class

                #Test where industry_tag=True
                total_urls = len(top_similarity_class.url.unique())
                total_correct=len(top_similarity_class[(top_similarity_class.correct_yn==True)].url.unique())

                print('Accuracy: '+str(total_correct/total_urls))
                print('Correct: '+str(total_correct)+'/'+str(total_urls))

            print('-----------------------------------------')



# # RUN IT
# root_ =industry_doc_similarity2(url_text=clean_text_dict.values()
#                                ,url_lables=clean_text_dict.keys()
#                                ,industry_text=tag_corpus['text'].values
#                                ,industry_labels=tag_corpus['industry'].values
#                                ,exclude_none_class = True
#                                ,gold_labels_df = industry_feature.sitemap_df[['url','industries_tag_yn','industries_name']]
#                                ,test_col = 'industry_predicted'
#                                ,gold_col = 'industries_name'
#                                 )
# root_.__dict__.keys()



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
