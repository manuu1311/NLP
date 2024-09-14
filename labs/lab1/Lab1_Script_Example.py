#  Sample by Y. Bounab'


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 08:19:33 2020

@author: polo
"""
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

# Resources for scrabing data -- Two methods are highlighted here beatiful soup or using Wikipedia API

import wikipedia

from urllib.error import HTTPError
from urllib.request import urlopen

from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


'https://levelup.gitconnected.com/two-simple-ways-to-scrape-text-from-wikipedia-in-python-9ce07426579b'
'https://www.datacamp.com/community/tutorials/stemming-lemmatization-python?utm_source=adwords_ppc&utm_campaignid=898687156&utm_adgroupid=48947256715&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=255798340456&utm_targetid=aud-299261629574:dsa-429603003980&utm_loc_interest_ms=&utm_loc_physical_ms=1005766&gclid=CjwKCAjwwab7BRBAEiwAapqpTN9AkZuHgQZK__Jj6Wrkh_2xHt3DUGLkdURgxSetxlRFJ9xY3XVwkRoCD-0QAvD_BwE'

BaseLink = 'https://en.wikipedia.org'

Q = 'I wanted to visit Oulu this summer if possible, otherwise I will go to Helsinki'

Finnish_Cities = {}

def WIKI(title):
    wiki = wikipedia.page(title)
    text = wiki.content
    text = text.replace('==', '')

    text = text.replace('\n', '')[:-12]
    return text

def get_Text(link):
    soup = BeautifulSoup(urlopen(link),'lxml')
    soup = soup.find('div', id='mw-content-text')#.find('div',)
    Text = ''
    for item in soup.find_all('p'):#, recursive=False):
        Text += item.text.strip()
    return Text

def get_Finnish_Cities():
    soup = BeautifulSoup(urlopen('https://en.wikipedia.org/wiki/List_of_cities_and_towns_in_Finland'),'lxml')
    soup = soup.find('div',class_='mw-parser-output')
    Table = soup.find('table', style="text-align:right;").find('tbody')
    
    for item in Table.find_all('tr'):
        if item.find('td'):
           name = item.find('td').find('a')
           if name:
              Finnish_Cities[name['title']] = get_Text(BaseLink+name['href'])
           else:
               name = item.find_all('td')[1].find('a')
               Finnish_Cities[name['title']] = get_Text(BaseLink+name['href'])
               
        
def preProcess(doc):
    Stopwords = list(set(nltk.corpus.stopwords.words('english')))
    stemmer = SnowballStemmer("english")
    WN_lemmatizer = WordNetLemmatizer()

    sentences = sent_tokenize(doc)
    Tokens = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [stemmer.stem(word) for word in words]
        words = [WN_lemmatizer.lemmatize(word, pos="v") for word in words]
        
        words = [word for word in words if word.isalpha() and word not in Stopwords] #get rid of numbers and Stopwords
        #words= [word for word in words if word.isalnum() and word not in Stopwords]
        Tokens.extend(words)
        
    return ' '.join(Tokens)
    
def BOW_model(corpus):
    BOW = CountVectorizer(preprocessor=preProcess)    #max_df=0.8, min_df=0.2, )  in case you want to reduce the size of the dictionary, you can change the default values of max_de, min_def and other parameters
    BOW.fit(corpus)
    #X = BOW.transform(corpus)
    #X = BOW.fit_transform(corpus)
    return BOW

def TFIDF(corpus):
    Tfidf = TfidfVectorizer(preprocessor=preProcess)
    Tfidf.fit(corpus)
#    print(Tfidf.get_feature_names())
#    print(Tfidf.vocabulary_)

    #X = Tfidf.transform(corpus)
    #X = Tfidf.fit_transform(corpus)
    return Tfidf

def Answering_Q(Q, Vectorizer):
    Vectors = Vectorizer.transform(list(Finnish_Cities.values())).toarray()
    Vq = Vectorizer.transform([Q]).toarray()[0]
    
    Scores = []
    for V in Vectors:
        Scores.append(np.inner(Vq, V))
        
    Max_index = Scores.index(max(Scores))
    print('City of',list(Finnish_Cities)[Max_index])
    print('___________________________')
    #print(Finnish_Cities[list(Finnish_Cities)[Max_index]])
    
    
get_Finnish_Cities()

Tfidf = TFIDF(list(Finnish_Cities.values()))
BOW = BOW_model(list(Finnish_Cities.values()))

print('_____________Using TFIDF_____________')
Answering_Q(Q, Tfidf)

print('______________Using BOW______________')
Answering_Q(Q, BOW)




