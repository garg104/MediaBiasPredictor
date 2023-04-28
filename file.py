import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from string import punctuation
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from gensim.models import Doc2Vec
import gensim
from gensim.models.doc2vec import TaggedDocument
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans



import warnings
import argparse
import sys
import os

import json

import math
from bs4 import BeautifulSoup

import multiprocessing


#Machine learning libraries
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#Deep Learning libraries
from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K


if __name__ == '__main__':
    train = pd.read_csv("./data/csvs/train_data_m_c.csv")
    test = pd.read_csv("./data/csvs/test_data_m_c.csv")
    
    sources = train['source'].to_numpy()
    content = train['stemmed'].to_numpy()
    media_content = sources + " : " + content
    train['media_content'] = media_content.tolist()
    train.to_csv('train_data_m_m.csv')
    
    sources = test['source'].to_numpy()
    content = test['stemmed'].to_numpy()
    media_content = sources + " : " + content
    test['media_content'] = media_content.tolist()
    test.to_csv('test_data_m_m.csv')
    
    train = []
    test = []
    
    train = pd.read_csv("./data/csvs/train_data_r_c.csv")
    test = pd.read_csv("./data/csvs/test_data_r_c.csv")
    
    sources = train['source'].to_numpy()
    content = train['stemmed'].to_numpy()
    media_content = sources + " : " + content
    train['media_content'] = media_content.tolist()
    train.to_csv('train_data_r_m.csv')
    
    sources = test['source'].to_numpy()
    content = test['stemmed'].to_numpy()
    media_content = sources + " : " + content
    test['media_content'] = media_content.tolist()
    test.to_csv('test_data_r_m.csv')
    
    
    
    # import pdb
    # pdb.set_trace()