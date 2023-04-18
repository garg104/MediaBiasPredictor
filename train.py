# Import the libraries required
import numpy as np
import pandas as pd
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
import warnings
import argparse
import sys
import os

import json

def preprocess(data_path):
    # get the files from the folder
    files = os.listdir(data_path)
    data = pd.DataFrame()
    i = 0
    for file in files:
        i = i + 1
        if i > 10:
            break
        str = "./data/jsons/" + file
        with open(str, 'r') as datafile:
            temp = json.load(datafile)
            # temp = pd.read_json(str, lines=True, orient='split')
            data = data.append(temp, ignore_index = True)
    
    bias = data['bias']
    text = data['content']
        
    data = data.drop("url", axis='columns')
    data = data.drop("source_url", axis='columns')
    
    
    import pdb
    pdb.set_trace()
    
    
    return data
    

def processTrainData(data):
    # preprocessing the text
    # 1. tokenizing the string
    # 2. convert tweets in lower case and split the tweets into tokens(words)
    # 3. remove stop words and punctuations
    # 4. removing commonly used words like # and what not
    # 5. stemming the words
    
    # create a bag of words while preprocessing
    bag = {}
    for text in data['text']:
        # tokenize the words
        words = text.split()
        # print(words)
        
        # remove the stop words
        for word in words:
            if word not in stopwords.words('english'):
                # stemming the words
                word = ps.stem(word)
                if word in bag:
                    bag[word] += 1
                else:
                    bag[word] = 1
    
    # print(len(bag))
    
    # create a countVectorizer
    count_vector = []
    for text in data['text']:
        words = text.split()
        
        # get a Bag of words representation of this sentence
        temp={}
        for word in words:
            if word in temp:
                temp[word] += 1
            else:
                temp[word] = 1
                
        textVector=[]
        for w in bag:
            if w in temp:
                textVector.append(temp[w])
            else:
                textVector.append(0)
                
        count_vector.append(textVector)
           
    count_vector=np.asarray(count_vector)
    # print(len(count_vector))
    
    return bag, count_vector
    

#################################
#                               #
# main                          #
#                               #
#################################

if __name__ == '__main__':
    
    # parser=argparse.ArgumentParser(description='Assignment 1')
    # parser.add_argument('-i', '--train', type=str, required=True, help='Training Data file')
    # args = parser.parse_args()
    # print(args)
    
    ps = PorterStemmer()
    warnings.filterwarnings("ignore")
    np.random.seed(0)
    
    
    # preprocess data
    data = preprocess(data_path='./data/jsons/')
    
    
        
