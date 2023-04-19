# Import the libraries required
import numpy as np
import pandas as pd
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
import re
from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from gensim.models import Doc2Vec
import gensim
from gensim.models.doc2vec import TaggedDocument

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



def preprocess(data_path):
    # get the files from the folder
    files = os.listdir(data_path)
    data = pd.DataFrame()
    i = 0
    for file in files:
        i = i + 1
        if i > 1000:
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
    for text in data['content']:
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
    # import pdb
    # pdb.set_trace()
    
    # create a countVectorizer
    count_vector = []
    for text in data['content']:
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


def clean(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = text.replace('„','')
    text = text.replace('“','')
    text = text.replace('"','')
    text = text.replace('\'','')
    text = text.replace('-','')
    text = text.lower()
    return text

def remove_stopwords(content):
    stopWords = set(stopwords.words('english'))
    for word in stopWords:
        content = content.replace(' '+word+' ',' ')
    return content

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 3:
                continue
            tokens.append(word.lower())
    return tokens


def acc(true, pred):
    acc = 0
    for x,y in zip(true,pred):
        if(x == y): acc += 1
    return acc/len(pred)

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    classes, features = zip(*[(doc.tags[0],
      model.infer_vector(doc.words)) for doc in sents])
    return features, classes
    

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

    print('done')

    # bag, count_vector = processTrainData(data)

    # import pdb
    # pdb.set_trace()

    train, test = train_test_split(data, test_size=0.2)

    print('train = ',len(train))
    print('test = ',len(test))

    train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['content']), tags=  [r.bias]), axis=1)
    test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['content']), tags=[r.bias]), axis=1)

    print('train = ',len(train_tagged))
    print('test = ',len(test_tagged))

    cores = multiprocessing.cpu_count()
    models = [
    # PV-DBOW 
    Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, sample=0, min_count=2, workers=cores),
    # PV-DM
    Doc2Vec(dm=1, vector_size=300, negative=5, hs=0, sample=0,    min_count=2, workers=cores)
    ]

    for model in models:
        model.build_vocab(train_tagged.values)
        model.train(utils.shuffle(train_tagged.values),
            total_examples=len(train_tagged.values),epochs=30)

    models[0].save("doc2vec_articles_0.model")
    models[1].save("doc2vec_articles_1.model")

    # PV_DBOW encoded text
    train_x_0, train_y_0 = vec_for_learning(models[0], train_tagged)
    test_x_0, test_y_0 = vec_for_learning(models[0], test_tagged)
    # PV_DM encoded text
    train_x_1, train_y_1 = vec_for_learning(models[1], train_tagged)
    test_x_1, test_y_1 = vec_for_learning(models[1], test_tagged)

    svc_0 = SVC()
    svc_1 = SVC()

    svc_0.fit(train_x_0,train_y_0)
    svc_1.fit(train_x_1,train_y_1)
    print(acc(test_y_0,svc_0.predict(test_x_0)))
    print(acc(test_y_1,svc_1.predict(test_x_1)))

    bayes_0 = GaussianNB()
    bayes_1 = GaussianNB()

    bayes_0.fit(train_x_0,train_y_0)
    bayes_1.fit(train_x_1,train_y_1)

    print(acc(test_y_0,bayes_0.predict(test_x_0)))
    print(acc(test_y_1,bayes_1.predict(test_x_1)))

    # Create random forests with 100 decision trees
    forest_0 = RandomForestClassifier(n_estimators=100)
    forest_1 = RandomForestClassifier(n_estimators=100)

    forest_0.fit(train_x_0,train_y_0)
    forest_1.fit(train_x_1,train_y_1)

    print(acc(test_y_0,forest_0.predict(test_x_0)))
    print(acc(test_y_1,forest_1.predict(test_x_1)))

    # count values in train
    left = 0 
    center = 0 
    right = 0
    for i in train['bias']:
        if (i == 0):
            left += 1
        elif(i == 1):
            center += 1
        elif(i == 2):
            right += 1
        
    print(left)
    print(center)
    print(right)



    
    
        
