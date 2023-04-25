# Import the libraries required
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



def preprocess(data_path):
    # get the files from the folder
    files = os.listdir(data_path)
    data = pd.DataFrame()
    i = 0
    for file in files:
        i = i + 1
        if i > 37000:
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


def remove_stopwords(text):
    stops = list(set(stopwords.words('english'))) + list(punctuation) + ['s', "'", 't', 'and', '"', 'a', 'or', '/', 'in',
                                                               'for', '&', '-', "''"]
    text_no_stops = ''
    lemmatizer = nltk.stem.WordNetLemmatizer()
    
    words = text.split()
    # import pdb
    # pdb.set_trace()

        
    # remove the stop words
    for word in words:
        if word not in stops:
            # text_no_stops += (' ' + lemmatizer.lemmatize(word))
            text_no_stops += (' ' + ps.stem(word.lower()))
            ps.stem(word)
    
    return text_no_stops


def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 3:
                continue
            tokens.append(word.lower())
    return tokens


def train_SVC(train_x_0, train_y_0, train_x_1, train_y_1):
    svc_0 = SVC()
    svc_1 = SVC()

    svc_0.fit(train_x_0,train_y_0)
    svc_1.fit(train_x_1,train_y_1)

    return svc_0, svc_1


def train_NB(train_x_0, train_y_0, train_x_1, train_y_1):
    bayes_0 = GaussianNB()
    bayes_1 = GaussianNB()

    bayes_0.fit(train_x_0,train_y_0)
    bayes_1.fit(train_x_1,train_y_1)

    return bayes_0, bayes_1


def train_RF(train_x_0, train_y_0, train_x_1, train_y_1):
    # Create random forests with 100 decision trees
    forest_0 = RandomForestClassifier(n_estimators=100)
    forest_1 = RandomForestClassifier(n_estimators=100)

    forest_0.fit(train_x_0,train_y_0)
    forest_1.fit(train_x_1,train_y_1)

    return forest_0, forest_1


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


def prepare_data_keras(train_x,train_y,test_x,test_y):
    tx = np.asarray(train_x)
    ty = np.asarray(train_y)
    tex = np.asarray(test_x)
    tey = np.asarray(test_y)
    
    ty = np.asarray(list([np.asarray([0,0,1]) if el == 0 else np.asarray([0,1,0]) 
                    if el == 1 else np.asarray([1,0,0]) for el in ty]))
    tey = np.asarray(list([np.asarray([0,0,1]) if el == 0 else np.asarray([0,1,0])
                    if el == 1 else np.asarray([1,0,0]) for el in tey]))
    
    return tx,ty,tex,tey
  
  
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def vectorize_texts(list_of_strings):
    print('Performing vectorization and TF/IDF transformation on texts...')
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(list_of_strings)
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(X)
    
    # import pdb
    # pdb.set_trace()
    return tfidf


def cluster_texts(num_clusters, tfidf):
    #perform kmeans clustering for range of clusters
    print('Beginning KMeans Clustering, number of clusters = ', num_clusters, '\n') 
    km = KMeans(n_clusters=num_clusters, max_iter = 100, verbose = 2, n_init = 1).fit(tfidf)
    
    return km




def get_most_common_words(df, num_words, clusters):
    cluster_dic = {}
    for i in range(df.shape[0]):
        if cluster_dic.get(df.loc[i]['cluster']):
            cluster_dic[df.loc[i]['cluster']] = cluster_dic[df.loc[i]['cluster']] + " " + df.loc[i]['stemmed']
        else: 
            cluster_dic[df.loc[i]['cluster']] = df.loc[i]['stemmed']
    
    
    
    common_words = {}
    from collections import Counter
    for i in range(clusters):
        content = cluster_dic[i]
        common = Counter(content.split()).most_common(num_words)
        list = []
        for j in common:
            list.append(j[0])
        common_words[i] = list
    

            
    return common_words


def tag_data(train, test):
    train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['content']), tags=  [r.bias]), axis=1)
    test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['content']), tags=[r.bias]), axis=1)

    return train_tagged, test_tagged


#################################
#                               #
# main                          #
#                               #
#################################

if __name__ == '__main__':
    
    ps = PorterStemmer()
    warnings.filterwarnings("ignore")
    np.random.seed(0)
    
    
    ########## preprocess data ##########
    data = preprocess(data_path='./data/jsons/')


    ########## random division ##########
    # train2, test2 = train_test_split(data, test_size=0.2)

    # train2.to_csv('train_data_r.csv')
    # test2.to_csv('test_data_r.csv')

    ########## media division ##########

    # list_sour = []
    # dfu = pd.DataFrame()
    # dfu = data['source'].value_counts(ascending = True)
    # list_sour = data['source'].value_counts(ascending = True).index.to_list()
    # dfu = dfu.to_frame()

    # x = 0.2*37000
    # i = 0
    # count = 0

    # list_test = []
    # list_train = []

    # for source_count in dfu['source']:
    #     count += source_count
    #     list_test.append(list_sour[i])
    #     if (count > x):
    #         break
    #     i += 1

    # for source in list_sour:
    #     if source not in list_test:
    #         list_train.append(source)
    

    # train1 = pd.DataFrame()
    # test1 = pd.DataFrame()

    # for index, row in data.iterrows():
    #     if row['source'] in list_train:
    #         train1 = train1.append(row, ignore_index = True)
    #     if row['source'] in list_test:
    #         test1 = test1.append(row, ignore_index = True)

    # train1.to_csv('train_data_m.csv')
    # test1.to_csv('test_data_m.csv')

    ########## getting data from csv files ##########

    train = pd.DataFrame()
    test = pd.DataFrame()

    train = pd.read_csv("train_data_r.csv")
    test = pd.read_csv("test_data_r.csv")

    train_tagged, test_tagged = tag_data(train, test)


    ########## build DOc2Vec models ##########

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


    ########## SVC ##########

    svc_0, svc_1 = train_SVC(train_x_0, train_y_0, train_x_1, train_y_1)

    accuracy_model_0 = acc(test_y_0, svc_0.predict(test_x_0))
    accuracy_model_1 = acc(test_y_1, svc_1.predict(test_x_1))

    print("SVC accuracy model 0: ", accuracy_model_0)
    print("SVC accuracy model 1: ", accuracy_model_1)


    ########## Naive Bayes ##########

    bayes_0, bayes_1 = train_NB(train_x_0, train_y_0, train_x_1, train_y_1)

    accuracy_model_0 = acc(test_y_0, bayes_0.predict(test_x_0))
    accuracy_model_1 = acc(test_y_1, bayes_1.predict(test_x_1))

    print("NB accuracy model 0: ", accuracy_model_0)
    print("NB accuracy model 1: ", accuracy_model_1)


    ########## Random Forest ##########

    forest_0, forest_1 = train_RF(train_x_0, train_y_0, train_x_1, train_y_1)

    accuracy_model_0 = acc(test_y_0, forest_0.predict(test_x_0))
    accuracy_model_1 = acc(test_y_1, forest_1.predict(test_x_1))

    print("RF accuracy model 0: ", accuracy_model_0)
    print("RF accuracy model 1: ", accuracy_model_1)


    ########## DL model Sequential ##########

    train_x_0, train_y_0, test_x_0, test_y_0 = prepare_data_keras(train_x_0, train_y_0, test_x_0, test_y_0)

    deep_models = [Sequential(),Sequential()]

    for model in deep_models:
        model.add(Dense(512, activation='relu', input_shape=(300,)))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(3,activation='softmax'))
        model.compile(loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001),
            metrics=['acc',recall_m,precision_m,f1_m])

    # fit with 90 epochs
    history_0 = deep_models[0].fit(train_x_0,train_y_0,epochs=90,validation_data=(test_x_0,test_y_0), verbose=1)
    history_1 = deep_models[1].fit(train_x_0,train_y_0,epochs=90,validation_data=(test_x_0,test_y_0), verbose=0)
    
    # evaluate the models
    for model in deep_models:
        model.evaluate(test_x_0, test_y_0, batch_size=128)

        
        
    #######################################################################################################
    
    data['content2'] = data['content']
    
    for i in range(len(data['content2'])):
        data['content2'].iloc[i] = remove_stopwords(data['content2'].iloc[i])
        if len(data['content2'].iloc[i]) >= 100:
            data['content2'].iloc[i] = ' '.join((data['content2'].iloc[i]).split()[:100])
            

    
    
    vectorized = vectorize_texts(data['content2'].to_list())
    
    clusters = 10
    kmeans = cluster_texts(clusters, vectorized)
    kmeansdf = pd.DataFrame()
    import pdb
    pdb.set_trace()
    kmeansdf['cluster'] = kmeans.labels_
    kmeansdf['stemmed'] = data['content2']
    
    import seaborn as sns
    # ax = sns.countplot(x= 'kmeans10', data=kmeansdf)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    
    dic = get_most_common_words(kmeansdf, 25, clusters)
    for i in range(clusters):
        print(dic[i])
    # import pdb
    # pdb.set_trace()