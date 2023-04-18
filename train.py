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
    
    
        
