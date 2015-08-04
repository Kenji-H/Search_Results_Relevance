# -*- coding: utf-8 -*-

import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

STOPS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

def preprocess(text):
        
    # create words list
    words = re.split("\W+", text)

    # remove empty word and lower all letters
    words = [w.lower() for w in words if len(w) > 0]
    
    # remove stop words
    words = [w for w in words if w not in STOPS]
    
    # stemming
    words = [STEMMER.stem(w) for w in words]
    
    return str(" ".join(words))

if __name__ == '__main__':
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # text pre-processing
    train["q"] = train["query"].map(preprocess)
    train["t"] = train["product_title"].map(preprocess)
    
    test["q"] = test["query"].map(preprocess)
    test["t"] = test["product_title"].map(preprocess)
    
    # query extension
    ex_query = {}
    for i in range(len(train.index)):
        q = train.loc[i]["query"]
        if q not in ex_query:
            ex_query[q] = train.loc[i]['q']
        if train.loc[i]["median_relevance"] == 4.0 and train.loc[i]["relevance_variance"] == 0.0:
            if len(ex_query[q]) < 8 * len(train.loc[i]["t"]):
                ex_query[q] += train.loc[i]["t"]

#     len1 = 0
#     len2 = 0
#     for k, v in ex_query.iteritems():
#         len1 += len(k)
#         len2 += len(v)
#      
#     print float(len2) / len1
        
    train["q_ex"] = train["query"].map(lambda x : ex_query[x])
    test["q_ex"] = test["query"].map(lambda x : ex_query[x])
    
    train.to_csv("train_preprocessed.csv", index=False)
    test.to_csv("test_preprocessed.csv", index=False)
