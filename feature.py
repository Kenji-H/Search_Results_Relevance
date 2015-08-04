# -*- coding: utf-8 -*-

import re
import pandas as pd
import distance
import zlib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from scipy.sparse import vstack

def word_len(text):
    words = re.split("\W+", str(text))
    words = [w for w in words if len(w) > 0]
    return len(words)

"""Normalized compression distance

See the page below for more details:
https://en.wikipedia.org/wiki/Normalized_compression_distance#Normalized_compression_distance

"""
def ncd(x,y):
    if x == y:
        return 0
    z_x = len(zlib.compress(x))
    z_y = len(zlib.compress(y))
    z_xy = len(zlib.compress(x + y))    
    return float(z_xy - min(z_x, z_y)) / max(z_x, z_y)

def get_uniq_words_text(text):
    words = text.split(" ")
    words = list(set(words))
    return " ".join(words)

def get_features(raw_data):
    fet_data = pd.DataFrame()
    
    print "extracting count features..."
    fet_data['q_len'] = raw_data["query"].map(word_len)
    fet_data['t_len'] = raw_data["product_title"].map(word_len)
    fet_data['d_len'] = raw_data["product_description"].map(word_len)
    
    print "extracting basic distance features from q and t..."
    fet_data['nleven1'] = raw_data.apply(lambda x : distance.nlevenshtein(x.q, x.t, method=1), axis=1) 
    fet_data['nleven2'] = raw_data.apply(lambda x : distance.nlevenshtein(x.q, x.t, method=2), axis=1) 
    fet_data['sorensen'] = raw_data.apply(lambda x : distance.sorensen(x.q, x.t), axis=1) 
    fet_data['jaccard'] = raw_data.apply(lambda x : distance.jaccard(x.q, x.t), axis=1) 
    fet_data['ncd'] = raw_data.apply(lambda x : ncd(x.q, x.t), axis=1) 

    print "extracting basic distance features from q_ex and t..."
    fet_data['sorensen_ex'] = raw_data.apply(lambda x : distance.sorensen(get_uniq_words_text(x.q_ex), x.t), axis=1) 
    print "extracting basic distance features from q_ex and t..."
    fet_data['jaccard_ex'] = raw_data.apply(lambda x : distance.jaccard(get_uniq_words_text(x.q_ex), x.t), axis=1) 
    print "extracting basic distance features from q_ex and t..."
    fet_data['ncd_ex'] = raw_data.apply(lambda x : ncd(get_uniq_words_text(x.q_ex), x.t), axis=1) 

    return fet_data

if __name__ == '__main__':
        
    # read data from CSV
    train = pd.read_csv('train_preprocessed.csv')
    test = pd.read_csv('test_preprocessed.csv')

    # extract basic features
    train_f = get_features(train)
    test_f = get_features(test)
    
    # extract TF-IDF based features
    text = list(train['q'])
    text.extend(train['t'])
    text.extend(test['q'])
    text.extend(test['t'])
     
    for ngram in [1,2,3]:
        for svd_components in [100, 250, 500, 1000]:
            print "processing ngram=%d, svd_components=%d..." % (ngram, svd_components)
             
            tfv = TfidfVectorizer(min_df=3, max_df=0.9, strip_accents='unicode', 
                                  token_pattern=r'\w{1,}', ngram_range=(1, ngram))
            tfv.fit(text)
             
            Q = tfv.transform(list(train['q']))
            R = tfv.transform(list(train['q_ex']))
            X = tfv.transform(list(train['t']))
            Qt = tfv.transform(list(test['q']))
            Rt = tfv.transform(list(test['q_ex']))
            Xt = tfv.transform(list(test['t']))
  
            svd = TruncatedSVD(n_components = svd_components)
            svd.fit(vstack([Q,X,Qt,Xt]))
            normalizer = Normalizer()
             
            Q = svd.transform(Q)
            Q = normalizer.transform(Q)
            R = svd.transform(R)
            R = normalizer.transform(R)
            X = svd.transform(X)
            X = normalizer.transform(X)
             
            Qt = svd.transform(Qt)
            Qt = normalizer.transform(Qt)
            Rt = svd.transform(Rt)
            Rt = normalizer.transform(Rt)
            Xt = svd.transform(Xt)
            Xt = normalizer.transform(Xt)
             
            colname = "cos_dist_%d_%d" % (ngram, svd_components)
            D = [np.dot(a, b) for (a, b) in zip(Q, X)]
            Dt = [np.dot(a, b) for (a, b) in zip(Qt, Xt)]             
            train_f[colname] = D
            test_f[colname] = Dt
    
            colname = "cos_dist_%d_%d_ex" % (ngram, svd_components)
            D_ex = [np.dot(a, b) for (a, b) in zip(R, X)]
            D_ext = [np.dot(a, b) for (a, b) in zip(Rt, Xt)]
            train_f[colname] = D_ex
            test_f[colname] = D_ext
    
    # query-wise features
    qmap = {}
    cnt = 0
    for i in range(len(train.index)):
        q = train.loc[i]['query']
        if q not in qmap:
            qmap[q] = (cnt, [])
            cnt += 1
        qmap[q][1].append(train.loc[i]['median_relevance'])
        
    train_f['qid'] = train["query"].map(lambda x : qmap[x][0])
    train_f['max_relevance'] = train["query"].map(lambda x : np.max(qmap[x][1]))
    train_f['min_relevance'] = train["query"].map(lambda x : np.min(qmap[x][1]))
    train_f['mean_relevance'] = train["query"].map(lambda x : np.mean(qmap[x][1]))

    test_f['qid'] = test["query"].map(lambda x : qmap[x][0])
    test_f['max_relevance'] = test["query"].map(lambda x : np.max(qmap[x][1]))
    test_f['min_relevance'] = test["query"].map(lambda x : np.min(qmap[x][1]))
    test_f['mean_relevance'] = test["query"].map(lambda x : np.mean(qmap[x][1]))

    # write to CSV  
    train_f['label'] = train['median_relevance']
    test_f['id'] = test['id']
    train_f.to_csv("train_f.csv", index=False)
    test_f.to_csv("test_f.csv", index=False)
