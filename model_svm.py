# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.grid_search import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pickle

if __name__ == '__main__':
    
    np.random.seed(474747)
    
    train = pd.read_csv("train_f.csv")
    test = pd.read_csv("test_f.csv")
    
    Y = train["label"].values
    train = train.drop('label', axis=1)
    
    ids = test["id"]
    test = test.drop('id', axis=1)
    
    X = train.values
    Xt = test.values    
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Xt = scaler.transform(Xt)
    
    svr = SVR()
    parameters = {"C" : np.logspace(-4, 4, 17), 
                  "gamma": np.logspace(-4, 4, 17), 
                  "epsilon": [0.001, 0.01, 0.1, 1.0]}
    
    gscv = RandomizedSearchCV(svr, parameters, n_iter=100, verbose=10, n_jobs=-1, cv=3, scoring='mean_squared_error')
    gscv.fit(X, Y)
            
    print "best score=", gscv.best_score_
    print "best_parameter=", gscv.best_params_
    pickle.dump(gscv.best_params_, open("svr.p", "wb"))
    