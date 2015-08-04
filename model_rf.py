# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
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
    
    rfr = RandomForestRegressor(n_estimators = 1000)
    parameters = {"max_features": [0.3, 0.5, 0.7, 1.0], 
                  "min_samples_split": [2, 3, 5, 7, 10], 
                  "max_depth": [None, 1, 3, 5, 10]}
    
    gscv = RandomizedSearchCV(rfr, parameters, n_iter=30, verbose=10, n_jobs=-1, cv=3, scoring='mean_squared_error')
    gscv.fit(X, Y)
            
    print "best score=", gscv.best_score_
    print "best_parameter=", gscv.best_params_
    pickle.dump(gscv.best_params_, open("rfr.p", "wb"))
    