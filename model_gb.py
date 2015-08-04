# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
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
    
    gbr = GradientBoostingRegressor(n_estimators = 1000)
    parameters = {'learning_rate' : [0.1, 0.05, 0.02, 0.01], 
                  'max_depth': [2, 4, 6],
                  'min_samples_leaf': [1, 3, 5, 10, 15],
                  'subsample': [1.0, 0.8, 0.6, 0.4], 
                  'max_features': [1.0, 0.7, 0.5, 0.3, 0.1]}    
    gscv = RandomizedSearchCV(gbr, parameters, n_iter=100, verbose=10, n_jobs=-1, cv=3, scoring='mean_squared_error')
    gscv.fit(X, Y)
    
    # tune learning rate with larger n_estimators
    gbr = gscv.best_estimator_
    gbr.set_params(n_estimators=3000)
    parameters = {'learning_rate' : [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]}
    gscv = GridSearchCV(gbr, parameters, verbose=10, n_jobs=-1, cv=3, scoring='mean_squared_error')
    gscv.fit(X, Y)

    print "best score=", gscv.best_score_
    print "best_parameter=", gscv.best_params_
    pickle.dump(gscv.best_estimator_.get_params(), open("gbr.p", "wb"))
    