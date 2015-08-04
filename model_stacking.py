# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn import cross_validation
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

"""Kappa estimator
The following 3 functions are used to calculate Kappa estimator.
"""
def getFreq(a):
    ret = np.zeros(4)
    for x in a:
        ret[x-1] += 1
    return np.array(ret) / len(a)

def getFreq2(a, b):
    ret = np.zeros((4, 4))
    
    N = len(a)
    for i in range(N):
        at = a[i]-1
        bt = b[i]-1
        ret[at][bt] += 1
        
    return np.array(ret) / N
        
def getKappa(actual, predict):

    if list(actual) == list(predict):
        return 1.0

    N = 4

    A = getFreq(actual)
    B = getFreq(predict)
    O = getFreq2(actual, predict)
    
    A = A.reshape(4, 1)
    B = B.reshape(1, 4)
    E = np.dot(A, B)
    s = sum(sum(E))
    E /= s
    
    num = 0.0
    den = 0.0
        
    for i in range(4):
        for j in range(4):
            w = pow(i-j, 2.0) / pow(N-1, 2.0)
            num += w * O[i][j]
            den += w * E[i][j]
                        
    ret = 1.0 - num / den
        
    return ret

"""ranking function
The following 2 functions are used to calculate ranking(discrete value) from
continuous value obtained by regression models.
"""
def getRank(f, weight):
    ret = 0
    for w in weight:
        if f <= w:
            break
        ret += 1
    return ret

def convertScore(y, weight):
    cnt = 0.0
    n = len(y)
    
    tmp = np.argsort(y)
    for i in tmp:
        cnt += 1.0
        y[i] = getRank(cnt/n, weight)
    
    return y.astype(int)

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
        
    ##############################################################
    # stacking result = svr + α * rfr + β * gbr
    # tune α and β with cross validation
    ##############################################################
    scores = dict()
    skf = cross_validation.StratifiedKFold(Y, n_folds=3)
    for train_index, test_index in skf:
        X1, X2 = X[train_index], X[test_index]
        Y1, Y2 = Y[train_index], Y[test_index]
        
        # predict with SVR
        svr = SVR()
        svr.set_params(**pickle.load(open("svr.p", "rb" )))
        svr.fit(X1, Y1)
        Y_svr = svr.predict(X2)

        # predict with RF
        rfr = RandomForestRegressor(n_estimators = 1000)
        rfr.set_params(**pickle.load(open("rfr.p", "rb" )))
        rfr.fit(X1, Y1)
        Y_rfr = rfr.predict(X2)
    
        # predict with GBT
        gbr = GradientBoostingRegressor(n_estimators=3000)
        gbr.set_params(**pickle.load(open("gbr.p", "rb" )))
        gbr.fit(X1, Y1)
        Y_gbr = gbr.predict(X2)
        
        # stacking
        for alpha in np.logspace(-10, 10, 21, base=2):
            for beta in np.logspace(-10, 10, 21, base=2):
                y_pred = Y_svr + alpha * Y_rfr + beta * Y_gbr
                y_rank = convertScore(y_pred, 
                           [0.0, 0.0761961015948, 0.221500295334, 0.392498523331, 1.0])
                
                if (alpha, beta) not in scores:
                    scores[(alpha, beta)] = 0.0
                scores[(alpha, beta)] += getKappa(Y2, y_rank)
    
    ##############################################################
    # predict ranking with optimal alpha and beta
    ##############################################################
    alpha, beta = max(scores, key=scores.get)
    print "best Kappa score = %f, when alpha=%f, beta=%f" % (scores[(alpha, beta)]/3, alpha, beta) 
    
    # predict with SVR
    svr = SVR()
    svr.set_params(**pickle.load(open("svr.p", "rb" )))
    svr.fit(X, Y)
    Y_svr = svr.predict(Xt)

    # predict with RF
    rfr = RandomForestRegressor(n_estimators = 1000)
    rfr.set_params(**pickle.load(open("rfr.p", "rb" )))
    rfr.fit(X, Y)
    Y_rfr = rfr.predict(Xt)

    # predict with GBT
    gbr = GradientBoostingRegressor(n_estimators=3000)
    gbr.set_params(**pickle.load(open("gbr.p", "rb" )))
    gbr.fit(X, Y)
    Y_gbr = gbr.predict(Xt)
    
    # stacking three models
    y_pred = Y_svr + alpha * Y_rfr + beta * Y_gbr
    y_rank = convertScore(y_pred, [0.0, 0.0761961015948, 0.221500295334, 0.392498523331, 1.0])
    
    # write the result to CSV
    submission = pd.DataFrame({"id": ids, "prediction": y_rank})
    submission.to_csv("submission.csv", index=False)
    