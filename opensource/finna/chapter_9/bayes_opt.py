# -*- coding: utf-8 -*-
import pandas as pd
import datetime
import collections
import numpy as np
import numbers
import random
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection  import train_test_split
from sklearn.metrics import *
from sklearn.model_selection  import GridSearchCV
from bayes_opt import BayesianOptimization
import statsmodels.api as sm
import xgboost as xgb
import sys
import pickle
reload(sys)
sys.setdefaultencoding( "utf-8")
import scorecard_fucntions
reload(scorecard_fucntions)
from scorecard_fucntions import *
from sklearn.linear_model import LogisticRegressionCV

path= '/home/tanglek/workspace/DataScience/data/ppd/'

trainData=pd.read_csv(path+'bank_default/trainData.csv',index=False,encoding='gbk')

var_WOE_list = pickle.load(open(path+'bank_default/var_WOE_list.pkl','r'))

X = trainData[var_WOE_list]   #by default  LogisticRegressionCV() fill fit the intercept
X = np.matrix(X)
y = trainData['target']
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
X_train.shape, y_train.shape


negative,positive = trainData.groupby('target').count()['Idx']
scale_pos_weight = negative*1.0/positive

def xgbcv(max_depth,n_estimators,learning_rate,subsample,colsample_bytree,min_child_weight,gamma):
    gbm = xgb.XGBClassifier(max_depth=int(max_depth), n_estimators=int(n_estimators), learning_rate=learning_rate,
                            subsample=subsample, colsample_bytree=colsample_bytree,
                            min_child_weight = min_child_weight, gamma = gamma,
                            objective="binary:logistic", seed=999,nthread=5,scale_pos_weight=scale_pos_weight)
    gbm.fit(X,y)
    pred_y = gbm.predict(X_test)
    pred_score_y = gbm.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred_score_y)
    with open('xgb_bayes_opt_results.txt','a') as f: f.write("max_depth:%f,n_estimators:%f,learning_rate:%f,subsample:%f,colsample_bytree:%f,min_child_weight:%f,gamma:%f,auc:%f\n"%(max_depth,n_estimators,learning_rate,subsample,colsample_bytree,min_child_weight,gamma,auc))
    return auc

xgbBO = BayesianOptimization(xgbcv,
    {
    'max_depth':(int(10),int(12)),
    'n_estimators':(int(20),int(100)),
    'learning_rate':(0.05,0.1),
    'subsample':(0.3,0.5),
    'colsample_bytree':(0.6,0.8),
    'min_child_weight':(1,40),
    'gamma':(0.05,1)
    })

xgbBO.maximize(niter=20)
print('-'*53)
print('Final Result')
print('xgboost:%f' % xgbBO.res['max']['max_val'])
print('xgboost:%s' % xgbBO.res['max']['max_params'])
