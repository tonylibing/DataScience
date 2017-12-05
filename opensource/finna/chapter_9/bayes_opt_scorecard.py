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
from sklearn.model_selection  import cross_val_score
from sklearn.model_selection  import StratifiedKFold
from sklearn.metrics import *
from sklearn.model_selection  import GridSearchCV
from bayes_opt import BayesianOptimization
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek,SMOTEENN
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

trainData=pd.read_csv(path+'bank_default/trainData.csv',encoding='gbk')

var_WOE_list = pickle.load(open(path+'bank_default/var_WOE_list.pkl','r'))

X = trainData[var_WOE_list]   #by default  LogisticRegressionCV() fill fit the intercept
X = np.matrix(X)
y = trainData['target']
y = np.array(y)

oversampler=SMOTEENN(random_state=2017)
X_os,y_os=oversampler.fit_sample(X,y)
print X_os.shape, y_os.shape


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
X_train.shape, y_train.shape


negative,positive = trainData.groupby('target').count()['Idx']
scale_pos_weight = negative*1.0/positive

# cv_result = xgb.cv(params, xgtrain, num_boost_round=num_rounds, nfold=5,
#                    seed=random_state,
#                    callbacks=[xgb.callback.early_stop(50)])
#
# return -cv_result['test-mae-mean'].values[-1]

def xgbcv(max_depth,n_estimators,learning_rate,subsample,colsample_bytree,min_child_weight,gamma):
    gbm = xgb.XGBClassifier(max_depth=int(max_depth), n_estimators=int(n_estimators), learning_rate=learning_rate,
                            subsample=subsample, colsample_bytree=colsample_bytree,
                            min_child_weight = min_child_weight, gamma = gamma,
                            objective="binary:logistic", seed=999,nthread=5,scale_pos_weight=scale_pos_weight)
    gbm.fit(X,y)
    pred_y = gbm.predict(X_test)
    pred_score_y = gbm.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred_score_y)
    scorecard_result = pd.DataFrame({'prob': pred_y, 'target': y_test})
    performance = KS_AR(scorecard_result, 'prob', 'target')
    KS = performance['KS']
    with open('xgb_bayes_opt_results.txt','a') as f: f.write("max_depth:%f,n_estimators:%f,learning_rate:%f,subsample:%f,colsample_bytree:%f,min_child_weight:%f,gamma:%f,auc:%f,KS:%f\n"%(max_depth,n_estimators,learning_rate,subsample,colsample_bytree,min_child_weight,gamma,auc,KS))
    return auc

def xgbfcv(max_depth,n_estimators,learning_rate,subsample,colsample_bytree,min_child_weight,gamma):
    gbm = xgb.XGBClassifier(max_depth=int(max_depth), n_estimators=int(n_estimators), learning_rate=learning_rate,
                            subsample=subsample, colsample_bytree=colsample_bytree,
                            min_child_weight = min_child_weight, gamma = gamma,
                            objective="binary:logistic", seed=999,nthread=5,scale_pos_weight=scale_pos_weight)
    scores = cross_val_score(gbm,X,y,scoring='roc_auc',cv=StratifiedKFold(5,shuffle=True,random_state=2017))
    #print scores
    return scores.mean()

#oversampling
def os_xgbfcv(max_depth,n_estimators,learning_rate,subsample,colsample_bytree,min_child_weight,gamma):
    gbm = xgb.XGBClassifier(max_depth=int(max_depth), n_estimators=int(n_estimators), learning_rate=learning_rate,
                            subsample=subsample, colsample_bytree=colsample_bytree,
                            min_child_weight = min_child_weight, gamma = gamma,
                            objective="binary:logistic", seed=999,nthread=5)
    scores = cross_val_score(gbm, X_os, y_os, scoring='roc_auc', cv=StratifiedKFold(5, shuffle=True, random_state=2017))
    #print scores
    return scores.mean()


xgbBO = BayesianOptimization(os_xgbfcv,
    {
    'max_depth':(int(10),int(12)),
    'n_estimators':(int(20),int(100)),
    'learning_rate':(0.05,0.1),
    'subsample':(0.3,0.5),
    'colsample_bytree':(0.6,0.8),
    'min_child_weight':(1,40),
    'gamma':(0.05,1)
    })
num_iter = 100
init_points = 5

xgbBO.maximize(init_points=init_points, n_iter=num_iter)
print('-'*53)
print('Final Result')
print('xgboost:%f' % xgbBO.res['max']['max_val'])
print('xgboost:%s' % xgbBO.res['max']['max_params'])

# StratifiedKFold(5,shuffle=True) cv
# Final Result
# xgboost:0.681187
# xgboost:{'colsample_bytree': 0.60410840574194091, 'learning_rate': 0.053154289795393717, 'min_child_weight': 27.171585489741449, 'n_estimators': 49.852454110263047, 'subsample': 0.44408010700744305, 'max_depth': 10.239885116586866, 'gamma': 0.96567380833757099}

# StratifiedKFold(3,shuffle=False)
# Final Result
# xgboost:0.939819
# xgboost:{'colsample_bytree': 0.77700860883714562, 'learning_rate': 0.098796944573766435, 'min_child_weight': 1.0, 'n_estimators': 90.684655134199701, 'subsample': 0.47733345993884213, 'max_depth': 12.0, 'gamma': 0.050000000000000003}
#

# train test split
# Final Result
# xgboost:0.902749
# xgboost:{'colsample_bytree': 0.66272561328627511, 'learning_rate': 0.09156099963472944, 'min_child_weight': 1.0482321352908903, 'n_estimators': 99.962069818954291, 'subsample': 0.36195063022161922, 'max_depth': 11.699764521378645, 'gamma': 0.067422526986835915}
