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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2017,stratify=y)
X_train.shape, y_train.shape

oversampler=SMOTE(random_state=2017)
X_train_os,y_train_os=oversampler.fit_sample(X_train,y_train)
print X_train_os.shape, y_train_os.shape
# (33362, 14) (33362,)

negative,positive = trainData.groupby('target').count()['Idx']
scale_pos_weight = negative*1.0/positive

gbm = xgb.XGBClassifier(max_depth=12, n_estimators=30, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.7, max_delta_step=3,
                              objective="binary:logistic", seed=999,scale_pos_weight=scale_pos_weight)

gbm.fit(X_train_os,y_train_os)

pred_y = gbm.predict(X_test)
pred_score_y = gbm.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, pred_score_y)
accuracy = accuracy_score(y_test, pred_y)
cm = confusion_matrix(y_test, pred_y)
print 'gbm training:', auc, accuracy, cm

#
# gbm training: 0.633175313561 0.617
# [[6902 4219]
#  [ 377  502]]
