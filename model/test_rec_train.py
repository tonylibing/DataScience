import pandas as pd
import numpy as np
import gc
import sys
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
import processor
from importlib import reload
reload(processor)
from processor import *
from imblearn.ensemble import EasyEnsemble
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

data=pd.read_csv("~/dataset/rec_data_train_3w.csv",sep=',')
#data=pd.read_csv("/media/sf_D_DRIVE/download/rec_data_train_save.csv",sep=',')
print(data.columns.values)
y=data['invest']
data.drop(['rd','click','invest','invest_amount','mobile_no_attribution'],axis=1,inplace=True)
#X=data[[col for col in data.columns if col not in ['invest','invest_amount']]]
X=data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999,stratify=y)

n_subsets = int(sum(y==0)/sum(y==1))
# ee = EasyEnsemble(n_subsets=n_subsets)
# sample_X, sample_y = ee.fit_sample(X, y)

# rus = RandomUnderSampler(random_state=42)
# X_res, y_res = rus.fit_sample(X, y)

from sklearn.model_selection import train_test_split
X_n = X[y==0]
y_n = y[y==0]
X_y = X[y==1]
y_y = y[y==1]
X_n_drop,X_n_retain,y_n_drop,y_n_retain = train_test_split(X_n,y_n,test_size= 1.0/n_subsets, random_state=0, stratify=X_n[['cust_level']])
X_new = pd.concat([X_n_retain,X_y],axis=0)
y_new = pd.concat([y_n_retain,y_y],axis=0)

#no weight
X=X_new
bfp = FeatureProcessor(X)
feature_matrix = bfp.fit_transform(X)

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=50, learning_rate=0.3,
                              subsample=0.8, colsample_bytree=0.7,
                              objective="binary:logistic", seed=999)

X_train, X_test, y_train, y_test = train_test_split(feature_matrix, y_new, test_size=0.2, random_state=999,stratify=y_new)

lr = LogisticRegression(C=1.0, penalty='l2', tol=1e-4,solver='liblinear',random_state=42)
lr.fit(X_train,y_train)
y_pre= lr.predict(X_test)
y_pro= lr.predict_proba(X_test)[:,1]
print("="*60)
print("LR Test AUC Score : {0}".format(metrics.roc_auc_score(y_test, y_pro)))
print("LR  Test Accuracy : {0}".format(metrics.accuracy_score(y_test, y_pre)))
print("="*60)

gbm = xgb.XGBClassifier(n_estimators=30,learning_rate =0.3,max_depth=3,min_child_weight=1,gamma=0.3,subsample=0.7,colsample_bytree=0.7,objective= 'binary:logistic',nthread=-1,scale_pos_weight=1,reg_alpha=1e-05,reg_lambda=1,seed=27)
gbm.fit(X_train,y_train)
y_pre= gbm.predict(X_test)
# y_pre_leaf = gbm.predict(X_test,pred_leaf=True)
# print(y_pre_leaf.shape)
y_pro= gbm.predict_proba(X_test)[:,1]
print("="*60)
print("Xgboost model Test AUC Score: {0}".format(metrics.roc_auc_score(y_test, y_pro)))
print("Xgboost model Test Accuracy : {0}".format(metrics.accuracy_score(y_test, y_pre)))
print("="*60)

lgbm = lgb.LGBMClassifier(boosting_type='gbdt',  max_depth=3, learning_rate=0.3, n_estimators=30, min_child_weight=1,subsample=0.7,  colsample_bytree=0.7, reg_alpha=1e-05, reg_lambda=1)
lgbm.fit(X_train,y_train)
y_pre= lgbm.predict(X_test)
y_pro= lgbm.predict_proba(X_test)[:,1]
print("="*60)
print("lightgbm model Test AUC Score: {0}".format(metrics.roc_auc_score(y_test, y_pro)))
print("lightgbm model Test Accuracy : {0}".format(metrics.accuracy_score(y_test, y_pre)))
print("="*60)

gbdtlr = XgboostLRClassifier()
gbdtlr.fit(X_train,y_train)
y_pre= gbdtlr.predict(X_test)
y_pro= gbdtlr.predict_proba(X_test)[:,1]
print("="*60)
print("Xgboost+LR Test AUC Score : {0}".format(metrics.roc_auc_score(y_test, y_pro)))
print("Xgboost+LR  Test Accuracy : {0}".format(metrics.accuracy_score(y_test, y_pre)))
print("="*60)

lgbmlr = LightgbmLRClassifier()
lgbmlr.fit(X_train,y_train)
y_pre= lgbmlr.predict(X_test)
y_pro= lgbmlr.predict_proba(X_test)[:,1]
print("="*60)
print("Lightgbm+LR Test AUC Score : {0}".format(metrics.roc_auc_score(y_test, y_pro)))
print("Lightgbm+LR  Test Accuracy : {0}".format(metrics.accuracy_score(y_test, y_pre)))
print("="*60)

gbdtlr = XgboostLRClassifier(combine_feature=False)
gbdtlr.fit(X_train,y_train)
y_pre= gbdtlr.predict(X_test)
y_pro= gbdtlr.predict_proba(X_test)[:,1]
print("="*60)
print("Xgboost+LR Test AUC Score : {0}".format(metrics.roc_auc_score(y_test, y_pro)))
print("Xgboost+LR  Test Accuracy : {0}".format(metrics.accuracy_score(y_test, y_pre)))
print("="*60)

lgbmlr = LightgbmLRClassifier(combine_feature=False)
lgbmlr.fit(X_train,y_train)
y_pre= lgbmlr.predict(X_test)
y_pro= lgbmlr.predict_proba(X_test)[:,1]
print("="*60)
print("Lightgbm+LR Test AUC Score : {0}".format(metrics.roc_auc_score(y_test, y_pro)))
print("Lightgbm+LR  Test Accuracy : {0}".format(metrics.accuracy_score(y_test, y_pre)))
print("="*60)