import pandas as pd
import numpy as np
import json
from dateutil.parser import parse
from pandas.io.json import json_normalize

#data=pd.read_csv("E:/dataset/rec_data.csv",sep=',')
data=pd.read_csv("~/dataset/rec_data.csv",sep=',')
cat_df = json_normalize(data['cat_pref'].apply(lambda x:json.loads(x)).tolist())
cat_df.fillna(0,inplace=True)

amt_df = json_normalize(data['amount_pref'].apply(lambda x:json.loads(x)).tolist())
amt_df.fillna(0,inplace=True)

period_df = json_normalize(data['period_pref'].apply(lambda x:json.loads(x)).tolist())
period_df.fillna(0,inplace=True)


data['dayOfWeek'] = data['request_time'].apply(lambda x: parse(x).weekday())
data['hourOfDay'] = data['request_time'].apply(lambda x: parse(x).hour)
data['dayOfMonth'] = data['request_time'].apply(lambda x: parse(x).day)

pref = pd.concat([cat_df,amt_df,period_df],axis=1)

columns = ['rd','app_version','hourOfDay','dayOfWeek','dayOfMonth','address','rec','user_id','user_group','click','invest','invest_amount','cust_level','gender','age','nationality','total_balance','curr_aum_amt','highest_asset_amt','risk_verify_status_cn','fst_invest_days','invest_period_by_days','product_price','product_category','risk_level','item','transfer_flag']

ff = pd.concat([data[columns],pref],axis=1)
fo= ff.loc[~(ff['invest']==1 & ff['invest_amount'].isnull())]
fo.to_csv("E:/dataset/rec_data_train_save.csv",index=False,header=True)


import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
import lightgbm as lgb
from feature.processor import *
data=pd.read_csv("~/dataset/rec_data_train_save.csv",sep=',')
sp_wts =(1 - data.groupby('invest').size()/data.shape[0]).reset_index()
sp_wts.columns=['invest','sample_weight']
data2=pd.merge(data,sp_wts,how='left',on='invest')
data=data2
del data2
gc.collect()
print(data.columns.values)
y=data['invest']
data.drop(['invest','invest_amount'],axis=1,inplace=True)
#X=data[[col for col in data.columns if col not in ['invest','invest_amount']]]
X=data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999,stratify=y)

bfp = FeatureProcessor(X)
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=50, learning_rate=0.08,
                              subsample=0.8, colsample_bytree=0.7,
                              objective="binary:logistic", seed=999)

gbm.fit(X_train[[col for col in X_train.columns if col not in ['sample_weight'] ]],y_train,sample_weight=X_train['sample_weight'])

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
X_new
bfp = FeatureProcessor(X_new)
feature_matrix = bfp.fit_transform(X_new)
X_tain = xgb.DMatrix(feature_matrix, label=y_new)



gbm = xgb.XGBClassifier(max_depth=3, n_estimators=50, learning_rate=0.08,
                              subsample=0.8, colsample_bytree=0.7,
                              objective="binary:logistic", seed=999)

gbm.fit(feature_matrix,y_new)
#gbm.fit(X_train,y_train)
#test gbdt lr 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.datasets  import  make_hastie_10_2
from GBDTLRClassifier import XgboostLRClassifier
from GBDTLRClassifier import LightgbmLRClassifier

X, y = make_hastie_10_2(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999)


lr = LogisticRegression(C=1.0, penalty='l1',tol=1e-4, solver='liblinear', random_state=42)
lr.fit(X_train,y_train)
y_pre= lr.predict(X_test)
y_pro= lr.predict_proba(X_test)[:,1]
print("LR Test AUC Score : {0}", metrics.roc_auc_score(y_test, y_pro))
print("LR  Test Accuracy : {0}" , metrics.accuracy_score(y_test, y_pre))

gbdtlr = XgboostLRClassifier()
gbdtlr.fit(X_train,y_train)
y_pre= gbdtlr.predict(X_test)
y_pro= gbdtlr.predict_proba(X_test)[:,1]
print("GBDT+LR Test AUC Score : {0}", metrics.roc_auc_score(y_test, y_pro))
print("GBDT+LR  Test Accuracy : {0}" , metrics.accuracy_score(y_test, y_pre))

gbm = xgb.XGBClassifier(n_estimators=30,learning_rate =0.3,max_depth=3,min_child_weight=1,gamma=0.3,subsample=0.7,colsample_bytree=0.7,objective= 'binary:logistic',nthread=-1,scale_pos_weight=1,reg_alpha=1e-05,reg_lambda=1,seed=27)
gbm.fit(X_train,y_train)
y_pre= gbm.predict(X_test)
y_pred = gbm.apply(X_test)

transformed_training_matrix = np.zeros([len(y_pred),len(y_pred[0]) * num_leaf],dtype=np.int64)
for i in range(0,len(y_pred)):
	temp = np.arange(len(y_pred[0])) * num_leaf - 1 + np.array(y_pred[i])
	transformed_training_matrix[i][temp] += 1
    
    
y_pro= gbm.predict_proba(X_test)[:,1]
print("Xgboost model Test AUC Score : {0}", metrics.roc_auc_score(y_test, y_pro))
print("Xgboost model Test Accuracy : {0}" , metrics.accuracy_score(y_test, y_pre))

lgbm = lgb.LGBMClassifier(boosting_type='gbdt',  max_depth=3, learning_rate=0.3, n_estimators=30, min_child_weight=1,subsample=0.7,  colsample_bytree=0.7, reg_alpha=1e-05, reg_lambda=1)
lgbm.fit(X_train,y_train)
y_pre= lgbm.predict(X_test)
y_pro= lgbm.predict_proba(X_test)[:,1]
print("lightgbm model Test AUC Score : {0}", metrics.roc_auc_score(y_test, y_pro))
print("lightgbm model Test Accuracy : {0}" , metrics.accuracy_score(y_test, y_pre))