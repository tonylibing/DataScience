import pandas as pd
import numpy as np
from dateutil.parser import parse
from pandas.io.json import json_normalize

data=pd.read_csv("E:/dataset/rec_data_save.csv",sep=',')
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

columns = ['app_version','hourOfDay','dayOfWeek','dayOfMonth','address','rec','user_id','user_group','click','invest','invest_amount','cust_level','gender','age','nationality','mobile_no_attribution','total_balance','curr_aum_amt','highest_asset_amt','risk_verify_status_cn','fst_invest_days','invest_period_by_days','product_price','product_category','risk_level','item','transfer_flag']

ff = pd.concat([data[columns],pref],axis=1)
fo= ff.loc[~(ff['invest']==1 & ff['invest_amount'].isnull())]
fo.to_csv("E:/dataset/rec_data_save.csv",index=False,header=True)


import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
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






#no weight



import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
from feature.processor import *
data=pd.read_csv("~/dataset/rec_data_train_save.csv",sep=',')
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

gbm.fit(X_train,y_train)

        