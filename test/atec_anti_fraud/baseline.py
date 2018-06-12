import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import gc
import math
import numpy as np
from tqdm import tqdm

data_dir="/home/tanglek/dataset/atec/anti_fraud"
train = pd.read_csv(os.path.join(data_dir,"atec_anti_fraud_train.csv"))
test = pd.read_csv(os.path.join(data_dir,"atec_anti_fraud_test_a.csv"))
continous_feature = ["f"+str(i) for i in range(1,298)]

for feature in tqdm(continous_feature,desc='feature'):
    train[feature].fillna(train[feature].median(),inplace=True)
    test[feature].fillna(train[feature].median(),inplace=True)


train.loc[train['label']==-1,'label']=1
train_x = train[continous_feature]
train_y = train['label']
test_x = test[continous_feature]
res = test[['id']]
scale_pos_weight = (train_y[train_y == 0].shape[0]) * 1.0 / (train_y[train_y == 1].shape[0])
print("scale_pos_weight:",scale_pos_weight)


def LGB_predict(train_x,train_y,test_x,res):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=5000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=1,scale_pos_weight=scale_pos_weight
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc',early_stopping_rounds=100)
    res['score'] = clf.predict(test_x)
    res.to_csv('submission.csv', index=False)
    return clf

model=LGB_predict(train_x,train_y,test_x,res)

