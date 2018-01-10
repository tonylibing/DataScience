import pandas as pd
import numpy as np
import scipy
import gc
import sys
sys.path.append("../..")
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
import feature.processor
from importlib import reload

reload(feature.processor)
from feature.processor import *
from imblearn.ensemble import EasyEnsemble
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score,accuracy_score,average_precision_score
from eval.metrics import ks_statistic
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from model.GBDTLRClassifier import *

sampling_flag = True
for i in range (1,4):
    if sampling_flag:
        if os.path.exists("/home/tanglek/dataset/rec_data_train_sampled.csv"):
            print("load saved sampling training data")
            data = pd.read_csv("~/dataset/rec_data_train_sampled.csv", sep=',')
            print(data.groupby("invest").size())
            print(data.columns)
            y = data['invest']
            data['ratio'] = data['total_balance'] * 1.0 / (data['product_price'] + 1.0)
            X = data.drop(['invest','user_group','app_version','transfer_flag'], axis=1)
            upp = pd.read_csv("~/dataset/user_profile_products{0}.csv".format(i), sep=',')
            upp['ratio'] = upp['total_balance'] * 1.0 / (upp['product_price'] + 1.0)
            X_s = upp[X.columns.values]
            # print("X_s:".format(X_s.columns.values))
        else:
            data=pd.read_csv("~/dataset/rec_data.csv",sep=',')
            # data=pd.read_csv("~/dataset/rec_data_train_3w.csv",sep=',')
            # data=pd.read_csv("E:/dataset/rec_data_train_3w.csv",sep=',')
            #data=pd.read_csv("/media/sf_D_DRIVE/download/rec_data_train_save.csv",sep=',')
            print(data.columns.values)
            y=data['invest']
            # data.loc[data['age'] < 0] = np.nan
            # data.loc[data['total_balance'] < 0] = 0
            # data.loc[data['fst_invest_days'] < 0] = 0
            # data.loc[data['highest_asset_amt'] < 0] = 0
            # data['invest_period_by_days'].fillna(0,inplace=True)
            # X = data.drop(['rd','click','invest','invest_amount','mobile_no_attribution'],axis=1)
            X = data.drop(['invest','user_group','app_version','transfer_flag'],axis=1)
            #X=data[[col for col in data.columns if col not in ['invest','invest_amount']]]
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999,stratify=y)

            # n_subsets = int(sum(y==0)/sum(y==1))
            n_subsets = (int(sum(y==0)/sum(y==1)))/10
            # ee = EasyEnsemble(n_subsets=n_subsets)
            # sample_X, sample_y = ee.fit_sample(X, y)

            # rus = RandomUnderSampler(random_state=42)
            # X_res, y_res = rus.fit_sample(X, y)

            X_n = X[y==0]
            y_n = y[y==0]
            X_y = X[y==1]
            y_y = y[y==1]
            X_n_drop,X_n_retain,y_n_drop,y_n_retain = train_test_split(X_n,y_n,test_size= 1.0/n_subsets, random_state=0, stratify=X_n[['cust_level','product_category']])
            X_new = pd.concat([X_n_retain,X_y],axis=0)
            y_new = pd.concat([y_n_retain,y_y],axis=0)

            sf = pd.concat([X_new,pd.DataFrame(y_new)],axis=1)
            sf.to_csv("~/dataset/rec_data_train_sampled.csv",index=False,header=True)
            #no weight
            X=X_new
            y=y_new
    else:
        print("load saved training data")
        data = pd.read_csv("~/dataset/rec_data_train_save.csv", sep=',')
        data[data['age'] < 0] = np.nan
        data[data['total_balance'] < 0] = 0
        data[data['fst_invest_days'] < 0] = 0
        data[data['highest_asset_amt'] < 0] = 0
        print(data.groupby("invest").size())
        print(data.columns)
        y = data['invest']
        X = data.drop(['rd','click','invest','invest_amount','mobile_no_attribution','user_group','app_version','transfer_flag'],axis=1)
        # X = data.drop(['rd','click','invest','invest_amount','mobile_no_attribution'],axis=1)

    scale_pos_weight=(y[y==0].shape[0])*1.0/(y[y==1].shape[0])
    print("scale_pos_weight:",scale_pos_weight)
    sl = FeatureSelection()
    sl.fit(X,y)
    X=sl.transform(X)
    bfp = FeatureEncoder()
    feature_matrix = bfp.fit_transform(X)
    X_s2 = bfp.transform(X_s[sl.selected_cols])
    # with open("~/dataset/rec_data_train_feature_matrix.npz","w") as f:
    scipy.sparse.save_npz("/home/tanglek/dataset/rec_data_train_feature_matrix.npz", feature_matrix)
    print(str(bfp))

    idx2featurename = dict((y,x) for x,y in bfp.feature_names.items())

    print("feature_matrix shape:{0}".format(feature_matrix.shape))


    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, y, test_size=0.3, random_state=999,stratify=y)
    print("test set y=0:{0}".format(y_test[y_test==0].shape[0]))
    print("test set y=1:{0}".format(y_test[y_test==1].shape[0]))


    lgbm = lgb.LGBMClassifier(boosting_type='gbdt',  max_depth=4, learning_rate=0.3, n_estimators=30,scale_pos_weight = scale_pos_weight, min_child_weight=1,subsample=0.7,  colsample_bytree=0.7, reg_alpha=1e-05, reg_lambda=1)
    lgbm.fit(X_train,y_train)
    y_pre= lgbm.predict(X_test)
    y_pro= lgbm.predict_proba(X_test)[:,1]

    #single test
    y_proxs =  lgbm.predict_proba(X_s2)
    dd = pd.concat([X_s,upp[['interest_rate','display_name']],pd.DataFrame(y_proxs[:,1])],axis=1)
    dd.columns.values[-1]='proba'
    print("*"*60)
    print(dd.columns.values)
    dd = dd.sort_values(['proba'], ascending=False).groupby(['item','product_category']).head(3)
    dd[['invest_period_by_days', 'product_price','item','product_category','interest_rate','display_name','proba']].to_csv("~/dataset/rec_results{0}.csv".format(i),index=False,header=True)

