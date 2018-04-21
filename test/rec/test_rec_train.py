import pandas as pd
import numpy as np
import scipy
import gc
import sys
sys.path.append("../..")
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn import metrics
import xgboost as xgb
import feature.processor
from importlib import reload

reload(feature.processor)
from feature.processor import *
from imblearn.ensemble import EasyEnsemble
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score,f1_score, accuracy_score, average_precision_score,log_loss
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from eval.metrics import ks_statistic
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from model.GBDTLRClassifier import *
from wordbatch.models import FTRL, FM_FTRL

sampling_flag = True
if sampling_flag:
    if os.path.exists("/home/tanglek/dataset/rec_data_train_sampled.csv"):
        print("load saved sampling training data")
        data = pd.read_csv("~/dataset/rec_data_train_sampled.csv", sep=',')
        print(data.groupby("invest").size())
        print(data.columns)
        y = data['invest']
        data['ratio'] = data['total_balance']*1.0/(data['product_price']+1.0)
        # data['total_balance'] = np.log(data['total_balance']+1.0)
        # data['curr_aum_amt'] = np.log(data['curr_aum_amt']+1.0)
        # data['highest_asset_amt'] = np.log(data['highest_asset_amt']+1.0)
        # data['ratio2'] = data['percent_inv_amt2']*1.0/(data['product_price']+1.0)
        # data['sub_interest_rate']=data['interest_rate']-0.04
        X = data.drop(['invest', 'user_group', 'app_version', 'transfer_flag'], axis=1)
    else:
        data = pd.read_csv("~/dataset/rec_data.csv", sep=',')
        # data=pd.read_csv("~/dataset/rec_data_train_3w.csv",sep=',')
        # data=pd.read_csv("E:/dataset/rec_data_train_3w.csv",sep=',')
        # data=pd.read_csv("/media/sf_D_DRIVE/download/rec_data_train_save.csv",sep=',')
        print(data.columns.values)
        y = data['invest']
        # data.loc[data['age'] < 0] = np.nan
        # data.loc[data['total_balance'] < 0] = 0
        # data.loc[data['fst_invest_days'] < 0] = 0
        # data.loc[data['highest_asset_amt'] < 0] = 0
        # data['invest_period_by_days'].fillna(0,inplace=True)
        # X = data.drop(['rd','click','invest','invest_amount','mobile_no_attribution'],axis=1)
        X = data.drop(['invest', 'user_group', 'app_version', 'transfer_flag'], axis=1)
        # X=data[[col for col in data.columns if col not in ['invest','invest_amount']]]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999,stratify=y)

        # n_subsets = int(sum(y==0)/sum(y==1))
        n_subsets = (int(sum(y == 0) / sum(y == 1))) / 10
        # ee = EasyEnsemble(n_subsets=n_subsets)
        # sample_X, sample_y = ee.fit_sample(X, y)

        # rus = RandomUnderSampler(random_state=42)
        # X_res, y_res = rus.fit_sample(X, y)

        X_n = X[y == 0]
        y_n = y[y == 0]
        X_y = X[y == 1]
        y_y = y[y == 1]
        X_n_drop, X_n_retain, y_n_drop, y_n_retain = train_test_split(X_n, y_n, test_size=1.0 / n_subsets,
                                                                      random_state=0,
                                                                      stratify=X_n[['cust_level', 'product_category']])
        X_new = pd.concat([X_n_retain, X_y], axis=0)
        y_new = pd.concat([y_n_retain, y_y], axis=0)

        sf = pd.concat([X_new, pd.DataFrame(y_new)], axis=1)
        sf.to_csv("~/dataset/rec_data_train_sampled.csv", index=False, header=True)
        # no weight
        X = X_new
        y = y_new
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
    X = data.drop(['rd', 'click', 'invest', 'invest_amount', 'mobile_no_attribution', 'user_group', 'app_version',
                   'transfer_flag'], axis=1)
    # X = data.drop(['rd','click','invest','invest_amount','mobile_no_attribution'],axis=1)

scale_pos_weight = (y[y == 0].shape[0]) * 1.0 / (y[y == 1].shape[0])
print("scale_pos_weight:", scale_pos_weight)
sl = FeatureSelection()
sl.fit(X, y)
X = sl.transform(X)
bfp = FeatureEncoder()
feature_matrix = bfp.fit_transform(X)
# with open("~/dataset/rec_data_train_feature_matrix.npz","w") as f:
scipy.sparse.save_npz("/home/tanglek/dataset/rec_data_train_feature_matrix.npz", feature_matrix)
print(str(bfp))

idx2featurename = dict((y, x) for x, y in bfp.feature_offset.items())

print("feature_matrix shape:{0}".format(feature_matrix.shape))
# scaler = StandardScaler(with_mean=False)//auc 0.85,without 0.82
# feature_matrix = scaler.fit_transform(feature_matrix)
# scaler = QuantileTransformer(random_state=999) //auc 0.85,without 0.82
# feature_matrix = scaler.fit_transform(feature_matrix)

X_train, X_test, y_train, y_test = train_test_split(feature_matrix, y, test_size=0.3, random_state=999, shuffle=True)
ftrl = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0,  iters=3000, inv_link="identity", threads=4)
ftrl.fit(X_train, y_train)
# y_pre = ftrl.predict(X_test)
y_pro = ftrl.predict(X_test)
print("=" * 60)
print("FTRL Test AUC Score : {0}".format(roc_auc_score(y_test, y_pro)))
# print("FTRL Test Precision: {0}".format(precision_score(y_test, y_pre)))
# print("FTRL Test   Recall : {0}".format(recall_score(y_test, y_pre)))
# print("FTRL Test F1 Score: {0}".format(f1_score(y_test, y_pre)))
print("FTRL Test Test AUC of PR-curve: {0}".format(average_precision_score(y_test, y_pro)))
print("FTRL Test Test logloss: {0}".format(log_loss(y_test, y_pro)))
print("FTRL KS: {0}".format(ks_statistic(y_test, y_pro)))
# print("FTRL Test confusion_matrix :")
# print(confusion_matrix(y_test, y_pre))
print("=" * 60)
data = {"y_test": y_test, "y_pro": y_pro}
df = pd.DataFrame(data)
df.to_csv("~/dataset/ftrl_y_test_pro.csv", index=False, header=True)

fm_ftrl = FM_FTRL(alpha=0.012, beta=0.01, L1=0.00001, L2=0.1,alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,D_fm=20, e_noise=0.0001, iters=3000, inv_link="identity", threads=4)
fm_ftrl.fit(X_train, y_train)
# y_pre = fm_ftrl.predict(X_test)
y_pro = fm_ftrl.predict(X_test)
print("y_pro")
print(y_pro)
print("=" * 60)
print("FM_FTRL Test AUC Score : {0}".format(roc_auc_score(y_test, y_pro)))
# print("FM_FTRL Test Precision: {0}".format(precision_score(y_test, y_pre)))
# print("FM_FTRL Test   Recall : {0}".format(recall_score(y_test, y_pre)))
# print("FM_FTRL Test F1 Score: {0}".format(f1_score(y_test, y_pre)))
print("FM_FTRL Test Test AUC of PR-curve: {0}".format(average_precision_score(y_test, y_pro)))
print("FM_FTRL Test Test logloss: {0}".format(log_loss(y_test, y_pro)))
print("FM_FTRL KS: {0}".format(ks_statistic(y_test, y_pro)))
# print("FM_FTRL Test confusion_matrix :")
# print(confusion_matrix(y_test, y_pre))
print("=" * 60)
data = {"y_test": y_test, "y_pro": y_pro}
df = pd.DataFrame(data)
df.to_csv("~/dataset/fm_ftrl_y_test_pro.csv", index=False, header=True)

X_train, X_test, y_train, y_test = train_test_split(feature_matrix, y, test_size=0.3, random_state=999, stratify=y)
print("test set y=0:{0}".format(y_test[y_test == 0].shape[0]))
print("test set y=1:{0}".format(y_test[y_test == 1].shape[0]))

lr = LogisticRegression(C=1.0, penalty='l2', tol=1e-4, solver='liblinear', random_state=42,class_weight='balanced')
lr.fit(X_train, y_train)
y_pre = lr.predict(X_test)
y_pro = lr.predict_proba(X_test)[:, 1]
print("=" * 60)
print("LR Test AUC Score : {0}".format(roc_auc_score(y_test, y_pro)))
print("LR Test Precision: {0}".format(precision_score(y_test, y_pre)))
print("LR Test   Recall : {0}".format(recall_score(y_test, y_pre)))
print("LR Test F1 Score: {0}".format(f1_score(y_test, y_pre)))
print("LR Test Test AUC of PR-curve: {0}".format(average_precision_score(y_test, y_pro)))
print("LR Test Test logloss: {0}".format(log_loss(y_test, y_pro)))
print("LR KS: {0}".format(ks_statistic(y_test, y_pro)))
print("LR Test confusion_matrix :")
print(confusion_matrix(y_test, y_pre))
print("=" * 60)
data = {"y_test": y_test, "y_pro": y_pro}
df = pd.DataFrame(data)
df.to_csv("~/dataset/lr_y_test_pro.csv", index=False, header=True)

gbm = xgb.XGBClassifier(n_estimators=30, learning_rate=0.3, max_depth=4, min_child_weight=1, gamma=0.3, subsample=0.7,
                        colsample_bytree=0.7, objective='binary:logistic', nthread=-1,
                        scale_pos_weight=scale_pos_weight, reg_alpha=1e-05, reg_lambda=1, seed=27)
gbm.fit(X_train, y_train)
y_pre = gbm.predict(X_test)
# y_pre_leaf = gbm.predict(X_test,pred_leaf=True)
# print(y_pre_leaf.shape)
y_pro = gbm.predict_proba(X_test)[:, 1]
y_pro2 = gbm.predict_proba(X_train)[:, 1]
print("=" * 60)
# print("Xgboost model Training AUC Score: {0}".format(roc_auc_score(y_train, y_pro2)))
print("Xgboost model Test AUC Score: {0}".format(roc_auc_score(y_test, y_pro)))
print("Xgboost model Test Precision: {0}".format(precision_score(y_test, y_pre)))
print("Xgboost model Test   Recall : {0}".format(recall_score(y_test, y_pre)))
print("Xgboost model Test F1 Score: {0}".format(f1_score(y_test, y_pre)))
print("Xgboost model Test AUC of PR-curve: {0}".format(average_precision_score(y_test, y_pro)))
print("Xgboost model Test logloss: {0}".format(log_loss(y_test, y_pro)))
print("Xgboost KS: {0}".format(ks_statistic(y_test, y_pro)))
print("Xgboost Test confusion_matrix :")
print(confusion_matrix(y_test, y_pre))
print("=" * 60)
data = {"y_test": y_test, "y_pro": y_pro}
df = pd.DataFrame(data)
df.to_csv("~/dataset/xgb_y_test_pro.csv", index=False, header=True)

lgbm = lgb.LGBMClassifier(boosting_type='gbdt', max_depth=4, learning_rate=0.3, n_estimators=30,
                          scale_pos_weight=scale_pos_weight, min_child_weight=1, subsample=0.7, colsample_bytree=0.7,
                          reg_alpha=1e-05, reg_lambda=1)
lgbm.fit(X_train, y_train)
y_pre = lgbm.predict(X_test)
y_pro = lgbm.predict_proba(X_test)[:, 1]
print("=" * 60)
print("lightgbm model Test AUC Score: {0}".format(roc_auc_score(y_test, y_pro)))
print("lightgbm model Test Precision: {0}".format(precision_score(y_test, y_pre)))
print("lightgbm model Test   Recall : {0}".format(recall_score(y_test, y_pre)))
print("lightgbm model Test F1 Score: {0}".format(f1_score(y_test, y_pre)))
print("lightgbm model Test AUC of PR-curve: {0}".format(average_precision_score(y_test, y_pro)))
print("lightgbm model Test logloss: {0}".format(log_loss(y_test, y_pro)))
print("Lightgbm KS: {0}".format(ks_statistic(y_test, y_pro)))
print("Lightgbm Test confusion_matrix :")
print(confusion_matrix(y_test, y_pre))
print("=" * 60)
data = {"y_test": y_test, "y_pro": y_pro}
df = pd.DataFrame(data)
df.to_csv("~/dataset/lgb_y_test_pro.csv", index=False, header=True)

gbdtlr = XgboostLRClassifier(scale_pos_weight=scale_pos_weight)
gbdtlr.fit(X_train, y_train)
y_pre = gbdtlr.predict(X_test)
y_pro = gbdtlr.predict_proba(X_test)[:, 1]
print("=" * 60)
print("Xgboost+LR Test AUC Score : {0}".format(roc_auc_score(y_test, y_pro)))
print("Xgboost+LR  Test Precision: {0}".format(precision_score(y_test, y_pre)))
print("Xgboost+LR  Test   Recall : {0}".format(recall_score(y_test, y_pre)))
print("Xgboost+LR Test F1 Score: {0}".format(f1_score(y_test, y_pre)))
print("Xgboost+LR  Test AUC of PR-curve: {0}".format(average_precision_score(y_test, y_pro)))
print("Xgboost+LR Test logloss: {0}".format(log_loss(y_test, y_pro)))
print("Xgboost+LR KS: {0}".format(ks_statistic(y_test, y_pro)))
print("Xgboost+LR  Test confusion_matrix :")
print(confusion_matrix(y_test, y_pre))
print("=" * 60)
fi = gbdtlr.feature_importance()
fi = list(zip(list(range(len(fi))), fi))
fi = sorted(fi, key=lambda tup: tup[1], reverse=True)
for idx, importance in fi:
    print("{0}:{1}".format(idx2featurename[idx], importance))

data = {"y_test": y_test, "y_pro": y_pro}
df = pd.DataFrame(data)
df.to_csv("~/dataset/xgb_lr_no_combine_y_test_pro.csv", index=False, header=True)

lgbmlr = LightgbmLRClassifier(scale_pos_weight=scale_pos_weight)
lgbmlr.fit(X_train, y_train)
y_pre = lgbmlr.predict(X_test)
y_pro = lgbmlr.predict_proba(X_test)[:, 1]
y_pro2 = gbm.predict_proba(X_train)[:, 1]
print("=" * 60)
# print("Lightgbm+LR Training AUC Score : {0}".format(roc_auc_score(y_train, y_pro2)))
print("Lightgbm+LR Test AUC Score : {0}".format(roc_auc_score(y_test, y_pro)))
print("Lightgbm+LR  Test Precision: {0}".format(precision_score(y_test, y_pre)))
print("Lightgbm+LR  Test   Recall : {0}".format(recall_score(y_test, y_pre)))
print("Lightgbm+LR Test F1 Score: {0}".format(f1_score(y_test, y_pre)))
print("Lightgbm+LR Test AUC of PR-curve: {0}".format(average_precision_score(y_test, y_pro)))
print("Lightgbm+LR Test logloss: {0}".format(log_loss(y_test, y_pro)))
print("Lightgbm+LR KS: {0}".format(ks_statistic(y_test, y_pro)))
print("Lightgbm+LR  Test confusion_matrix :")
print(confusion_matrix(y_test, y_pre))

data = {"y_test": y_test, "y_pro": y_pro}
df = pd.DataFrame(data)
df.to_csv("~/dataset/lgb_lr_no_combine_y_test_pro.csv", index=False, header=True)

print("=" * 60)
fi = lgbmlr.feature_importance()
fi = list(zip(list(range(len(fi))), fi))
fi = sorted(fi, key=lambda tup: tup[1], reverse=True)
for idx, importance in fi:
    print("{0}:{1}".format(idx2featurename[idx], importance))

