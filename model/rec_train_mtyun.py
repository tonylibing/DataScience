import sys
import os
import argparse
import processor
import argparse
import os
import sys
from importlib import reload

import processor

reload(processor)
from processor import *
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score, average_precision_score
from metrics import ks_statistic
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import os.path
import numpy as np

FLAGS = None


def main(_):
    args_in = sys.argv[1:]
    print(args_in)
    parser = argparse.ArgumentParser()
    mtyunArgs = parser.add_argument_group('美团云选项')
    mtyunArgs.add_argument('--data_dir', type=str, default='',
                           help='input data path')
    mtyunArgs.add_argument('--model_dir', type=str, default='',
                           help='output model path')
    mtyunArgs.add_argument('--tf_fs', type=str, default='', help='output model path')
    mtyunArgs.add_argument('--tf_prefix', type=str, default='', help='output model path')
    mtyunArgs.add_argument('--default_fs', type=str, default='', help='output model path')
    mtyunArgs.add_argument('--worker_num', type=str, default='', help='output model path')
    mtyunArgs.add_argument('--num_gpus', type=str, default='', help='output model path')
    mtyunArgs.add_argument('--num_ps', type=str, default='', help='output model path')
    mtyunArgs.add_argument('--num_worker', type=str, default='', help='output model path')
    mtyunArgs.add_argument('--tensorboard_dir', type=str, default='', help='output model path')
    mtyunArgs.add_argument('--tb_dir', type=str, default='local_tensorbord_dir_0', help='output model path')
    FLAGS, _ = parser.parse_known_args()
    print('FLAGS')
    print(FLAGS)
    args = parser.parse_args(args_in)

    sampling_flag = False
    if sampling_flag:
        if tf.gfile.Exists(os.path.join(args.data_dir,"rec_data_train_sampled.csv")):
            print("load saved sampling training data")
            with tf.gfile.FastGFile(os.path.join(args.data_dir,"rec_data_train_sampled.csv", 'r')) as gf:
                data = pd.read_csv(gf, sep=',')
            print(data.groupby("invest").size())
            print(data.columns)
            y = data['invest']
            X = data.drop(['invest'], axis=1)
        else:
            with tf.gfile.FastGFile(os.path.join(args.data_dir, "rec_data_train_save.csv", 'r')) as gf:
                data=pd.read_csv(gf,sep=',')
            # data=pd.read_csv("~/dataset/rec_data_train_3w.csv",sep=',')
            # data=pd.read_csv("E:/dataset/rec_data_train_3w.csv",sep=',')
            #data=pd.read_csv("/media/sf_D_DRIVE/download/rec_data_train_save.csv",sep=',')
            print(data.columns.values)
            y=data['invest']
            data[data['age'] < 0] = np.nan
            data[data['total_balance'] < 0] = 0
            data[data['fst_invest_days'] < 0] = 0
            data[data['highest_asset_amt'] < 0] = 0
            X = data.drop(['rd','click','invest','invest_amount','mobile_no_attribution'],axis=1)
            #X=data[[col for col in data.columns if col not in ['invest','invest_amount']]]
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999,stratify=y)

            # n_subsets = int(sum(y==0)/sum(y==1))
            n_subsets = (int(sum(y==0)/sum(y==1)))/10
            # ee = EasyEnsemble(n_subsets=n_subsets)
            # sample_X, sample_y = ee.fit_sample(X, y)

            # rus = RandomUnderSampler(random_state=42)
            # X_res, y_res = rus.fit_sample(X, y)

            from sklearn.model_selection import train_test_split
            X_n = X[y==0]
            y_n = y[y==0]
            X_y = X[y==1]
            y_y = y[y==1]
            X_n_drop,X_n_retain,y_n_drop,y_n_retain = train_test_split(X_n,y_n,test_size= 1.0/n_subsets, random_state=0, stratify=X_n[['cust_level','product_category']])
            X_new = pd.concat([X_n_retain,X_y],axis=0)
            y_new = pd.concat([y_n_retain,y_y],axis=0)

            sf = pd.concat([X_new,pd.DataFrame(y_new)],axis=1)
            with tf.gfile.FastGFile(os.path.join(args.data_dir, "rec_data_train_sampled.csv", 'w')) as gf:
                sf.to_csv(gf,index=False,header=True)
            #no weight
            X=X_new
            y=y_new
    else:
        print("load saved training data")
        with tf.gfile.FastGFile(os.path.join(args.data_dir, "rec_data_train_save.csv", 'r')) as gf:
            data = pd.read_csv(gf, sep=',')
        data[data['age'] < 0] = np.nan
        data[data['total_balance'] < 0] = 0
        data[data['fst_invest_days'] < 0] = 0
        data[data['highest_asset_amt'] < 0] = 0
        print(data.groupby("invest").size())
        print(data.columns)
        y = data['invest']
        X = data.drop(['rd','click','invest','invest_amount','mobile_no_attribution'],axis=1)

    scale_pos_weight=(y[y==0].shape[0])*1.0/(y[y==1].shape[0])
    print("scale_pos_weight:",scale_pos_weight)
    bfp = FeatureProcessor(X,y)
    feature_matrix = bfp.fit_transform(X)
    print(str(bfp))

    idx2featurename = dict((y,x) for x,y in bfp.feature_names.items())

    print("feature_matrix shape:{0}".format(feature_matrix.shape))


    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, y, test_size=0.2, random_state=999,stratify=y)
    print("test set y=0:{0}".format(y_test[y_test==0].shape[0]))
    print("test set y=1:{0}".format(y_test[y_test==1].shape[0]))

    lr = LogisticRegression(C=1.0, penalty='l2', tol=1e-4,solver='liblinear',random_state=42)
    lr.fit(X_train,y_train)
    y_pre= lr.predict(X_test)
    y_pro= lr.predict_proba(X_test)[:,1]
    print("="*60)
    print("LR Test AUC Score : {0}".format(roc_auc_score(y_test, y_pro)))
    print("LR  Test Precision: {0}".format(precision_score(y_test, y_pre)))
    print("LR  Test   Recall : {0}".format(recall_score(y_test, y_pre)))
    print("LR  Test Test AUC of PR-curve: {0}".format(average_precision_score(y_test, y_pro)))
    print("Lightgbm+LR  Test confusion_matrix :")
    print(confusion_matrix(y_test, y_pre))
    print("="*60)

    gbm = xgb.XGBClassifier(n_estimators=30,learning_rate =0.3,max_depth=3,min_child_weight=1,gamma=0.3,subsample=0.7,colsample_bytree=0.7,objective= 'binary:logistic',nthread=-1,scale_pos_weight = scale_pos_weight,reg_alpha=1e-05,reg_lambda=1,seed=27)
    gbm.fit(X_train,y_train)
    y_pre= gbm.predict(X_test)
    # y_pre_leaf = gbm.predict(X_test,pred_leaf=True)
    # print(y_pre_leaf.shape)
    y_pro= gbm.predict_proba(X_test)[:,1]
    print("="*60)
    print("Xgboost model Test AUC Score: {0}".format(roc_auc_score(y_test, y_pro)))
    print("Xgboost model Test Precision: {0}".format(precision_score(y_test, y_pre)))
    print("Xgboost model Test   Recall : {0}".format(recall_score(y_test, y_pre)))
    print("Xgboost model Test AUC of PR-curve: {0}".format(average_precision_score(y_test, y_pro)))
    print("Lightgbm+LR  Test confusion_matrix :")
    print(confusion_matrix(y_test, y_pre))
    print("="*60)

    lgbm = lgb.LGBMClassifier(boosting_type='gbdt',  max_depth=3, learning_rate=0.3, n_estimators=30,scale_pos_weight = scale_pos_weight, min_child_weight=1,subsample=0.7,  colsample_bytree=0.7, reg_alpha=1e-05, reg_lambda=1)
    lgbm.fit(X_train,y_train)
    y_pre= lgbm.predict(X_test)
    y_pro= lgbm.predict_proba(X_test)[:,1]
    print("="*60)
    print("lightgbm model Test AUC Score: {0}".format(roc_auc_score(y_test, y_pro)))
    print("lightgbm model Test Precision: {0}".format(precision_score(y_test, y_pre)))
    print("lightgbm model Test   Recall : {0}".format(recall_score(y_test, y_pre)))
    print("lightgbm model Test AUC of PR-curve: {0}".format(average_precision_score(y_test, y_pro)))
    print("Lightgbm+LR  Test confusion_matrix :")
    print(confusion_matrix(y_test, y_pre))
    print("="*60)

    gbdtlr = XgboostLRClassifier(scale_pos_weight=scale_pos_weight)
    gbdtlr.fit(X_train,y_train)
    y_pre= gbdtlr.predict(X_test)
    y_pro= gbdtlr.predict_proba(X_test)[:,1]
    print("="*60)
    print("Xgboost+LR Test AUC Score : {0}".format(roc_auc_score(y_test, y_pro)))
    print("Xgboost+LR  Test Precision: {0}".format(precision_score(y_test, y_pre)))
    print("Xgboost+LR  Test   Recall : {0}".format(recall_score(y_test, y_pre)))
    print("Xgboost+LR Test AUC of PR-curve: {0}".format(average_precision_score(y_test, y_pro)))
    print("Lightgbm+LR  Test confusion_matrix :")
    print(confusion_matrix(y_test, y_pre))
    print("="*60)

    fi = gbdtlr.feature_importance()
    fi = list(zip(list(range(len(fi))),fi))
    fi = sorted(fi, key=lambda tup: tup[1],reverse=True)
    for idx,importance in fi:
        print("{0}:{1}".format(idx2featurename[idx],importance))


    lgbmlr = LightgbmLRClassifier(scale_pos_weight=scale_pos_weight)
    lgbmlr.fit(X_train,y_train)
    y_pre= lgbmlr.predict(X_test)
    y_pro= lgbmlr.predict_proba(X_test)[:,1]
    print("="*60)
    print("Lightgbm+LR Test AUC Score : {0}".format(roc_auc_score(y_test, y_pro)))
    print("Lightgbm+LR  Test Precision: {0}".format(precision_score(y_test, y_pre)))
    print("Lightgbm+LR  Test   Recall : {0}".format(recall_score(y_test, y_pre)))
    print("Lightgbm+LR  Test AUC of PR-curve: {0}".format(average_precision_score(y_test, y_pro)))
    print("Lightgbm+LR  Test confusion_matrix :")
    print(confusion_matrix(y_test, y_pre))
    print("="*60)

    fi = lgbmlr.feature_importance()
    fi = list(zip(list(range(len(fi))),fi))
    fi = sorted(fi, key=lambda tup: tup[1],reverse=True)
    for idx,importance in fi:
        print("{0}:{1}".format(idx2featurename[idx],importance))

    gbdtlr = XgboostLRClassifier(combine_feature=False,scale_pos_weight=scale_pos_weight)
    gbdtlr.fit(X_train,y_train)
    y_pre= gbdtlr.predict(X_test)
    y_pro= gbdtlr.predict_proba(X_test)[:,1]
    print("="*60)
    print("Xgboost+LR Test AUC Score : {0}".format(roc_auc_score(y_test, y_pro)))
    print("Xgboost+LR  Test Precision: {0}".format(precision_score(y_test, y_pre)))
    print("Xgboost+LR  Test   Recall : {0}".format(recall_score(y_test, y_pre)))
    print("Xgboost+LR  Test AUC of PR-curve: {0}".format(average_precision_score(y_test, y_pro)))
    print("Lightgbm+LR  Test confusion_matrix :")
    print(confusion_matrix(y_test, y_pre))
    print("="*60)
    fi = gbdtlr.feature_importance()
    fi = list(zip(list(range(len(fi))),fi))
    fi = sorted(fi, key=lambda tup: tup[1],reverse=True)
    for idx,importance in fi:
        print("{0}:{1}".format(idx2featurename[idx],importance))

    lgbmlr = LightgbmLRClassifier(combine_feature=False,scale_pos_weight=scale_pos_weight)
    lgbmlr.fit(X_train,y_train)
    y_pre= lgbmlr.predict(X_test)
    y_pro= lgbmlr.predict_proba(X_test)[:,1]
    print("="*60)
    print("Lightgbm+LR Test AUC Score : {0}".format(roc_auc_score(y_test, y_pro)))
    print("Lightgbm+LR  Test Precision: {0}".format(precision_score(y_test, y_pre)))
    print("Lightgbm+LR  Test   Recall : {0}".format(recall_score(y_test, y_pre)))
    print("Lightgbm+LR Test AUC of PR-curve: {0}".format(average_precision_score(y_test, y_pro)))
    print("Lightgbm+LR              KS: {0}".format(ks_statistic(y_test, y_pro)))
    print("Lightgbm+LR  Test confusion_matrix :")
    print(confusion_matrix(y_test, y_pre))
    print("="*60)
    fi = lgbmlr.feature_importance()
    fi = list(zip(list(range(len(fi))),fi))
    fi = sorted(fi, key=lambda tup: tup[1],reverse=True)
    for idx,importance in fi:
        print("{0}:{1}".format(idx2featurename[idx],importance))


if __name__ == '__main__':
    tf.app.run(main=main)