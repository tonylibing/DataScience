# -*- coding: utf-8 -*-
import os
import argparse
import processor
import argparse
import os
import sys
sys.path.append("../..")
import scipy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score, \
    average_precision_score
from eval.metrics import ks_statistic
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import torch
import os.path

from importlib import reload
import feature.processor

reload(feature.processor)
from feature.processor import *

import model.wide_deep.torch_model

reload(model.wide_deep.torch_model)
from model.wide_deep.torch_model import WideDeep

import model.wide_deep.data_utils

reload(model.wide_deep.data_utils)
from model.wide_deep.data_utils import prepare_data

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
        if tf.gfile.Exists(os.path.join(args.data_dir, "rec_data_train_sampled.csv")):
            print("load saved sampling training data")
            with tf.gfile.FastGFile(os.path.join(args.data_dir, "rec_data_train_sampled.csv"), 'rb') as gf:
                data = pd.read_csv(gf, sep=',')
            print(data.groupby("invest").size())
            print(data.columns)
            y = data['invest']
            X = data.drop(['invest'], axis=1)
        else:
            with tf.gfile.FastGFile(os.path.join(args.data_dir, "rec_data_train_save.csv"), 'rb') as gf:
                data = pd.read_csv(gf, sep=',')
            # data=pd.read_csv("~/dataset/rec_data_train_3w.csv",sep=',')
            # data=pd.read_csv("E:/dataset/rec_data_train_3w.csv",sep=',')
            # data=pd.read_csv("/media/sf_D_DRIVE/download/rec_data_train_save.csv",sep=',')
            print(data.columns.values)
            y = data['invest']
            data[data['age'] < 0] = np.nan
            data[data['total_balance'] < 0] = 0
            data[data['fst_invest_days'] < 0] = 0
            data[data['highest_asset_amt'] < 0] = 0
            data['invest_period_by_days'].fillna(0, inplace=True)
            X = data.drop(['rd', 'click', 'invest', 'invest_amount', 'mobile_no_attribution'], axis=1)
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
                                                                          random_state=0, stratify=X_n[
                    ['cust_level', 'product_category']])
            X_new = pd.concat([X_n_retain, X_y], axis=0)
            y_new = pd.concat([y_n_retain, y_y], axis=0)

            sf = pd.concat([X_new, pd.DataFrame(y_new)], axis=1)
            with tf.gfile.FastGFile(os.path.join(args.data_dir, "rec_data_train_sampled.csv"), 'wb') as gf:
                sf.to_csv(gf, index=False, header=True)
            # no weight
            X = X_new
            y = y_new
    else:
        print("load saved training data rec_data.csv")
        with tf.gfile.FastGFile(os.path.join(args.data_dir, "rec_data.csv"), 'rb') as gf:
            data = pd.read_csv(gf, sep=',')
        # data.loc[data['age'] < 0] = np.nan
        # data.loc[data['total_balance'] < 0] = 0
        # data.loc[data['fst_invest_days'] < 0] = 0
        # data.loc[data['highest_asset_amt'] < 0] = 0
        # data['invest_period_by_days'].fillna(0,inplace=True)
        print(data.groupby("invest").size())
        print(data.columns)
        y = data['invest']
        X = data.drop(['invest'], axis=1)
        # X = data.drop(['rd','click','rec','nationality','invest_amount','mobile_no_attribution'],axis=1)

    scale_pos_weight = (y[y == 0].shape[0]) * 1.0 / (y[y == 1].shape[0])
    print("scale_pos_weight:", scale_pos_weight)

    use_cuda = torch.cuda.is_available()
    # Experiment set up
    # "user_id","transfer_flag",
    wide_cols = ["app_version", "hourOfDay", "dayOfWeek", "dayOfMonth", "address", "risk_verify_status_cn",
                 "product_category", "risk_level", "item", "user_group", "cust_level", "gender", "age"]
    crossed_cols = (['education', 'occupation'], ['native_country', 'occupation'])
    embeddings_cols = [('education', 10), ('relationship', 8), ('workclass', 10), ('occupation', 10),
                       ('native_country', 10)]
    continuous_cols = ["total_balance", "fst_invest_days", "invest_period_by_days", "product_price", "curr_aum_amt",
                       "highest_asset_amt", "B2C", "P2P", "XJ", "eshare", "hshare", "avg_inv_amt", "max_inv_amt",
                       "percent_inv_amt1", "percent_inv_amt2", "percent_inv_amt3", "12m-24m", "1m-3m", "24m-36m",
                       "36m+", "3m-6m", "6m-12m", "<1m"]
    target = 'invest'
    method = 'logistic'

    # Prepare data
    wd_dataset = prepare_data(
        data, wide_cols,
        crossed_cols,
        embeddings_cols,
        continuous_cols,
        target,
        scale=True)

    # Network set up
    wide_dim = wd_dataset['train_dataset'].wide.shape[1]
    n_unique = len(np.unique(wd_dataset['train_dataset'].labels))
    if (method == "regression") or (method == "logistic"):
        n_class = 1
    else:
        n_class = n_unique
    deep_column_idx = wd_dataset['deep_column_idx']
    embeddings_input = wd_dataset['embeddings_input']
    encoding_dict = wd_dataset['encoding_dict']
    hidden_layers = [100, 50]
    dropout = [0.5, 0.2]

    model = WideDeep(
        wide_dim,
        embeddings_input,
        continuous_cols,
        deep_column_idx,
        hidden_layers,
        dropout,
        encoding_dict,
        n_class)
    model.compile(method=method)
    if use_cuda:
        model = model.cuda()

    train_dataset = wd_dataset['train_dataset']
    model.fit(dataset=train_dataset, n_epochs=10, batch_size=64)

    test_dataset = wd_dataset['test_dataset']
    print(model.predict(dataset=test_dataset)[:10])
    print(model.predict_proba(dataset=test_dataset)[:10])
    print(model.get_embeddings('education'))

    # save
    MODEL_DIR = args.model_dir
    # if not os.path.exists(MODEL_DIR):
    #     os.makedirs(MODEL_DIR)
    with tf.gfile.FastGFile(os.path.join(MODEL_DIR, 'logistic.pkl'), 'wb') as gf:
        torch.save(model.state_dict(), gf)


if __name__ == '__main__':
    tf.app.run(main=main)