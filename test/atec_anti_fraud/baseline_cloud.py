import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import sys
import argparse
import gc
import math
import numpy as np
from tqdm import tqdm
import tensorflow as tf
def run(args):
    data_dir = args.data_dir
    with tf.gfile.FastGFile(os.path.join(data_dir, "atec_anti_fraud_train.csv"), 'r') as gf:
        train = pd.read_csv(gf)
    with tf.gfile.FastGFile(os.path.join(data_dir, "atec_anti_fraud_test_a.csv"), 'r') as gf:
        test = pd.read_csv(gf)

    continous_feature = ["f" + str(i) for i in range(1, 298)]

    for feature in tqdm(continous_feature, desc='feature'):
        train[feature].fillna(train[feature].median(), inplace=True)
        test[feature].fillna(train[feature].median(), inplace=True)

    train.loc[train['label'] == -1, 'label'] = 1
    train_x = train[continous_feature]
    train_y = train['label']
    test_x = test[continous_feature]
    res = test[['id']]
    scale_pos_weight = (train_y[train_y == 0].shape[0]) * 1.0 / (train_y[train_y == 1].shape[0])
    print("scale_pos_weight:", scale_pos_weight)

    def LGB_predict(train_x, train_y, test_x, res):
        print("LGB test")
        clf = lgb.LGBMClassifier(
            boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
            max_depth=-1, n_estimators=5000, objective='binary',
            subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
            learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=1, scale_pos_weight=scale_pos_weight
        )
        clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc', early_stopping_rounds=100)
        res['score'] = clf.predict(test_x)
        with tf.gfile.FastGFile(os.path.join(data_dir, "submission.csv"), 'w') as gf:
            res.to_csv(gf, index=False)
        return clf

    model = LGB_predict(train_x, train_y, test_x, res)

def main(_):
    args_in = sys.argv[1:]
    print(args_in)
    parser = argparse.ArgumentParser()
    mtyunArgs = parser.add_argument_group('cloud option')
    mtyunArgs.add_argument('--data_dir', type=str, default='',
                           help='input data path')
    mtyunArgs.add_argument('--model_dir', type=str, default='',help='output model path')
    mtyunArgs.add_argument('--model_type', type=str, default='',help='model type')
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
    run(args)


if __name__ == '__main__':
    tf.app.run(main=main)