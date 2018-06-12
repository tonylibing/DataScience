# coding=utf-8
# @author:bryan
# blog: https://blog.csdn.net/bryan__
# github: https://github.com/YouChouNoBB/2018-tencent-ad-competition-baseline
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import sys
import argparse
import tensorflow as tf
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold
from category_encoders import TargetEncoder

def run(args):
    data_dir=args.data_dir
    with tf.gfile.FastGFile(os.path.join(data_dir, 'adFeature.csv')) as gf:
        ad_feature = pd.read_csv(gf)
    if tf.gfile.Exists(os.path.join(data_dir, 'userFeature.csv')):
        with tf.gfile.FastGFile(os.path.join(data_dir, 'userFeature.csv')) as gf:
            user_feature = pd.read_csv(gf)
    else:
        userFeature_data = []
        with tf.gfile.FastGFile(os.path.join(data_dir, 'userFeature.data'), 'r') as f:
            for i, line in enumerate(f):
                line = line.strip().split('|')
                userFeature_dict = {}
                for each in line:
                    each_list = each.split(' ')
                    userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
                userFeature_data.append(userFeature_dict)
                if i % 100000 == 0:
                    print(i)
            user_feature = pd.DataFrame(userFeature_data)
            with tf.gfile.FastGFile(os.path.join(data_dir, 'userFeature.csv'), 'w') as gf:
                user_feature.to_csv(gf, index=False)

    with tf.gfile.FastGFile(os.path.join(data_dir, 'train.csv'), 'r') as gf:
        train = pd.read_csv(gf)

    with tf.gfile.FastGFile(os.path.join(data_dir, 'test1.csv'), 'r') as gf:
        predict = pd.read_csv(gf)

    train.loc[train['label']==-1,'label']=0
    predict['label']=-1
    data=pd.concat([train,predict])
    data=pd.merge(data,ad_feature,on='aid',how='left')
    data=pd.merge(data,user_feature,on='uid',how='left')
    data=data.fillna('-1')
    one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
           'adCategoryId', 'productId', 'productType']
    vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
    for feature in one_hot_feature:
        try:
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])

    train=data[data.label!=-1]
    train_y=train.pop('label')
    # train, test, train_y, test_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
    test=data[data.label==-1]
    res=test[['aid','uid']]
    test=test.drop('label',axis=1)
    enc = OneHotEncoder()
    train_x=train[['creativeSize']]
    test_x=test[['creativeSize']]

    for feature in one_hot_feature:
        enc.fit(data[feature].values.reshape(-1, 1))
        train_a=enc.transform(train[feature].values.reshape(-1, 1))
        test_a = enc.transform(test[feature].values.reshape(-1, 1))
        train_x= sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
    print('one-hot prepared !')

    cv=CountVectorizer()
    for feature in vector_feature:
        cv.fit(data[feature])
        train_a = cv.transform(train[feature])
        test_a = cv.transform(test[feature])
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
    print('cv prepared !')

    def LGB_test(train_x,train_y,test_x,test_y):
        print("LGB test")
        clf = lgb.LGBMClassifier(
            boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
            max_depth=-1, n_estimators=1000, objective='binary',
            subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
            learning_rate=0.05, min_child_weight=50,random_state=2018,n_jobs=-1
        )
        clf.fit(train_x, train_y,eval_set=[(train_x, train_y),(test_x,test_y)],eval_metric='auc',early_stopping_rounds=100)
        # print(clf.feature_importances_)
        return clf,clf.best_score_[ 'valid_1']['auc']

    def LGB_predict(data_dir,train_x,train_y,test_x,res,evals,evals_y):
        print("LGB test")
        scale_pos_weight = (train_y[train_y == 0].shape[0]) * 1.0 / (train_y[train_y == 1].shape[0])
        print("scale_pos_weight:",scale_pos_weight)
        clf = lgb.LGBMClassifier(
            boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
            max_depth=-1, n_estimators=5000, objective='binary',
            subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
            learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1,scale_pos_weight=scale_pos_weight
        )
        clf.fit(train_x, train_y, eval_set=[(evals, evals_y)], eval_metric='auc',early_stopping_rounds=100)
        res['score'] = clf.predict_proba(test_x)[:,1]
        res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
        with tf.gfile.FastGFile(os.path.join(data_dir, 'submission_5000scale.csv'), 'w') as gf:
            res.to_csv(gf, index=False)
        return clf

    # model=LGB_predict(train_x,train_y,test_x,res)
    train, evals, train_y, evals_y = train_test_split(train_x,train_y,test_size=0.2, random_state=2018,stratify=train_y)
    # enc = OneHotEncoder()
    model = LGB_predict(data_dir,train, train_y, test_x, res,evals,evals_y)


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

