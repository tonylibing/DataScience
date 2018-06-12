# -*- coding:utf-8 -*-
#
import os
import sys
import io
import argparse
import pandas as pd
import numpy as np
import math
import gc
import lightgbm as lgb
import xlearn as xl
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from gensim.models.word2vec import Word2Vec
import collections
from tqdm import tqdm
import tensorflow as tf
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold
from category_encoders import TargetEncoder
from sklearn.datasets import dump_svmlight_file


def base_word2vec(x, model, size):
    vec = np.zeros(size)
    x = [item for item in x if model.wv.__contains__(item)]

    for item in x:
        vec += model.wv[item]
    if len(x) == 0:
        return vec
    else:
        return vec / len(x)


def select_topk(data):
    """
    选择频率最高的 k 个word 此处为前20%
    :param data:
    :return:
    """
    word_list = []
    for words in data:
        word_list += words
    result = collections.Counter(word_list)
    size = len(result)
    result = result.most_common(int(size * 0.8))

    word_dict = {}
    for re in result:
        word_dict[re[0]] = 1
    print('word_vec: ', size, len(result))
    return word_dict


def base_process(data):
    one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                       'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                       'adCategoryId', 'productId', 'productType']

    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                      'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']

    lbc = LabelEncoder()
    for feature in one_hot_feature:
        print("this is feature:", feature)
        try:
            data[feature] = lbc.fit_transform(data[feature].apply(int))
        except:
            data[feature] = lbc.fit_transform(data[feature])

    for feature in vector_feature:
        print("this is feature:", feature)

        data[feature] = data[feature].apply(lambda x: str(x).split(' '))
        word_dict = select_topk(data[feature])
        data[feature] = data[feature].apply(lambda x: ' '.join(
            [word for word in x if word_dict.__contains__(word)]))

        model = Word2Vec(data[feature], size=10, min_count=1, iter=5, window=2)
        data_vec = []
        for row in data[feature]:
            data_vec.append(base_word2vec(row, model, size=10))
        column_names = []
        for i in range(10):
            column_names.append(feature + str(i))
        data_vec = pd.DataFrame(data_vec, columns=column_names)
        data = pd.concat([data, data_vec], axis=1)
        del data[feature]
    return data


def bayessearchcv(X, y, args):
    print('init tuning...')
    model_dir = args.model_dir

    # clf = lgb.LGBMClassifier(
    #     boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    #     max_depth=-1, n_estimators=5000, objective='binary',
    #     subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    #     learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
    # )
    task_type = args.task
    bayes_cv_tuner = BayesSearchCV(
        estimator=lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            n_jobs=8),
        search_spaces={
            'learning_rate': (0.01, 0.3, 'log-uniform'),
            'num_leaves': (1, 100),
            'max_depth': (3, 6),
            'min_child_samples': (0, 50),
            'max_bin': (100, 1000),
            'subsample': (0.5, 1.0, 'uniform'),
            'subsample_freq': (0, 10),
            'colsample_bytree': (0.5, 1.0, 'uniform'),
            'min_child_weight': (0, 80),
            'subsample_for_bin': (100000, 500000),
            'reg_lambda': (1e-9, 1000, 'log-uniform'),
            'reg_alpha': (1e-9, 1.0, 'log-uniform'),
            'scale_pos_weight': (1, 500, 'log-uniform'),
            'n_estimators': (5000, 10000),
        },
        scoring='roc_auc',
        cv=StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        ),
        n_jobs=1,
        n_iter=100,
        verbose=0,
        refit=True,
        random_state=42
    )

    def status_print(optim_result):
        """Status callback durring bayesian hyperparameter search"""

        # Get all the models tested so far in DataFrame format
        all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)

        # Get current parameters and the best parameters
        best_params = pd.Series(bayes_cv_tuner.best_params_)
        print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
            len(all_models),
            np.round(bayes_cv_tuner.best_score_, 4),
            bayes_cv_tuner.best_params_
        ))

        # Save all model results
        clf_name = bayes_cv_tuner.estimator.__class__.__name__
        with tf.gfile.FastGFile(os.path.join(model_dir, clf_name, task_type, "_cv_results.csv"), 'wb') as gf:
            all_models.to_csv(gf)

    # Fit the model
    print('bayes tuning...')
    result = bayes_cv_tuner.fit(X, y, callback=status_print)
    return result


def run_with_embedding(args):
    data_dir = args.data_dir
    # data_dir = "/home/tanglek/dataset/tencent_lookalike/preliminary_contest_data"
    if tf.gfile.Exists(os.path.join(data_dir, 'data_with_embedding.csv')):
        with tf.gfile.FastGFile(os.path.join(data_dir, 'data_with_embedding.csv'), 'r') as gf:
            data = pd.read_csv(gf)
    else:
        if tf.gfile.Exists(os.path.join(data_dir, 'all_data.csv')):
            with tf.gfile.FastGFile(os.path.join(data_dir, 'all_data.csv'), 'r') as gf:
                data = pd.read_csv(gf)
        else:
            with tf.gfile.FastGFile(os.path.join(data_dir, 'adFeature.csv'), 'r') as gf:
                ad_feature = pd.read_csv(gf)

            if tf.gfile.Exists(os.path.join(data_dir, 'userFeature.csv')):
                with tf.gfile.FastGFile(os.path.join(data_dir, 'userFeature.csv'), 'r') as gf:
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

            train.loc[train['label'] == -1, 'label'] = 0
            predict['label'] = -1
            data = pd.concat([train, predict])
            data = pd.merge(data, ad_feature, on='aid', how='left')
            data = pd.merge(data, user_feature, on='uid', how='left')
            data = data.fillna('-1')

            with tf.gfile.FastGFile(os.path.join(data_dir, 'all_data.csv'), 'w') as gf:
                data.to_csv(gf, index=False)

        data = base_process(data)
        print("data columns after embedding:", data.columns.values)
        with tf.gfile.FastGFile(os.path.join(data_dir, 'data_with_embedding.csv'), 'w') as gf:
            data.to_csv(gf, index=False)

    # one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
    #        'adCategoryId', 'productId', 'productType']
    # vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
    # for feature in one_hot_feature:
    #     try:
    #         data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    #     except:
    #         data[feature] = LabelEncoder().fit_transform(data[feature])
    #
    print("data.columns:", data.columns.values)
    cols = [col for col in data.columns if col not in ['aid', 'uid']]
    train = data.loc[data.label != -1, cols]
    train_y = train.pop('label')
    train_x = train
    test = data.loc[data.label == -1, cols]
    res = data.loc[data.label == -1, ['aid', 'uid']]
    test_x = test.drop(['label'], axis=1)
    train, evals, train_y, evals_y = train_test_split(train, train_y, test_size=0.2, random_state=2018,
                                                      stratify=train_y)
    # enc = OneHotEncoder()
    model = LGB_predict(args, train, train_y, test_x, res, evals, evals_y)
    # for feature in one_hot_feature:
    #     enc.fit(data[feature].values.reshape(-1, 1))
    #     train_a=enc.transform(train[feature].values.reshape(-1, 1))
    #     test_a = enc.transform(test[feature].values.reshape(-1, 1))
    #     evals_a = enc.transform(evals[feature].values.reshape(-1, 1))
    #     train_x= sparse.hstack((train_x, train_a))
    #     test_x = sparse.hstack((test_x, test_a))
    #     test_x = sparse.hstack((evals_x, evals_a))
    # print('one-hot prepared !')
    #
    # cv=CountVectorizer()
    # for feature in vector_feature:
    #     cv.fit(data[feature])
    #     train_a = cv.transform(train[feature])
    #     evals_a = cv.transform(evals[feature])
    #     test_a = cv.transform(test[feature])
    #     train_x = sparse.hstack((train_x, train_a))
    #     evals_x = sparse.hstack((evals_x, evals_a))
    #     test_x = sparse.hstack((test_x, test_a))
    # print('cv prepared !')


def LGB_predict(args, train_x, train_y, test_x, res, evals_x, evals_y, gpu=False):
    print("LGB test")
    # print("train_x:",train_x.columns.values)
    # print("test_x:",test_x.columns.values)
    # print("evals_x:",evals_x.columns.values)
    if gpu:
        clf = lgb.LGBMClassifier(
            boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
            max_depth=-1, n_estimators=args.n_estimators, objective='binary',
            subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
            learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1, device='gpu'
        )
    else:
        clf = lgb.LGBMClassifier(
            boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
            max_depth=-1, n_estimators=args.n_estimators, objective='binary',
            subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
            learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
        )
    clf.fit(train_x, train_y, eval_set=[(evals_x, evals_y)], eval_metric='auc', early_stopping_rounds=200)
    res['score'] = clf.predict_proba(test_x)[:, 1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    with tf.gfile.FastGFile(os.path.join(args.data_dir, "submission_{0}_{1}.csv".format(args.task, args.n_estimators)),
                            'w') as gf:
        res.to_csv(gf, index=False)
        # os.system('zip ../data/baseline.zip ../data/submission.csv')
    return clf


def LGB_only_predict(args, train_x, train_y, test_x, res, evals_x, evals_y, gpu=False):
    print("LGB test")
    # print("train_x:",train_x.columns.values)
    # print("test_x:",test_x.columns.values)
    # print("evals_x:",evals_x.columns.values)
    if gpu:
        clf = lgb.LGBMClassifier(
            boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
            max_depth=-1, n_estimators=args.n_estimators, objective='binary',
            subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
            learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1, device='gpu'
        )
    else:
        clf = lgb.LGBMClassifier(
            boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
            max_depth=-1, n_estimators=args.n_estimators, objective='binary',
            subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
            learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
        )
    clf.fit(train_x, train_y, eval_set=[(evals_x, evals_y)], eval_metric='auc', early_stopping_rounds=100)


def run(args):
    data_dir = args.data_dir
    if tf.gfile.Exists(os.path.join(data_dir, 'train_x.npz')):
        with tf.gfile.FastGFile(os.path.join(data_dir, 'train_x.npz'), 'rb') as f:
            contents = f.read()
            train_x = sparse.load_npz(io.BytesIO(contents))
        with tf.gfile.FastGFile(os.path.join(data_dir, 'test_x.npz'), 'rb') as f:
            contents = f.read()
            test_x = sparse.load_npz(io.BytesIO(contents))
        with tf.gfile.FastGFile(os.path.join(data_dir, 'train_y.csv'), 'r') as f:
            train_y = pd.read_csv(f)
        with tf.gfile.FastGFile(os.path.join(data_dir, 'res.csv'), 'r') as f:
            res = pd.read_csv(f)
        print("load previous npz")
        print("train_x:", train_x.shape)
        print("train_y:", train_y.shape)
        print("test_x:", test_x.shape)
        print("res:", res.shape)
    else:
        with tf.gfile.FastGFile(os.path.join(data_dir, 'adFeature.csv')) as gf:
            ad_feature = pd.read_csv(gf)
        if tf.gfile.Exists(os.path.join(data_dir, 'userFeature.csv')):
            with tf.gfile.FastGFile(os.path.join(data_dir, 'userFeature.csv'), 'r') as gf:
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

        train.loc[train['label'] == -1, 'label'] = 0
        predict['label'] = -1
        data = pd.concat([train, predict])
        data = pd.merge(data, ad_feature, on='aid', how='left')
        data = pd.merge(data, user_feature, on='uid', how='left')
        data = data.fillna('-1')
        one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                           'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                           'adCategoryId', 'productId', 'productType']
        vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4',
                          'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
        for feature in one_hot_feature:
            try:
                data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
            except:
                data[feature] = LabelEncoder().fit_transform(data[feature])

        # with tf.gfile.FastGFile(os.path.join(data_dir, 'all_data.csv'), 'w') as gf:
        #     data.to_csv(gf, index=False)

        train = data[data.label != -1]
        train_y = train.pop('label')
        print("train_y shape:", train_y.shape)
        # train, test, train_y, test_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
        test = data[data.label == -1]
        res = test[['aid', 'uid']]
        test = test.drop('label', axis=1)
        enc = OneHotEncoder()
        train_x = train[['creativeSize']]
        test_x = test[['creativeSize']]

        for feature in one_hot_feature:
            enc.fit(data[feature].values.reshape(-1, 1))
            train_a = enc.transform(train[feature].values.reshape(-1, 1))
            test_a = enc.transform(test[feature].values.reshape(-1, 1))
            train_x = sparse.hstack((train_x, train_a))
            test_x = sparse.hstack((test_x, test_a))
        print('one-hot prepared !')

        cv = CountVectorizer()
        for feature in vector_feature:
            cv.fit(data[feature])
            train_a = cv.transform(train[feature])
            test_a = cv.transform(test[feature])
            train_x = sparse.hstack((train_x, train_a))
            test_x = sparse.hstack((test_x, test_a))
        print('cv prepared !')
        print("train_x shape:", train_x.shape)
        dump_path = os.path.join(data_dir, 'train_x.npz')
        with tf.gfile.FastGFile(dump_path, 'w') as gf:
            sparse.save_npz(train_x, gf)
        dump_path = os.path.join(data_dir, 'test_x.npz')
        with tf.gfile.FastGFile(dump_path, 'w') as gf:
            sparse.save_npz(test_x, gf)
        dump_path = os.path.join(data_dir, 'train_y.csv')
        with tf.gfile.FastGFile(dump_path, 'w') as gf:
            train_y.to_csv(gf, index=False)
        dump_path = os.path.join(data_dir, 'res.csv')
        with tf.gfile.FastGFile(dump_path, 'w') as gf:
            res.to_csv(gf, index=False)

    train_x, evals_x, train_y, evals_y = train_test_split(train_x, train_y, test_size=0.2, random_state=2018,
                                                          stratify=train_y)
    LGB_predict(args, train_x, train_y, test_x, res, evals_x, evals_y)


def run_submit(args):
    data_dir = args.data_dir
    with tf.gfile.FastGFile(os.path.join(data_dir, 'adFeature.csv')) as gf:
        ad_feature = pd.read_csv(gf)
    if tf.gfile.Exists(os.path.join(data_dir, 'userFeature.csv')):
        with tf.gfile.FastGFile(os.path.join(data_dir, 'userFeature.csv'), 'r') as gf:
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

    train.loc[train['label'] == -1, 'label'] = 0
    predict['label'] = -1
    data = pd.concat([train, predict])
    data = pd.merge(data, ad_feature, on='aid', how='left')
    data = pd.merge(data, user_feature, on='uid', how='left')
    data = data.fillna('-1')

    data['aid_age'] = data[['aid', 'age']].apply(lambda x: '_'.join(x.apply(str)), axis=1)
    data['aid_consumptionAbility'] = data[['aid', 'consumptionAbility']].apply(lambda x: '_'.join(x.apply(str)), axis=1)
    data['aid_gender'] = data[['aid', 'gender']].apply(lambda x: '_'.join(x.apply(str)), axis=1)
    data['aid_education'] = data[['aid', 'education']].apply(lambda x: '_'.join(x.apply(str)), axis=1)

    one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                       'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                       'adCategoryId', 'productId', 'productType', 'aid_age', 'aid_consumptionAbility', 'aid_gender',
                       'aid_education']
    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                      'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
    for feature in one_hot_feature:
        try:
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])

    # with tf.gfile.FastGFile(os.path.join(data_dir, 'all_data.csv'), 'w') as gf:
    #     data.to_csv(gf, index=False)

    train = data[data.label != -1]
    train_y = train.pop('label')
    print("train_y shape:", train_y.shape)
    # train, test, train_y, test_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
    test = data[data.label == -1]
    res = test[['aid', 'uid']]
    test = test.drop('label', axis=1)
    enc = OneHotEncoder()
    train_x = train[['creativeSize']]
    test_x = test[['creativeSize']]

    for feature in one_hot_feature:
        enc.fit(data[feature].values.reshape(-1, 1))
        train_a = enc.transform(train[feature].values.reshape(-1, 1))
        test_a = enc.transform(test[feature].values.reshape(-1, 1))
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
    print('one-hot prepared !')

    cv = CountVectorizer()
    for feature in vector_feature:
        cv.fit(data[feature])
        train_a = cv.transform(train[feature])
        test_a = cv.transform(test[feature])
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
    print('cv prepared !')
    print("train_x shape:", train_x.shape)

    train_x, evals_x, train_y, evals_y = train_test_split(train_x, train_y, test_size=0.2, random_state=20180515,
                                                          stratify=train_y)
    LGB_predict(args, train_x, train_y, test_x, res, evals_x, evals_y)


def run_target_encoding(args):
    data_dir = args.data_dir
    with tf.gfile.FastGFile(os.path.join(data_dir, 'adFeature.csv')) as gf:
        ad_feature = pd.read_csv(gf)
    if tf.gfile.Exists(os.path.join(data_dir, 'userFeature.csv')):
        with tf.gfile.FastGFile(os.path.join(data_dir, 'userFeature.csv'), 'r') as gf:
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

    train.loc[train['label'] == -1, 'label'] = 0
    predict['label'] = -1
    data = pd.concat([train, predict])
    data = pd.merge(data, ad_feature, on='aid', how='left')
    data = pd.merge(data, user_feature, on='uid', how='left')
    data = data.fillna('-1')
    one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                       'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                       'adCategoryId', 'productId', 'productType']
    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                      'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']

    for feature in one_hot_feature:
        try:
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])

    # with tf.gfile.FastGFile(os.path.join(data_dir, 'all_data.csv'), 'w') as gf:
    #     data.to_csv(gf, index=False)

    train = data[data.label != -1]
    train_y = train.pop('label')
    print("train_y shape:", train_y.shape)
    # train, test, train_y, test_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
    test = data[data.label == -1]
    res = test[['aid', 'uid']]
    test = test.drop('label', axis=1)
    train_x = train[['creativeSize']]
    test_x = test[['creativeSize']]

    # for feature in one_hot_feature:
    #     enc.fit(data[feature].values.reshape(-1, 1))
    #     train_a=enc.transform(train[feature].values.reshape(-1, 1))
    #     test_a = enc.transform(test[feature].values.reshape(-1, 1))
    #     train_x= sparse.hstack((train_x, train_a))
    #     test_x = sparse.hstack((test_x, test_a))
    target_encoder = TargetEncoder(cols=one_hot_feature)
    target_encoder.fit(train[one_hot_feature], train_y)
    train_a = target_encoder.transform(train[one_hot_feature])
    test_a = target_encoder.transform(test[one_hot_feature])
    train_x = pd.concat([train_x, train_a], axis=1)
    test_x = pd.concat([test_x, test_a], axis=1)
    # train_x= sparse.hstack((train_x, train_a))
    # test_x = sparse.hstack((test_x, test_a))
    print('mean encoding done!')

    cv = CountVectorizer()
    for feature in vector_feature:
        cv.fit(data[feature].apply(str))
        train_a = cv.transform(train[feature].apply(str))
        test_a = cv.transform(test[feature].apply(str))
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
    print('cv prepared !')

    train_x, evals_x, train_y, evals_y = train_test_split(train_x, train_y, test_size=0.2, random_state=2018,
                                                          stratify=train_y)
    LGB_predict(args, train_x, train_y, test_x, res, evals_x, evals_y)


def run_cv(args):
    data_dir = args.data_dir
    if tf.gfile.Exists(os.path.join(data_dir, 'train_x.npz')):
        with tf.gfile.FastGFile(os.path.join(data_dir, 'train_x.npz'), 'rb') as f:
            train_x = sparse.load_npz(f)
        with tf.gfile.FastGFile(os.path.join(data_dir, 'test_x.npz'), 'rb') as f:
            test_x = sparse.load_npz(f)
        with tf.gfile.FastGFile(os.path.join(data_dir, 'train_y.csv'), 'r') as f:
            train_y = pd.read_csv(f)
        with tf.gfile.FastGFile(os.path.join(data_dir, 'res.csv'), 'r') as f:
            res = pd.read_csv(f)
    else:
        with tf.gfile.FastGFile(os.path.join(data_dir, 'adFeature.csv')) as gf:
            ad_feature = pd.read_csv(gf)
        if tf.gfile.Exists(os.path.join(data_dir, 'userFeature.csv')):
            with tf.gfile.FastGFile(os.path.join(data_dir, 'userFeature.csv'), 'r') as gf:
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

        train.loc[train['label'] == -1, 'label'] = 0
        predict['label'] = -1
        data = pd.concat([train, predict])
        data = pd.merge(data, ad_feature, on='aid', how='left')
        data = pd.merge(data, user_feature, on='uid', how='left')
        data = data.fillna('-1')
        one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                           'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                           'adCategoryId', 'productId', 'productType']
        vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4',
                          'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
        for feature in one_hot_feature:
            try:
                data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
            except:
                data[feature] = LabelEncoder().fit_transform(data[feature])

        # with tf.gfile.FastGFile(os.path.join(data_dir, 'all_data.csv'), 'w') as gf:
        #     data.to_csv(gf, index=False)

        train = data[data.label != -1]
        train_y = train.pop('label')
        print("train_y shape:", train_y.shape)
        # train, test, train_y, test_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
        test = data[data.label == -1]
        res = test[['aid', 'uid']]
        test = test.drop('label', axis=1)
        enc = OneHotEncoder()
        train_x = train[['creativeSize']]
        test_x = test[['creativeSize']]

        for feature in one_hot_feature:
            enc.fit(data[feature].values.reshape(-1, 1))
            train_a = enc.transform(train[feature].values.reshape(-1, 1))
            test_a = enc.transform(test[feature].values.reshape(-1, 1))
            train_x = sparse.hstack((train_x, train_a))
            test_x = sparse.hstack((test_x, test_a))
        print('one-hot prepared !')

        cv = CountVectorizer()
        for feature in vector_feature:
            cv.fit(data[feature])
            train_a = cv.transform(train[feature])
            test_a = cv.transform(test[feature])
            train_x = sparse.hstack((train_x, train_a))
            test_x = sparse.hstack((test_x, test_a))
        print('cv prepared !')
        print("train_x shape:", train_x.shape)
        dump_path = os.path.join(data_dir, 'train_x.npz')
        with tf.gfile.FastGFile(dump_path, 'wb') as gf:
            output = io.BytesIO()
            sparse.save_npz(output, train_x)
            gf.write(output.getvalue())
        dump_path = os.path.join(data_dir, 'test_x.npz')
        with tf.gfile.FastGFile(dump_path, 'wb') as gf:
            output = io.BytesIO()
            sparse.save_npz(output, test_x)
            gf.write(output.getvalue())
        dump_path = os.path.join(data_dir, 'train_y.csv')
        with tf.gfile.FastGFile(dump_path, 'w') as gf:
            train_y.to_csv(gf, index=False)
        dump_path = os.path.join(data_dir, 'res.csv')
        with tf.gfile.FastGFile(dump_path, 'w') as gf:
            res.to_csv(gf, index=False)

    # train_x, evals_x, train_y, evals_y = train_test_split(train_x, train_y, test_size=0.2, random_state=2018,
    #                                                   stratify=train_y)
    opt = bayessearchcv(train_x, train_y)
    print("bayes tunning result:")
    print(opt)
    # opt.fit(X_train, y_train)
    #
    # train_x, evals_x, train_y, evals_y = train_test_split(train_x, train_y, test_size=0.2, random_state=2018,
    #                                                   stratify=train_y)
    # # LGB_predict(data_dir, train_x, train_y, test_x, res, evals_x, evals_y)
    # print("val. score: %s" % opt.best_score_)
    # print("test score: %s" % opt.score(X_test, y_test))


def split_data(args):
    print("split data")
    data_dir = args.data_dir
    with tf.gfile.FastGFile(os.path.join(data_dir, 'adFeature.csv')) as gf:
        ad_feature = pd.read_csv(gf)
    if tf.gfile.Exists(os.path.join(data_dir, 'userFeature.csv')):
        with tf.gfile.FastGFile(os.path.join(data_dir, 'userFeature.csv'), 'r') as gf:
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

    train.loc[train['label'] == -1, 'label'] = 0
    predict['label'] = -1
    data = pd.concat([train, predict])
    data = pd.merge(data, ad_feature, on='aid', how='left')
    data = pd.merge(data, user_feature, on='uid', how='left')
    data = data.fillna('-1')
    train = data[data['label'] != -1]
    test = data[data['label'] == -1]
    cnt = 100
    size = math.ceil(len(train) / cnt)
    for i in tqdm(range(cnt)):
        start = size * i
        end = (i + 1) * size if (i + 1) * size < len(train) else len(train)
        slice = train[start:end]
        result = pd.concat([slice, test])
        with tf.gfile.FastGFile(
                os.path.join(os.path.join(data_dir, 'cache'), "lookalike_{0}_split_{1}.csv".format(cnt, i)), 'w') as gf:
            result.to_csv(gf, index=False)

        del result
        gc.collect()


def data2csv(args):
    print("split data")
    data_dir = args.data_dir
    with tf.gfile.FastGFile(os.path.join(data_dir, 'adFeature.csv')) as gf:
        ad_feature = pd.read_csv(gf)
    if tf.gfile.Exists(os.path.join(data_dir, 'userFeature.csv')):
        with tf.gfile.FastGFile(os.path.join(data_dir, 'userFeature.csv'), 'r') as gf:
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

    train.loc[train['label'] == -1, 'label'] = 0
    predict['label'] = -1
    data = pd.concat([train, predict])
    data = pd.merge(data, ad_feature, on='aid', how='left')
    data = pd.merge(data, user_feature, on='uid', how='left')
    data = data.fillna('-1')
    # train = data[data['label'] != -1]
    # test = data[data['label'] == -1]

    with tf.gfile.FastGFile(os.path.join(os.path.join(data_dir, 'cache'), "lookalike_data_all.csv"), 'w') as gf:
        data.to_csv(gf, index=False)

    del data
    gc.collect()


def small_test(args):
    data_dir = args.data_dir
    # data_dir = "E:/dataset/tencent_lookalike/preliminary_contest_data"
    with tf.gfile.FastGFile(os.path.join(data_dir, 'lookalike_split_0.csv'), 'r') as gf:
        data = pd.read_csv(gf)

    one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                       'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                       'adCategoryId', 'productId', 'productType']
    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                      'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']

    for feature in one_hot_feature:
        try:
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])

    train = data[data.label != -1]
    train_y = train.pop('label')
    print("train_y shape:", train_y.shape)
    # train, test, train_y, test_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
    test = data[data.label == -1]
    res = test[['aid', 'uid']]
    test = test.drop('label', axis=1)
    enc = OneHotEncoder()
    train_x = train[['creativeSize']]
    test_x = test[['creativeSize']]

    target_encoder = TargetEncoder(cols=one_hot_feature)
    target_encoder.fit(train[one_hot_feature], train_y)
    train_a = target_encoder.transform(train[one_hot_feature])
    test_a = target_encoder.transform(test[one_hot_feature])
    train_x = pd.concat([train_x, train_a], axis=1)
    test_x = pd.concat([test_x, test_a], axis=1)
    print("mean encoding done")

    cv = CountVectorizer()
    for feature in vector_feature:
        cv.fit(data[feature].apply(str))
        train_a = cv.transform(train[feature].apply(str))
        test_a = cv.transform(test[feature].apply(str))
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))

    print("cv done")
    opt = bayessearchcv(train_x, train_y, args)
    print("bayes tunning result:")
    print(opt)


def small_test_cross_feature(args):
    data_dir = args.data_dir
    # data_dir = "E:/dataset/tencent_lookalike/preliminary_contest_data"
    with tf.gfile.FastGFile(os.path.join(data_dir, 'lookalike_100_split_0.csv'), 'r') as gf:
        data = pd.read_csv(gf)

    train = data[data.label != -1]
    data_clicked = train[train['label'] == 1]
    #    aid_age_ratio = (data_clicked.groupby(['aid', 'age']).size() / data_clicked.shape[0]).reset_index()
    #    aid_age_ratio.rename(columns={0: 'aid_age_ratio'}, inplace=True)
    #    aid_gender_ratio = (data_clicked.groupby(['aid', 'gender']).size() / data_clicked.shape[0]).reset_index()
    #    aid_gender_ratio.rename(columns={0: 'aid_gender_ratio'}, inplace=True)

    data['aid_age'] = data[['aid', 'age']].apply(lambda x: '_'.join(x.apply(str)), axis=1)
    data['aid_consumptionAbility'] = data[['aid', 'consumptionAbility']].apply(lambda x: '_'.join(x.apply(str)), axis=1)
    data['aid_gender'] = data[['aid', 'gender']].apply(lambda x: '_'.join(x.apply(str)), axis=1)
    data['aid_education'] = data[['aid', 'education']].apply(lambda x: '_'.join(x.apply(str)), axis=1)

    # one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house','advertiserId', 'campaignId', 'creativeId',
    #                    'adCategoryId', 'productId', 'productType','aid_age','aid_consumptionAbility','aid_gender','aid_education',
    #                    'aid_interest1','aid_interest2','aid_interest3']

    one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                       'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                       'adCategoryId', 'productId', 'productType', 'aid_age', 'aid_consumptionAbility', 'aid_gender',
                       'aid_education']
    # one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
    #                    'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
    #                    'adCategoryId', 'productId', 'productType']
    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                      'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
    for feature in one_hot_feature:
        try:
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])

    # with tf.gfile.FastGFile(os.path.join(data_dir, 'all_data.csv'), 'w') as gf:
    #     data.to_csv(gf, index=False)

    #    data = pd.merge(data,aid_age_ratio,how='left',on=['aid','age'])
    #    data = pd.merge(data,aid_gender_ratio,how='left',on=['aid','gender'])
    train = data[data.label != -1]
    train_y = train.pop('label')
    print("train_y shape:", train_y.shape)
    # train, test, train_y, test_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
    test = data[data.label == -1]
    res = test[['aid', 'uid']]
    test = test.drop('label', axis=1)
    enc = OneHotEncoder()
    train_x = train[['creativeSize']]
    test_x = test[['creativeSize']]

    for feature in one_hot_feature:
        enc.fit(data[feature].values.reshape(-1, 1))
        train_a = enc.transform(train[feature].values.reshape(-1, 1))
        test_a = enc.transform(test[feature].values.reshape(-1, 1))
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
    print('one-hot prepared !')

    cv = CountVectorizer()
    for feature in vector_feature:
        cv.fit(data[feature].apply(str))
        train_a = cv.transform(train[feature].apply(str))
        test_a = cv.transform(test[feature].apply(str))
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))

    print('cv prepared !')
    print("train_x shape:", train_x.shape)

    train_x, evals_x, train_y, evals_y = train_test_split(train_x, train_y, test_size=0.2, random_state=2018,
                                                          stratify=train_y)
    LGB_only_predict(args, train_x, train_y, test_x, res, evals_x, evals_y, args.lgbgpu)


def data2ffm(args):
    data_dir = args.data_dir
    cache_dir = os.path.join(data_dir,'cache')
    with tf.gfile.FastGFile(os.path.join(cache_dir, 'lookalike_data_all.csv'), 'r') as gf:
        data = pd.read_csv(gf)

    one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                       'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                       'adCategoryId', 'productId', 'productType']
    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                      'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
    continus_feature = ['creativeSize']
    data = data.fillna(-1)

    for feature in one_hot_feature:
        try:
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])

    data = data[one_hot_feature + vector_feature + continus_feature]

    class FFMFormat:
        def __init__(self, vector_feat, one_hot_feat, continus_feat):
            self.field_index_ = None
            self.feature_index_ = None
            self.vector_feat = vector_feat
            self.one_hot_feat = one_hot_feat
            self.continus_feat = continus_feat

        def get_params(self):
            pass

        def set_params(self, **parameters):
            pass

        def fit(self, df, y=None):
            self.field_index_ = {col: i for i, col in enumerate(df.columns)}
            self.feature_index_ = dict()
            last_idx = 0
            for col in df.columns:
                if col in self.one_hot_feat:
                    print(col)
                    df[col] = df[col].astype('int')
                    vals = np.unique(df[col])
                    for val in vals:
                        if val == -1: continue
                        name = '{}_{}'.format(col, val)
                        if name not in self.feature_index_:
                            self.feature_index_[name] = last_idx
                            last_idx += 1
                elif col in self.vector_feat:
                    print(col)
                    vals = []
                    for data in df[col].apply(str):
                        if data != "-1":
                            for word in data.strip().split(' '):
                                vals.append(word)
                    vals = np.unique(vals)
                    for val in vals:
                        if val == "-1": continue
                        name = '{}_{}'.format(col, val)
                        if name not in self.feature_index_:
                            self.feature_index_[name] = last_idx
                            last_idx += 1
                self.feature_index_[col] = last_idx
                last_idx += 1
            return self

        def fit_transform(self, df, y=None):
            self.fit(df, y)
            return self.transform(df)

        def transform_row_(self, row):
            ffm = []

            for col, val in row.loc[row != 0].to_dict().items():
                if col in self.one_hot_feat:
                    name = '{}_{}'.format(col, val)
                    if name in self.feature_index_:
                        ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[name]))
                    # ffm.append('{}:{}:{}'.format(self.field_index_[col], self.feature_index_[col], 1))
                elif col in self.vector_feat:
                    for word in str(val).split(' '):
                        name = '{}_{}'.format(col, word)
                        if name in self.feature_index_:
                            ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[name]))
                elif col in self.continus_feat:
                    if val != -1:
                        ffm.append('{}:{}:{}'.format(self.field_index_[col], self.feature_index_[col], val))
            return ' '.join(ffm)

        def transform(self, df):
            # val=[]
            # for k,v in self.feature_index_.items():
            #     val.append(v)
            # val.sort()
            # print(val)
            # print(self.field_index_)
            # print(self.feature_index_)
            return pd.Series({idx: self.transform_row_(row) for idx, row in df.iterrows()})

    tr = FFMFormat(vector_feature, one_hot_feature, continus_feature)
    user_ffm = tr.fit_transform(data)
    with tf.gfile.FastGFile(os.path.join(data_dir, 'ffm.csv'), 'w') as gf:
        user_ffm.to_csv(gf, index=False)

    with tf.gfile.FastGFile(os.path.join(data_dir, 'train.csv'), 'r') as gf:
        train = pd.read_csv(gf)
    with tf.gfile.FastGFile(os.path.join(data_dir, 'test1.csv'), 'r') as gf:
        test = pd.read_csv(gf)

    Y = np.array(train.pop('label'))
    len_train = len(train)

    with tf.gfile.FastGFile(os.path.join(data_dir, 'ffm.csv'),'r') as fin:
        with tf.gfile.FastGFile(os.path.join(data_dir, 'train_ffm.csv'), 'w') as f_train_out:
            with tf.gfile.FastGFile(os.path.join(data_dir, 'test_ffm.csv'), 'w') as f_test_out:
                for (i, line) in enumerate(fin):
                    if i < len_train:
                        f_train_out.write(str(Y[i]) + ' ' + line)
                    else:
                        f_test_out.write(line)

    # path = 'data/'
    #leave for local
    # ffm_model = xl.create_ffm()
    # ffm_model.setTrain(os.path.join(data_dir, 'train_ffm.csv'))
    # ffm_model.setTest(os.path.join(data_dir, 'test_ffm.csv'))
    # ffm_model.setSigmoid()
    # param = {'task': 'binary', 'lr': 0.01, 'lambda': 0.001, 'metric': 'auc', 'opt': 'ftrl', 'epoch': 5, 'k': 4,
    #          'alpha': 1.5, 'beta': 0.01, 'lambda_1': 0.0, 'lambda_2': 0.0}
    # ffm_model.fit(param, os.path.join(data_dir,"model.out"))
    # ffm_model.predict(os.path.join(data_dir,"model.out"), os.path.join(data_dir,"output.txt"))
    # sub = pd.DataFrame()
    # sub['aid'] = test['aid']
    # sub['uid'] = test['uid']
    # sub['score'] = np.loadtxt(os.path.join(data_dir,"output.txt"))
    # sub.to_csv('submission.csv', index=False)
    # os.system('zip baseline_ffm.zip submission.csv')

def data2ffm2(args):
    data_dir = args.data_dir

    with tf.gfile.FastGFile(os.path.join(data_dir, 'train.csv'), 'r') as gf:
        train = pd.read_csv(gf)

    Y = np.array(train.pop('label'))
    len_train = len(train)

    with tf.gfile.FastGFile(os.path.join(data_dir, 'ffm.csv'),'r') as fin:
        with tf.gfile.FastGFile(os.path.join(data_dir, 'train_ffm.csv'), 'w') as f_train_out:
            with tf.gfile.FastGFile(os.path.join(data_dir, 'test_ffm.csv'), 'w') as f_test_out:
                for (i, line) in enumerate(fin):
                    if i < len_train:
                        f_train_out.write(str(Y[i]) + ' ' + line)
                    else:
                        f_test_out.write(line)

def cloud2local(cloud_dir,local_dir,fname):
    with tf.gfile.FastGFile(os.path.join(cloud_dir, fname), 'rb') as f:
        with tf.gfile.FastGFile(os.path.join(local_dir,fname),'wb') as outf:
            contents = f.read()
            outf.write(contents)
            # outf.write(io.BytesIO(contents))

def local2cloud(cloud_dir,local_dir,fname):
    with tf.gfile.FastGFile(os.path.join(local_dir,fname),'rb') as f:
        with tf.gfile.FastGFile(os.path.join(cloud_dir, fname), 'wb') as outf:
            contents = f.read()
            outf.write(contents)
            # outf.write(io.BytesIO(contents))

def localtrainffm(args):
    cwd = os.getcwd()
    # / data4 / data / nm - local - dir / usercache / mt_tenant_4390104 / appcache / application_1520240304221_3990 / container_e14_1520240304221_3990_01_000002
    print("cwd {0}".format(cwd))
    data_dir = args.data_dir
    cache_dir = os.path.join(data_dir,'cache')
    with tf.gfile.FastGFile(os.path.join(cache_dir, 'lookalike_data_all.csv'), 'r') as gf:
        data = pd.read_csv(gf)

    test = data[data.label == -1]

    cloud2local(data_dir,cwd, 'train_ffm.csv')
    cloud2local(data_dir,cwd, 'test_ffm.csv')

    ffm_model = xl.create_ffm()
    ffm_model.setTrain(os.path.join(cwd, 'train_ffm.csv'))
    ffm_model.setTest(os.path.join(cwd, 'test_ffm.csv'))
    ffm_model.setSigmoid()
    param = {'task': 'binary', 'lr': 0.01, 'lambda': 0.001, 'metric': 'auc', 'opt': 'ftrl', 'epoch': 5, 'k': 4,
             'alpha': 1.5, 'beta': 0.01, 'lambda_1': 0.0, 'lambda_2': 0.0}
    ffm_model.fit(param, os.path.join(cwd,"model.out"))
    ffm_model.predict(os.path.join(cwd,"model.out"), os.path.join(cwd,"output.txt"))
    sub = pd.DataFrame()
    sub['aid'] = test['aid']
    sub['uid'] = test['uid']
    sub['score'] = np.loadtxt(os.path.join(cwd,"output.txt"))
    sub.to_csv(os.path.join(cwd,'submission.csv'), index=False)
    # os.system('zip baseline_ffm.zip submission.csv')
    local2cloud(cloud_dir,local_dir,'submission.csv')

def big_test(args):
    data_dir = args.data_dir
    with tf.gfile.FastGFile(os.path.join(data_dir, 'adFeature.csv')) as gf:
        ad_feature = pd.read_csv(gf)
    if tf.gfile.Exists(os.path.join(data_dir, 'userFeature.csv')):
        with tf.gfile.FastGFile(os.path.join(data_dir, 'userFeature.csv'), 'r') as gf:
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

    train.loc[train['label'] == -1, 'label'] = 0
    predict['label'] = -1
    data = pd.concat([train, predict])
    data = pd.merge(data, ad_feature, on='aid', how='left')
    data = pd.merge(data, user_feature, on='uid', how='left')
    data = data.fillna('-1')

    one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                       'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                       'adCategoryId', 'productId', 'productType']
    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                      'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']

    for feature in one_hot_feature:
        try:
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])

    train = data[data.label != -1]
    train_y = train.pop('label')
    print("train_y shape:", train_y.shape)
    test = data[data.label == -1]
    res = test[['aid', 'uid']]
    test = test.drop('label', axis=1)
    enc = OneHotEncoder()
    train_x = train[['creativeSize']]
    test_x = test[['creativeSize']]

    target_encoder = TargetEncoder(cols=one_hot_feature)
    target_encoder.fit(train[one_hot_feature], train_y)
    train_a = target_encoder.transform(train[one_hot_feature])
    test_a = target_encoder.transform(test[one_hot_feature])
    train_x = pd.concat([train_x, train_a], axis=1)
    test_x = pd.concat([test_x, test_a], axis=1)
    print("mean encoding done")

    cv = CountVectorizer()
    for feature in vector_feature:
        cv.fit(data[feature].apply(str))
        train_a = cv.transform(train[feature].apply(str))
        test_a = cv.transform(test[feature].apply(str))
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))

    print("cv done")
    opt = bayessearchcv(train_x, train_y, args)
    print("bayes tunning result:")
    print(opt)


def big_submit(args):
    data_dir = args.data_dir
    with tf.gfile.FastGFile(os.path.join(data_dir, 'adFeature.csv')) as gf:
        ad_feature = pd.read_csv(gf)
    if tf.gfile.Exists(os.path.join(data_dir, 'userFeature.csv')):
        with tf.gfile.FastGFile(os.path.join(data_dir, 'userFeature.csv'), 'r') as gf:
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

    train.loc[train['label'] == -1, 'label'] = 0
    predict['label'] = -1
    data = pd.concat([train, predict])
    data = pd.merge(data, ad_feature, on='aid', how='left')
    data = pd.merge(data, user_feature, on='uid', how='left')
    data = data.fillna('-1')

    one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                       'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                       'adCategoryId', 'productId', 'productType']
    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                      'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']

    for feature in one_hot_feature:
        try:
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])

    train = data[data.label != -1]
    train_y = train.pop('label')
    print("train_y shape:", train_y.shape)
    test = data[data.label == -1]
    res = test[['aid', 'uid']]
    test = test.drop('label', axis=1)
    train_x = train[['creativeSize']]
    test_x = test[['creativeSize']]

    target_encoder = TargetEncoder(cols=one_hot_feature)
    target_encoder.fit(train[one_hot_feature], train_y)
    train_a = target_encoder.transform(train[one_hot_feature])
    test_a = target_encoder.transform(test[one_hot_feature])
    train_x = pd.concat([train_x, train_a], axis=1)
    test_x = pd.concat([test_x, test_a], axis=1)
    print("mean encoding done")

    cv = CountVectorizer()
    for feature in vector_feature:
        cv.fit(data[feature].apply(str))
        train_a = cv.transform(train[feature].apply(str))
        test_a = cv.transform(test[feature].apply(str))
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))

    train_x, evals_x, train_y, evals_y = train_test_split(train_x, train_y, test_size=0.2, random_state=2018,
                                                          stratify=train_y)
    LGB_predict(args, train_x, train_y, test_x, res, evals_x, evals_y)


def dump_svm(args):
    data_dir = args.data_dir
    with tf.gfile.FastGFile(os.path.join(data_dir, 'adFeature.csv')) as gf:
        ad_feature = pd.read_csv(gf)
    if tf.gfile.Exists(os.path.join(data_dir, 'userFeature.csv')):
        with tf.gfile.FastGFile(os.path.join(data_dir, 'userFeature.csv'), 'r') as gf:
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

    train.loc[train['label'] == -1, 'label'] = 0
    predict['label'] = -1
    data = pd.concat([train, predict])
    data = pd.merge(data, ad_feature, on='aid', how='left')
    data = pd.merge(data, user_feature, on='uid', how='left')
    data = data.fillna('-1')

    one_hot_feature = ['aid', 'LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                       'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                       'adCategoryId', 'productId', 'productType']
    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                      'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']

    for feature in one_hot_feature:
        try:
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])

        # with tf.gfile.FastGFile(os.path.join(data_dir, 'all_data.csv'), 'w') as gf:
        #     data.to_csv(gf, index=False)

    train = data[data.label != -1]
    train_y = train.pop('label')
    print("train_y shape:", train_y.shape)
    # train, test, train_y, test_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
    test = data[data.label == -1]
    res = test[['aid', 'uid']]
    test = test.drop('label', axis=1)
    enc = OneHotEncoder()
    train_x = train[['creativeSize']]
    test_x = test[['creativeSize']]

    for feature in one_hot_feature:
        enc.fit(data[feature].values.reshape(-1, 1))
        train_a = enc.transform(train[feature].values.reshape(-1, 1))
        test_a = enc.transform(test[feature].values.reshape(-1, 1))
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
    print('one-hot prepared !')

    cv = CountVectorizer()
    for feature in vector_feature:
        cv.fit(data[feature].apply(str))
        train_a = cv.transform(train[feature].apply(str))
        test_a = cv.transform(test[feature].apply(str))
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))

    print('cv prepared !')

    with tf.gfile.FastGFile(os.path.join(data_dir, 'train.libsvm'), 'w') as gf:
        dump_svmlight_file(train_x, train_y, gf, zero_based=False)


def dump_split_svm(args):
    data_dir = args.data_dir
    with tf.gfile.FastGFile(os.path.join(data_dir, 'adFeature.csv')) as gf:
        ad_feature = pd.read_csv(gf)
    if tf.gfile.Exists(os.path.join(data_dir, 'userFeature.csv')):
        with tf.gfile.FastGFile(os.path.join(data_dir, 'userFeature.csv'), 'r') as gf:
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

    train.loc[train['label'] == -1, 'label'] = 0
    predict['label'] = -1
    data = pd.concat([train, predict])
    data = pd.merge(data, ad_feature, on='aid', how='left')
    data = pd.merge(data, user_feature, on='uid', how='left')
    data = data.fillna('-1')

    one_hot_feature = ['aid', 'LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                       'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                       'adCategoryId', 'productId', 'productType']
    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                      'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']

    for feature in one_hot_feature:
        try:
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])

        # with tf.gfile.FastGFile(os.path.join(data_dir, 'all_data.csv'), 'w') as gf:
        #     data.to_csv(gf, index=False)

    train = data[data.label != -1]
    train_y = train.pop('label')
    print("train_y shape:", train_y.shape)
    # train, test, train_y, test_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
    test = data[data.label == -1]
    res = test[['aid', 'uid']]
    test = test.drop('label', axis=1)
    enc = OneHotEncoder()
    train_x = train[['creativeSize']]
    test_x = test[['creativeSize']]

    for feature in one_hot_feature:
        enc.fit(data[feature].values.reshape(-1, 1))
        train_a = enc.transform(train[feature].values.reshape(-1, 1))
        test_a = enc.transform(test[feature].values.reshape(-1, 1))
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))
    print('one-hot prepared !')

    cv = CountVectorizer()
    for feature in vector_feature:
        cv.fit(data[feature].apply(str))
        train_a = cv.transform(train[feature].apply(str))
        test_a = cv.transform(test[feature].apply(str))
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))

    print('cv prepared !')

    cnt = 10
    size = math.ceil(train_x.shape[0] / cnt)
    for i in tqdm(range(cnt)):
        start = size * i
        end = (i + 1) * size if (i + 1) * size < train_x.shape[0] else train_x.shape[0]
        slice = train_x.tocsr()[start:end]
        result = sparse.vstack((slice, test_x.tocsr()))
        result_y = train_y[start:end]
        with tf.gfile.FastGFile(os.path.join(data_dir, 'train_split_%s.libsvm' % i), 'w') as gf:
            dump_svmlight_file(result, result_y, gf, zero_based=False)

    # with tf.gfile.FastGFile(os.path.join(data_dir, 'test.libsvm' % i), 'w') as gf:
    #     dump_svmlight_file(test_x, None, gf, zero_based=False)


def mean_encoding_tfidf_submit(args):
    data_dir = args.data_dir
    with tf.gfile.FastGFile(os.path.join(data_dir, 'adFeature.csv')) as gf:
        ad_feature = pd.read_csv(gf)
    if tf.gfile.Exists(os.path.join(data_dir, 'userFeature.csv')):
        with tf.gfile.FastGFile(os.path.join(data_dir, 'userFeature.csv'), 'r') as gf:
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

    train.loc[train['label'] == -1, 'label'] = 0
    predict['label'] = -1
    data = pd.concat([train, predict])
    data = pd.merge(data, ad_feature, on='aid', how='left')
    data = pd.merge(data, user_feature, on='uid', how='left')
    data = data.fillna('-1')

    one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct',
                       'marriageStatus', 'advertiserId', 'campaignId', 'creativeId',
                       'adCategoryId', 'productId', 'productType']
    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                      'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']

    for feature in one_hot_feature:
        try:
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])

    train = data[data.label != -1]
    train_y = train.pop('label')
    print("train_y shape:", train_y.shape)
    test = data[data.label == -1]
    res = test[['aid', 'uid']]
    test = test.drop('label', axis=1)
    train_x = train[['creativeSize']]
    test_x = test[['creativeSize']]

    target_encoder = TargetEncoder(cols=one_hot_feature)
    target_encoder.fit(train[one_hot_feature], train_y)
    train_a = target_encoder.transform(train[one_hot_feature])
    test_a = target_encoder.transform(test[one_hot_feature])
    train_x = pd.concat([train_x, train_a], axis=1)
    test_x = pd.concat([test_x, test_a], axis=1)
    print("mean encoding done")

    cv = CountVectorizer()
    for feature in vector_feature:
        cv.fit(data[feature].apply(str))
        train_a = cv.transform(train[feature].apply(str))
        test_a = cv.transform(test[feature].apply(str))
        # transformer = TfidfTransformer()
        # train_a = transformer.fit_transform(train_a)
        # test_a = transformer.transform(test_a)
        train_x = sparse.hstack((train_x, train_a))
        test_x = sparse.hstack((test_x, test_a))

    train_x, evals_x, train_y, evals_y = train_test_split(train_x, train_y, test_size=0.2, random_state=2018,
                                                          stratify=train_y)
    LGB_predict(args, train_x, train_y, test_x, res, evals_x, evals_y)


def main(_):
    args_in = sys.argv[1:]
    print(args_in)
    parser = argparse.ArgumentParser()
    mtyunArgs = parser.add_argument_group('cloud option')
    mtyunArgs.add_argument('--data_dir', type=str, default='',
                           help='input data path')
    mtyunArgs.add_argument('--cache_dir', type=str, default='',
                           help='cache data path')
    mtyunArgs.add_argument('--task', type=str, default='',
                           help='task type')
    mtyunArgs.add_argument('--model_dir', type=str, default='',
                           help='output model path')
    mtyunArgs.add_argument('--model_type', type=str, default='lgb',
                           help='model type str')
    mtyunArgs.add_argument('--tf_fs', type=str, default='', help='output model path')
    mtyunArgs.add_argument('--tf_prefix', type=str, default='', help='output model path')
    mtyunArgs.add_argument('--default_fs', type=str, default='', help='output model path')
    mtyunArgs.add_argument('--worker_num', type=str, default='', help='output model path')
    mtyunArgs.add_argument('--num_gpus', type=str, default='', help='output model path')
    mtyunArgs.add_argument('--num_ps', type=str, default='', help='output model path')
    mtyunArgs.add_argument('--num_worker', type=str, default='', help='output model path')
    mtyunArgs.add_argument('--tensorboard_dir', type=str, default='', help='output model path')
    mtyunArgs.add_argument('--tb_dir', type=str, default='local_tensorbord_dir_0', help='output model path')
    mtyunArgs.add_argument('--n_estimators', type=int, default=5000, help='gbdt n_estimators')
    mtyunArgs.add_argument('--lgbgpu', type=bool, default=False, help='gbdt use gpu')
    FLAGS, _ = parser.parse_known_args()
    print('FLAGS')
    print(FLAGS)
    args = parser.parse_args(args_in)
    if args.task == 'split':
        split_data(args)
    elif args.task == 'mean_encoding':
        run_target_encoding(args)
    elif args.task == 'bayescv':
        run_cv(args)
    elif args.task == 'small':
        small_test(args)
    elif args.task == 'big':
        big_test(args)
    elif args.task == 'bigsubmit':
        big_submit(args)
    elif args.task == 'feature':
        small_test_cross_feature(args)
    elif args.task == 'submit':
        run_submit(args)
    elif args.task == 'libsvm':
        dump_svm(args)
    elif args.task == 'splitsvm':
        dump_split_svm(args)
    elif args.task == 'tocsv':
        data2csv(args)
    elif args.task == 'toffm':
        data2ffm2(args)
    elif args.task == 'trainffm':
        localtrainffm(args)

if __name__ == '__main__':
    tf.app.run(main=main)

