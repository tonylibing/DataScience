import sys
sys.path.append("../..")
import feature.processor
from importlib import reload

reload(feature.processor)
from feature.processor import *

import hashlib
import json
import os
import shutil
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import jieba.analyse
from sklearn.model_selection import ParameterSampler

from spotlight.cross_validation import user_based_train_test_split
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.sequence.representations import CNNNet
from spotlight.evaluation import sequence_mrr_score
from spotlight.interactions import Interactions
from spotlight.evaluation import precision_recall_score
# CUDA = (os.environ.get('CUDA') is not None or
#         shutil.which('nvidia-smi') is not None)
CUDA=True
NUM_SAMPLES = 100

LEARNING_RATES = [1e-3, 1e-2, 5 * 1e-2, 1e-1]
LOSSES = ['bpr', 'hinge', 'adaptive_hinge', 'pointwise']
BATCH_SIZE = [8, 16, 32, 256]
EMBEDDING_DIM = [8, 16, 32, 64, 128, 256]
N_ITER = list(range(5, 20))
L2 = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.0]


class Results:

    def __init__(self, filename):

        self._filename = filename

        open(self._filename, 'a+')

    def _hash(self, x):

        return hashlib.md5(json.dumps(x, sort_keys=True).encode('utf-8')).hexdigest()

    def save(self, hyperparams, test_mrr, validation_mrr):

        result = {'test_mrr': test_mrr,
                  'validation_mrr': validation_mrr,
                  'hash': self._hash(hyperparams)}
        result.update(hyperparams)

        with tf.gfile.FastGFile(self._filename, 'a+') as out:
            out.write(json.dumps(result) + '\n')

    def best(self):

        results = sorted([x for x in self],
                         key=lambda x: -x['test_mrr'])

        if results:
            return results[0]
        else:
            return None

    def __getitem__(self, hyperparams):

        params_hash = self._hash(hyperparams)

        with tf.gfile.FastGFile(self._filename, 'r+') as fle:
            for line in fle:
                datum = json.loads(line)

                if datum['hash'] == params_hash:
                    del datum['hash']
                    return datum

        raise KeyError

    def __contains__(self, x):

        try:
            self[x]
            return True
        except KeyError:
            return False

    def __iter__(self):
        with tf.gfile.FastGFile(self._filename, 'r+') as fle:
            for line in fle:
                datum = json.loads(line)

                del datum['hash']

                yield datum


def sample_cnn_hyperparameters(random_state, num):

    space = {
        'n_iter': N_ITER,
        'batch_size': BATCH_SIZE,
        'l2': L2,
        'learning_rate': LEARNING_RATES,
        'loss': LOSSES,
        'embedding_dim': EMBEDDING_DIM,
        'kernel_width': [3, 5, 7],
        'num_layers': list(range(1, 10)),
        'dilation_multiplier': [1, 2],
        'nonlinearity': ['tanh', 'relu'],
        'residual': [True, False]
    }

    sampler = ParameterSampler(space,
                               n_iter=num,
                               random_state=random_state)

    for params in sampler:
        params['dilation'] = list(params['dilation_multiplier'] ** (i % 8)
                                  for i in range(params['num_layers']))

        yield params


def sample_lstm_hyperparameters(random_state, num):

    space = {
        'n_iter': N_ITER,
        'batch_size': BATCH_SIZE,
        'l2': L2,
        'learning_rate': LEARNING_RATES,
        'loss': LOSSES,
        'embedding_dim': EMBEDDING_DIM,
    }

    sampler = ParameterSampler(space,
                               n_iter=num,
                               random_state=random_state)

    for params in sampler:

        yield params


def sample_pooling_hyperparameters(random_state, num):

    space = {
        'n_iter': N_ITER,
        'batch_size': BATCH_SIZE,
        'l2': L2,
        'learning_rate': LEARNING_RATES,
        'loss': LOSSES,
        'embedding_dim': EMBEDDING_DIM,
    }

    sampler = ParameterSampler(space,
                               n_iter=num,
                               random_state=random_state)

    for params in sampler:

        yield params


def evaluate_cnn_model(hyperparameters, train, test, validation, random_state):

    h = hyperparameters

    net = CNNNet(train.num_items,
                 embedding_dim=h['embedding_dim'],
                 kernel_width=h['kernel_width'],
                 dilation=h['dilation'],
                 num_layers=h['num_layers'],
                 nonlinearity=h['nonlinearity'],
                 residual_connections=h['residual'])

    model = ImplicitSequenceModel(loss=h['loss'],
                                  representation=net,
                                  batch_size=h['batch_size'],
                                  learning_rate=h['learning_rate'],
                                  l2=h['l2'],
                                  n_iter=h['n_iter'],
                                  use_cuda=CUDA,
                                  random_state=random_state)

    model.fit(train, verbose=True)

    test_mrr = sequence_mrr_score(model, test)
    val_mrr = sequence_mrr_score(model, validation)

    return model,test_mrr,val_mrr


def evaluate_lstm_model(hyperparameters, train, test, validation, random_state):

    h = hyperparameters

    model = ImplicitSequenceModel(loss=h['loss'],
                                  representation='lstm',
                                  batch_size=h['batch_size'],
                                  learning_rate=h['learning_rate'],
                                  l2=h['l2'],
                                  n_iter=h['n_iter'],
                                  use_cuda=CUDA,
                                  random_state=random_state)

    model.fit(train, verbose=True)

    test_mrr = sequence_mrr_score(model, test)
    val_mrr = sequence_mrr_score(model, validation)

    return model,test_mrr, val_mrr


def evaluate_pooling_model(hyperparameters, train, test, validation, random_state):

    h = hyperparameters

    model = ImplicitSequenceModel(loss=h['loss'],
                                  representation='pooling',
                                  batch_size=h['batch_size'],
                                  learning_rate=h['learning_rate'],
                                  l2=h['l2'],
                                  n_iter=h['n_iter'],
                                  use_cuda=CUDA,
                                  random_state=random_state)

    model.fit(train, verbose=True)

    test_mrr = sequence_mrr_score(model, test)
    val_mrr = sequence_mrr_score(model, validation)

    return model,test_mrr, val_mrr


# precision, recall = precision_recall_score(model, test, train, k=k)

def run1(news,data,idx2newsid,idx2userid,train, test, validation, random_state, model_type):

    results = Results('{}_results.txt'.format(model_type))

    best_result = results.best()

    if model_type == 'pooling':
        eval_fnc, sample_fnc = (evaluate_pooling_model,
                                sample_pooling_hyperparameters)
    elif model_type == 'cnn':
        eval_fnc, sample_fnc = (evaluate_cnn_model,
                                sample_cnn_hyperparameters)
    elif model_type == 'lstm':
        eval_fnc, sample_fnc = (evaluate_lstm_model,
                                sample_lstm_hyperparameters)
    else:
        raise ValueError('Unknown model type')

    if best_result is not None:
        print('Best {} result: {}'.format(model_type, results.best()))

    for hyperparameters in sample_fnc(random_state, NUM_SAMPLES):

        if hyperparameters in results:
            continue

        print('Evaluating {}'.format(hyperparameters))

        (model,test_mrr, val_mrr) = eval_fnc(hyperparameters,
                                       train,
                                       test,
                                       validation,
                                       random_state)

        print('Test MRR {} val MRR {}'.format(
            test_mrr.mean(), val_mrr.mean()
        ))

        predictions = model.predict(1)
        print("len predictions:{}".format(len(predictions)))
        predictions = predictions.argsort()
        print("predictions:{}".format(predictions[:3]))

        data['rec_flag'] = data['news_id'].apply(lambda x: x in np.vectorize(idx2newsid.get)(predictions[:3]))
        k_res = [news[idx2newsid[idx]] for idx in predictions[:3]]
        user_id = idx2userid[1]
        print(data[data['user_id'] == user_id][['news_id', 'title']])

        print("predictions:{}".format(k_res))


        results.save(hyperparameters, test_mrr.mean(), val_mrr.mean())

    return results


def run2(news,data,idx2newsid,idx2userid,train, test, validation, random_state, model_type):
    results = Results('{}_results.txt'.format(model_type))

    best_result = results.best()

    if model_type == 'pooling':
        eval_fnc, sample_fnc = (evaluate_pooling_model,
                                sample_pooling_hyperparameters)
    elif model_type == 'cnn':
        eval_fnc, sample_fnc = (evaluate_cnn_model,
                                sample_cnn_hyperparameters)
    elif model_type == 'lstm':
        eval_fnc, sample_fnc = (evaluate_lstm_model,
                                sample_lstm_hyperparameters)
    else:
        raise ValueError('Unknown model type')

    if best_result is not None:
        print('Best {} result: {}'.format(model_type, results.best()))

    hyperparameters = {"n_iter": 16, "loss": "pointwise", "learning_rate": 0.1, "l2": 0.001, "embedding_dim": 64,
                       "batch_size": 16}
    print('Evaluating {}'.format(hyperparameters))

    (model,test_mrr, val_mrr) = eval_fnc(hyperparameters,
                                   train,
                                   test,
                                   validation,
                                   random_state)

    print('Test MRR {} val MRR {}'.format(
        test_mrr.mean(), val_mrr.mean()
    ))

    predictions = model.predict(1)
    print("len predictions:{}".format(len(predictions)))
    predictions = predictions.argsort()
    print("predictions:{}".format(predictions[:3]))

    data['rec_flag'] = data['news_id'].apply(lambda x: x in np.vectorize(idx2newsid.get)(predictions[:3]))
    k_res = [news[idx2newsid[idx]] for idx in predictions[:3]]
    user_id = idx2userid[1]
    print(data[data['user_id'] == user_id][['news_id', 'title']])

    print("predictions:{}".format(k_res))

    results.save(hyperparameters, test_mrr.mean(), val_mrr.mean())

    return results



def run3(train, test, validation, random_state, model_type):
    results = Results('{}_results.txt'.format(model_type))

    best_result = results.best()

    if model_type == 'pooling':
        eval_fnc, sample_fnc = (evaluate_pooling_model,
                                sample_pooling_hyperparameters)
    elif model_type == 'cnn':
        eval_fnc, sample_fnc = (evaluate_cnn_model,
                                sample_cnn_hyperparameters)
    elif model_type == 'lstm':
        eval_fnc, sample_fnc = (evaluate_lstm_model,
                                sample_lstm_hyperparameters)
    else:
        raise ValueError('Unknown model type')

    if best_result is not None:
        print('Best {} result: {}'.format(model_type, results.best()))

    hyperparameters = { "n_iter": 16, "loss": "pointwise", "learning_rate": 0.1, "l2": 0.001, "embedding_dim": 64, "batch_size": 16}
    print('Evaluating {}'.format(hyperparameters))

    (model,test_mrr, val_mrr) = eval_fnc(hyperparameters,
                                   train,
                                   test,
                                   validation,
                                   random_state)

    print('Test MRR {} val MRR {}'.format(
        test_mrr.mean(), val_mrr.mean()
    ))

    return model

def doc_tfidf_keywords(file_name, data):
    '''
    计算新闻的TFIDF值,返回每个新闻值最大的前10个关键词
    :param data:  The DataFrame where id and content exist
    :param id: The id of news
    :param content: The content of news
    :return: The frequence dict of each news
    '''

    id = 'news_id'
    content = 'content'
    news_id_content = data.loc[:, [id, content]].drop_duplicates().values

    freq_dict = {}
    for id, content in news_id_content:
        freq_dict[id] = set(jieba.analyse.extract_tags(content, topK=10))

    np.save(file_name, freq_dict)
    return  freq_dict


def user_tfidf_feature(file_name, training_data):
    '''
    返回每个用户的读过的新闻中最重要的关键词
    :param file_name:
    :param training_data:
    :return:
    '''
    user_id_news = training_data.loc[:, ['user_id', 'content']]
    grouped = user_id_news.groupby('user_id')
    user_dict = {}

    for name, df in grouped:
        strs = [content for id, content in df.values]
        strs = '.'.join(strs)
        features = set(jieba.analyse.extract_tags(strs, topK=10))

        user_dict[name] = features
    np.save(file_name, user_dict)
    return user_dict


if __name__ == '__main__':
    # http://www.pkbigdata.com/common/cmpt/CCF%E5%A4%A7%E6%95%B0%E6%8D%AE%E7%AB%9E%E8%B5%9B_%E8%B5%9B%E4%BD%93%E4%B8%8E%E6%95%B0%E6%8D%AE.html?lang=en_US
    # 用户编号	新闻编号	浏览时间	新闻标题	新闻详细内容	新闻发表时间
    # data = pd.read_csv("E:/dataset/ccf_news_rec/train.txt",sep='\t',header=None)
    data = pd.read_csv("~/dataset/ccf_news_rec/train.txt", sep='\t', header=None)
    data.columns = ['user_id', 'news_id', 'browse_time', 'title', 'content', 'published_at']
    data['title'] = data['title'].astype(str)
    data['content'] = data['content'].astype(str)
    #split train data to train,test
    last_view_idx = data.reset_index().sort_values(['browse_time'],ascending=True).groupby("user_id").index.last()
    train_data = data.drop(last_view_idx)
    test_data = data.loc[last_view_idx]
    doc_feature_dict = doc_tfidf_keywords("doc_features",train_data)
    user_feature_dict = user_tfidf_feature("user_feature",train_data)
    #gen user features
    # 当系统中用户量很大，而物品数量相对固定的时候，适合用item - based方法，例如电影、视频推荐等；当系统中物品数量很大，而用户数量相对固定的时候，适合用user - based方法，例如新闻推荐等。
    news_view_cnt = train_data.groupby("news_id").size().to_dict()
    np.save("news_view_cnt",news_view_cnt)
    # concat user feature,doc feature,user profile,doc similarity,view_cnt,pub_time
    # news = data[['news_id', 'title']].to_dict(orient='records')
    news = pd.Series(data.title.values, index=data.news_id).to_dict()
    print("len news:{}".format(len(news)))
    # test = pd.read_csv("E:/dataset/ccf_news_rec/test.csv",sep=',')
    # test = pd.read_csv("~/dataset/ccf_news_rec/test.csv", sep=',')

    cs = ColumnSummary(data[['user_id', 'news_id', 'browse_time', 'published_at']])
    print(cs)

    print("user num:{}".format(len(data['user_id'].unique())))
    print("news num:{}".format(len(data['news_id'].unique())))

    max_sequence_length = 200
    min_sequence_length = 20
    step_size = 200
    random_state = np.random.RandomState(100)

    userid2idx = {userid:idx+1 for idx,userid in enumerate(data['user_id'].unique().tolist())}
    idx2userid = {idx+1:userid for idx,userid in enumerate(data['user_id'].unique().tolist())}
    newsid2idx = {newsid:idx+1 for idx,newsid in enumerate(data['news_id'].unique().tolist())}
    idx2newsid = {idx+1:newsid for idx,newsid in enumerate(data['news_id'].unique().tolist())}

    data['user_ids']=data['user_id'].apply(lambda x:userid2idx[x]).astype(np.int32)
    data['item_ids']=data['news_id'].apply(lambda x:newsid2idx[x]).astype(np.int32)



    dataset = Interactions(user_ids=data['user_ids'].values,
                           item_ids=data['item_ids'].values,
                           timestamps=data['browse_time'].values.astype(np.int32))

    train, rest = user_based_train_test_split(dataset,
                                              random_state=random_state)
    test, validation = user_based_train_test_split(rest,
                                                   test_percentage=0.5,
                                                   random_state=random_state)
    train = train.to_sequence(max_sequence_length=max_sequence_length,
                              min_sequence_length=min_sequence_length,
                              step_size=step_size)
    test = test.to_sequence(max_sequence_length=max_sequence_length,
                            min_sequence_length=min_sequence_length,
                            step_size=step_size)
    validation = validation.to_sequence(max_sequence_length=max_sequence_length,
                                        min_sequence_length=min_sequence_length,
                                        step_size=step_size)

    mode = sys.argv[1]

    # run1(news,data,idx2newsid,idx2userid, train, test, validation, random_state, mode)
    # run2(news,train, test, validation, random_state, mode)
    # best_model = run2(train, test, validation, random_state, mode)

    ground_truth = pd.read_csv("~/dataset/ccf_news_rec/test.csv", sep=',')

    gt = pd.Series(ground_truth.newsid.values,index=ground_truth.userid).to_dict()
    print("user 1 will see: {id},{title}".format(id =gt[idx2userid[1]] ,title = news[gt[idx2userid[1]]]))
    # gt = pd.merge(ground_truth,)

    # predictions = best_model.predict(1)
    # print("len predictions:{}".format(len(predictions)))
    # predictions = predictions.argsort()
    #
    # data['rec_flag'] = data['news_id'].apply(lambda x:x in  np.vectorize(idx2newsid.get)(predictions[:3]))
    # k_res = data[data['rec_flag']==True][['news_id','title']]
    # user_id = idx2userid[1]
    # print(data[data['user_id']==user_id][['news_id','title']])
    #
    # print("predictions:{}".format(k_res))
    # print(data[data['user_id']==2][['news_id','title']])
    print(data[data['user_id']==idx2userid[1]][['browse_time','title','content']])
    # print("pred:{}".format(predictions))

    # for user_id, row in enumerate(test):
    #
    #     if not len(row.indices):
    #         continue
    #
    #     predictions = -model.predict(user_id)
    #
    #     if train is not None:
    #         rated = train[user_id].indices
    #         predictions[rated] = FLOAT_MAX
    #
    #     predictions = predictions.argsort()

    #关键词召回
    #协同过滤召回
    #热门 点击召回




