import sys
import hashlib
import json
import os
import shutil
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from sklearn.model_selection import ParameterSampler

from spotlight.cross_validation import user_based_train_test_split
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.sequence.representations import CNNNet
from spotlight.evaluation import sequence_mrr_score
from spotlight.interactions import Interactions

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

        # open(self._filename, 'a+')

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

    return test_mrr, val_mrr


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

    return test_mrr, val_mrr


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

    return test_mrr, val_mrr


def run(train, test, validation, random_state, model_type,opts):
    fname = os.path.join(opts.data_dir,'data','{0}_results.txt'.format(model_type))
    results = Results(fname)

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

        (test_mrr, val_mrr) = eval_fnc(hyperparameters,
                                       train,
                                       test,
                                       validation,
                                       random_state)

        print('Test MRR {} val MRR {}'.format(
            test_mrr.mean(), val_mrr.mean()
        ))

        results.save(hyperparameters, test_mrr.mean(), val_mrr.mean())

    return results


def main(_):
    args_in = sys.argv[1:]
    print(args_in)
    parser = argparse.ArgumentParser()
    mtyunArgs = parser.add_argument_group('美团云选项')
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
    opts = parser.parse_args(args_in)
    # http://www.pkbigdata.com/common/cmpt/CCF%E5%A4%A7%E6%95%B0%E6%8D%AE%E7%AB%9E%E8%B5%9B_%E8%B5%9B%E4%BD%93%E4%B8%8E%E6%95%B0%E6%8D%AE.html?lang=en_US
    # 用户编号	新闻编号	浏览时间	新闻标题	新闻详细内容	新闻发表时间
    # data = pd.read_csv("E:/dataset/ccf_news_rec/train.txt",sep='\t',header=None)
    with tf.gfile.FastGFile(os.path.join(opts.data_dir, "train.txt"), 'rb') as gf:
        data = pd.read_csv(gf, sep='\t', header=None)
    data.columns = ['user_id', 'news_id', 'browse_time', 'title', 'content', 'published_at']
    # test = pd.read_csv("E:/dataset/ccf_news_rec/test.csv",sep=',')
    with tf.gfile.FastGFile(os.path.join(opts.data_dir, "test.csv"), 'rb') as gf:
        test = pd.read_csv(gf, sep=',')

    # cs = ColumnSummary(data[['user_id', 'news_id', 'browse_time', 'published_at']])
    # print(cs)

    max_sequence_length = 200
    min_sequence_length = 20
    step_size = 200
    random_state = np.random.RandomState(100)

    dataset = Interactions(user_ids=data['user_id'].values.astype(np.int32),
                           item_ids=data['news_id'].values.astype(np.int32),
                           timestamps=data['browse_time'].values.astype(np.int32))
    # dataset = Interactions(user_ids=data['user_id'].values.astype(np.int32),
    #                        item_ids=data['news_id'].values.astype(np.int32))
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

    mode = opts.model_type

    run(train, test, validation, random_state, mode,opts)

if __name__ == '__main__':
    tf.app.run(main=main)