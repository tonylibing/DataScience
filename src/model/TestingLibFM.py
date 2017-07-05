
# coding: utf-8

# ## Testing implementations of LibFM
# 
# By **LibFM** I mean an approach to solve classification and regression problems.
# This approach is frequently used in recommendation systems, because it generalizes the matrix decompositions. 
# 
# **LibFM** proved to be quite useful to deal with highly categorical data (user id / movie id / movie language / site id / advertisement id / etc.).
# 
# Implementations tested
# 
# 
# * Original and widely known implementation was written by Steffen Rendle (and available on [github](https://github.com/srendle/libfm)).
#     * contains SGD, SGDA, ALS and MCMC optimizers
#     * command-line interface, does not have official python / R wrapper
#     * does not provide a way to save / load trained formula. Each time you want to predict something, you need to restart training process 
#     * has some extensions (that almost nobody uses)
#     * supports linux, mac os
#     * has non-oficial [pythonic wrapper](https://github.com/jfloff/pywFM)
# 
# 
# * FastFM ([github repo](https://github.com/ibayer/fastFM))
#     * claimed to be faster in the author's article
#     * has both command-line interface and convenient python wrapper, which *almost* follows scikit-learn conventions.
#     * supports SGD, ALS and MCMC optimizers
#     * supports save / load (for the except of MCMC)
#     * supports linux, mac os (though some issues with mac os)
#     
#     
# * pylibFM ([github repo](https://github.com/coreylynch/pyFM))
#     * uses SGDA
#     * pythonic library implemented with cython
#     * save / load operates normally
#     * supports any platform, provided cython operates normally
#     * slow and requires additional tuning, the number of iterations is reduced for pylibFM in tests
#     
# None of the libraries are pip-installable and all libraries need some manual setup. FastFM is the only to install itself normally into site-packages.

# ## What is tested
# 
# ALS (alternating least squares) is very useful optimization technique for factorization models, however
# there is still one parameter one has to pass - namely, regularization. Quality of classification / regression is quite sensible to this parameter, so for fast tests data analyst prefers to leave the question of selecting regularization to machine learning.
# 
# MCMC is usually proposed as a solution: optimization algorithm should "find" the optimal regularization. 
# MCMC uses however some priors (which don't influence the result that much).
# 
# So I am testing the quality libraries provide **without additional tuning** to check how bayesian inference and other heuristics work.
# 
# 
# ## Logistic regression
# 
# Logistic regression is used as a stable **baseline**, because it is basic method to work with highly categorical data.
# 
# However, logistic regression, for instance, does not encounter the relation between user variables and movie variables (in the context of movie recommendations), so this approach is not able to provide any senseful recommendations.

# In[1]:

import numpy
import pandas
import load_problems
import cPickle as pickle
from sklearn.metrics import roc_auc_score, mean_squared_error


# In[2]:

from fastFM.mcmc import FMClassification, FMRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.datasets import dump_svmlight_file


# ## Defining functions for benchmarking

# In[3]:

LIBFM_PATH = '/moosefs/ipython_env/python_libfm/bin/libFM'
PYLIBFM_PATH = '/moosefs/ipython_env/python_pylibFM/'

import sys
if PYLIBFM_PATH not in sys.path:
    sys.path.insert(0, PYLIBFM_PATH)
import pylibfm


def fitpredict_logistic(trainX, trainY, testX, classification=True, **params):
    encoder = OneHotEncoder(handle_unknown='ignore').fit(trainX)
    trainX = encoder.transform(trainX)
    testX = encoder.transform(testX)
    if classification:
        clf = LogisticRegression(**params)
        clf.fit(trainX, trainY)
        return clf.predict_proba(testX)[:, 1]
    else:
        clf = Ridge(**params)
        clf.fit(trainX, trainY)
        return clf.predict(testX)

def fitpredict_fastfm(trainX, trainY, testX, classification=True, rank=8, n_iter=100):
    encoder = OneHotEncoder(handle_unknown='ignore').fit(trainX)
    trainX = encoder.transform(trainX)
    testX = encoder.transform(testX)
    if classification:
        clf = FMClassification(rank=rank, n_iter=n_iter)
        return clf.fit_predict_proba(trainX, trainY, testX)
    else:
        clf = FMRegression(rank=rank, n_iter=n_iter)
        return clf.fit_predict(trainX, trainY, testX)  

def fitpredict_libfm(trainX, trainY, testX, classification=True, rank=8, n_iter=100):
    encoder = OneHotEncoder(handle_unknown='ignore').fit(trainX)
    trainX = encoder.transform(trainX)
    testX = encoder.transform(testX)
    train_file = 'libfm_train.txt'
    test_file = 'libfm_test.txt'
    with open(train_file, 'w') as f:
        dump_svmlight_file(trainX, trainY, f=f)
    with open(test_file, 'w') as f:
        dump_svmlight_file(testX, numpy.zeros(testX.shape[0]), f=f)
    task = 'c' if classification else 'r'
    console_output = get_ipython().getoutput(u"$LIBFM_PATH -task $task -method mcmc -train $train_file -test $test_file -iter $n_iter -dim '1,1,$rank' -out output.libfm")
    
    libfm_pred = pandas.read_csv('output.libfm', header=None).values.flatten()
    return libfm_pred

def fitpredict_pylibfm(trainX, trainY, testX, classification=True, rank=8, n_iter=10):
    encoder = OneHotEncoder(handle_unknown='ignore').fit(trainX)
    trainX = encoder.transform(trainX)
    testX = encoder.transform(testX)
    task = 'classification' if classification else 'regression'
    fm = pylibfm.FM(num_factors=rank, num_iter=n_iter, verbose=False, task=task)
    if classification:
        fm.fit(trainX, trainY)
    else:
        fm.fit(trainX, trainY * 1.)
    return fm.predict(testX)


# ### Executing all of the tests takes much time
# 
# Below is simple mechanism, which preserves results between runs.

# In[4]:

from collections import OrderedDict
import time

all_results = OrderedDict()
try:
    with open('./saved_results.pkl') as f:
        all_results = pickle.load(f)
except:
    pass

def test_on_dataset(trainX, testX, trainY, testY, task_name, classification=True, use_pylibfm=True):
    algorithms = OrderedDict()
    algorithms['logistic'] = fitpredict_logistic
    algorithms['libFM']    = fitpredict_libfm
    algorithms['fastFM']   = fitpredict_fastfm
    if use_pylibfm:
        algorithms['pylibfm']  = fitpredict_pylibfm
    
    results = pandas.DataFrame()
    for name, fit_predict in algorithms.items():
        start = time.time()
        predictions = fit_predict(trainX, trainY, testX, classification=classification)
        spent_time = time.time() - start
        results.ix[name, 'time'] = spent_time
        if classification:
            results.ix[name, 'ROC AUC'] = roc_auc_score(testY, predictions)
        else:
            results.ix[name, 'RMSE'] = numpy.mean((testY - predictions) ** 2) ** 0.5
            
    all_results[task_name] = results
    with open('saved_results.pkl', 'w') as f:
        pickle.dump(all_results, f)
        
    return results


# ## Testing on movielens-100k dataset, only ids
# 
# MovieLens dataset is famous dataset in recommender systems. The task is to predict ratings for movies

# In[5]:

trainX, testX, trainY, testY = load_problems.load_problem_movielens_100k(all_features=False)
trainX.head()


# In[6]:

test_on_dataset(trainX, testX, trainY, testY, task_name='ml100k, ids', classification=False)


# ## Testing on movielens-100k dataset, with additional information

# In[7]:

trainX, testX, trainY, testY = load_problems.load_problem_movielens_100k(all_features=True)
trainX.head()


# In[8]:

test_on_dataset(trainX, testX, trainY, testY, task_name='ml100k', classification=False)


# ## Testing on movielens-1m dataset, only ids

# In[14]:

trainX, testX, trainY, testY = load_problems.load_problem_movielens_1m(all_features=False)
trainX.head()


# In[15]:

test_on_dataset(trainX, testX, trainY, testY, task_name='ml-1m,ids', classification=False)


# ## Testing on movielens-1m dataset, with additional information

# In[ ]:

trainX, testX, trainY, testY = load_problems.load_problem_movielens_1m(all_features=True)
trainX.head()


# In[34]:

test_on_dataset(trainX, testX, trainY, testY, task_name='ml-1m', classification=False)


# ## Test on flights dataset - 1m
# 
# Flights dataset is quite famous due to [these benchmarks](github.com/szilard/benchm-ml) by szilard. 
# 
# Based on defferent charateristics the goal is to predict whether the flight was delayed by 15 minutes or more.

# In[17]:

trainX, testX, trainY, testY = load_problems.load_problem_flight(large=False, convert_to_ints=False)
trainX.head()


# In[24]:

trainX, testX, trainY, testY = load_problems.load_problem_flight(large=False, convert_to_ints=True)
trainX.head()


# In[23]:

test_on_dataset(trainX, testX, trainY, testY, task_name='flight1m', classification=True)


# ## Test on flights dataset - 10m
# 
# pylibFM drops the kernel, so doesn't participate in comparison

# In[ ]:

trainX, testX, trainY, testY = load_problems.load_problem_flight(large=True, convert_to_ints=True)
trainX.head()


# In[11]:

test_on_dataset(trainX, testX, trainY, testY, task_name='flight10m', classification=True, use_pylibfm=False)


# ## Flights dataset with additional features
# 
# We simply add some 'quadratic' features

# In[63]:

trainX, testX, trainY, testY = load_problems.load_problem_flight_extended(large=False)
trainX.head()


# In[64]:

test_on_dataset(trainX, testX, trainY, testY, task_name='flight1m, ext', classification=True)


# ## Test on Avazu dataset (100k)
# 
# Avazu dataset comes from kaggle challenge, goal is to predict Click-Through Rate. 
# 
# All the variables given are categorical, LibFM gave good results in this challenge.

# In[9]:

trainX, testX, trainY, testY = load_problems.load_problem_ad(train_size=100000)
# taking max hash of 1000 for each category
trainX = trainX % 1000
testX = testX % 1000
trainX.head()


# In[36]:

test_on_dataset(trainX, testX, trainY, testY, 
                task_name='avazu100k', classification=True, use_pylibfm=False)


# ## Avazu 1m

# In[35]:

trainX, testX, trainY, testY = load_problems.load_problem_ad(train_size=1000000)
# taking max hash of 1000 for each category
trainX = trainX % 1000
testX = testX % 1000
test_on_dataset(trainX, testX, trainY, testY, 
                task_name='avazu1m', classification=True, use_pylibfm=False)


# # Results
# 
# composing all results in one table. 
# RMSE should be minimal, ROC AUC - maximal.

# In[69]:

results_table = pandas.DataFrame()
tuples = []

for name in ['ml100k, ids', 'ml-1m,ids', 'ml100k', 'ml-1m', 'flight1m', 'flight1m, ext', 'flight10m', 'avazu100k', 'avazu1m']:
    df = all_results[name]
    results_table[name + ' (time)'] = df['time']
    metric_name = df.columns[-1]
    results_table[name + metric_name] = df[metric_name]
    tuples.append([name, 'time'])
    tuples.append([name, df.columns[-1]])
    
results_table = results_table.T
results_table.index = pandas.MultiIndex.from_tuples(tuples, names=['dataset', 'value'])
results_table.T


# # Conclusion
# 
# - `pylibfm` is out of the game. It is slow, it crashes on large datasets, sometimes simply diverge and hardly can compete in quality. <br />
#    Nothing new, adaptive methods require babysitting
# - `FastFM` and `LibFM` are quite stable and fast
# -  but `LibFM`, being a bit faster, on average provides much better results.
# 
# As a sidenote, we saw on example with flight dataset that some feature engineering with providing quadratic features gives very significant boost in quality - even logisitic regression can work much better and faster than FMs on original features.
