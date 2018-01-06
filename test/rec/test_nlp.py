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
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score,f1_score, accuracy_score, \
    average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from eval.metrics import ks_statistic
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from model.GBDTLRClassifier import *

# http://www.pkbigdata.com/common/cmpt/CCF%E5%A4%A7%E6%95%B0%E6%8D%AE%E7%AB%9E%E8%B5%9B_%E8%B5%9B%E4%BD%93%E4%B8%8E%E6%95%B0%E6%8D%AE.html?lang=en_US
# 用户编号	新闻编号	浏览时间	新闻标题	新闻详细内容	新闻发表时间
data = pd.read_csv("E:/dataset/ccf_news_rec/train.txt",sep='\t',header=None)
data.columns = ['user_id','news_id','browse_time','title','content','published_at']
test = pd.read_csv("E:/dataset/ccf_news_rec/test.csv",sep=',')

cs = ColumnSummary(data[['user_id','news_id','browse_time','published_at']])
print(cs)

