import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import os
import sys
import xgboost as xgb
import lightgbm as lgb
import pickle
from dateutil.parser import parse
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from importlib import reload
sys.path.append("../..")
import feature.processor
reload(feature.processor)
from feature.processor import *
import multiprocessing
data_dir = "E:/dataset/alimama"
cache_dir = "E:/dataset/alimama/cache"
train_data = pd.read_csv(os.path.join(data_dir,'round1_ijcai_18_train_20180301.txt'),sep=' ')
test_data = pd.read_csv(os.path.join(data_dir,'round1_ijcai_18_test_a_20180301.txt'),sep=' ')

import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import lightgbm as lgb
from dateutil.parser import parse
import matplotlib.pyplot as plt
import seaborn as sns
# from matplotlib.pylab import rcParams
import time
import os
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, GridSearchCV

data_dir = 'E:/dataset/alimama/'
train_data = pd.read_csv(os.path.join(data_dir, "round1_ijcai_18_train_20180301.txt"), sep=" ")
test_data = pd.read_csv(os.path.join(data_dir, "round1_ijcai_18_test_a_20180301.txt"), sep=" ")
# print(train_data.head())
# 看是否有缺失值
train_data.drop_duplicates(inplace=True)  # 默认为所有行相同去重，也可以指定列下去重
test_data.drop_duplicates(inplace=True)  # 默认为所有行相同去重，也可以指定列下去重


def timestamp_convert(value):
    format = '%Y-%m-%d %H:%M:%S'  # 要转化的格式
    value = time.localtime(value)
    newday = time.strftime(format, value)
    return newday


def covert_seconds_to_days(data):
    data['time'] = data['context_timestamp'].apply(timestamp_convert)  # 将这一列的每个数据进行一一转换
    data['date'] = data.time.apply(lambda x: x[0:10]) # 将这一列的每个数据进行一一转换
    data['day'] = data.time.apply(lambda x: int(x[8:10]))  # 将这一列的每个数据进行一一切片
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    del data['context_timestamp']
    return data



train_data = covert_seconds_to_days(train_data)
test_data = covert_seconds_to_days(test_data)
cols = ['context_id',
            'context_page_id',
            'instance_id',
            'item_brand_id',
            'item_city_id',
            'item_collected_level',
            'item_id',
            'item_price_level',
            'item_pv_level',
            'shop_id',
            'shop_review_positive_rate',
            'shop_score_delivery',
            'shop_score_description',
            'shop_star_level',
            'user_age_level',
            'user_gender_id',
            'user_id',
            'day',
            'hour']
labels = 'is_trade'

sl = FeatureSelection()
sl.fit(train_data[cols], train_data['is_trade'])
bfp = FeatureEncoder(None, sl.numerical_cols, sl.categorical_cols,True)
dump_path = os.path.join(cache_dir, 'train_matrix_one_hot.csv')
bfp.fit_transform(train_data[sl.selected_cols], train_data['is_trade'], dump_path)
y_train=train_data['is_trade']
scale_pos_weight = (y_train[y_train == 0].shape[0]) * 1.0 / (y_train[y_train == 1].shape[0])
cores = multiprocessing.cpu_count()
threads = int(cores * 0.8)
gbm = xgb.XGBClassifier(n_estimators=30, learning_rate=0.3, max_depth=4, min_child_weight=6, gamma=0.3,
                        subsample=0.7,
                        colsample_bytree=0.7, objective='binary:logistic', nthread=threads,
                        scale_pos_weight=scale_pos_weight, reg_alpha=1e-05, reg_lambda=1, seed=27)

data = pd.read_csv(os.path.join(cache_dir,'train_matrix_one_hot.csv'))
kfold = 5
skf = StratifiedKFold(n_splits=kfold, random_state=42)
cols = [col for col in data.columns.values if ('is_trade' not in col)]
scores = cross_val_score(gbm, data[cols], data['is_trade'], cv=skf,scoring='neg_log_loss')
print("LogLoss: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


cols = ['context_id',
            'context_page_id',
            'instance_id',
            'item_brand_id',
            'item_city_id',
            'item_collected_level',
            'item_id',
            'item_price_level',
            'item_pv_level',
            'shop_id',
            'shop_review_positive_rate',
            'shop_score_delivery',
            'shop_score_description',
            'shop_star_level',
            'user_age_level',
            'user_gender_id',
            'user_id',
            'day',
            'hour']
labels = 'is_trade'
sl = FeatureSelection()
sl.fit(train_data[cols], train_data['is_trade'])
bfp = FeatureEncoder(None, sl.numerical_cols, sl.categorical_cols)
dump_path = os.path.join(cache_dir, 'train_matrix.csv')
bfp.fit_transform(train_data[sl.selected_cols], train_data['is_trade'], dump_path)
y_train=train_data['is_trade']
scale_pos_weight = (y_train[y_train == 0].shape[0]) * 1.0 / (y_train[y_train == 1].shape[0])
cores = multiprocessing.cpu_count()
threads = int(cores * 0.8)
gbm = xgb.XGBClassifier(n_estimators=30, learning_rate=0.3, max_depth=4, min_child_weight=6, gamma=0.3,
                        subsample=0.7,
                        colsample_bytree=0.7, objective='binary:logistic', nthread=threads,
                        scale_pos_weight=scale_pos_weight, reg_alpha=1e-05, reg_lambda=1, seed=27)

data = pd.read_csv(os.path.join(cache_dir,'train_matrix.csv'))
kfold = 5
skf = StratifiedKFold(n_splits=kfold, random_state=42)
cols = [col for col in data.columns.values if ('is_trade' not in col)]
scores = cross_val_score(gbm, data[cols], data['is_trade'], cv=skf,scoring='neg_log_loss')
print("LogLoss: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



dump_path = os.path.join(cache_dir, 'test_matrix.csv')
bfp.transform(test_data[sl.selected_cols], None, dump_path)

test_data = pd.read_csv(os.path.join(cache_dir,'test_matrix.csv'))


