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
from sklearn.model_selection import train_test_split
from importlib import reload
sys.path.append("../..")
import feature.processor
reload(feature.processor)
from feature.processor import *
import multiprocessing
data_dir = "D:/dataset/kycdata"
cache_dir = "D:/dataset/kycdata/cache"
train_data = pickle.load(open(os.path.join(cache_dir,"million_train_set_2017-12-01_2017-12-30_2017-12-31_2018-01-29.pkl"),'rb'))


cols = [col for col in train_data.columns.values if col not in ['user_id', 'product_group', 'label']]
sl = FeatureSelection()
sl.fit(train_data[cols], train_data['label'])
bfp = FeatureEncoder(None, sl.numerical_cols, sl.categorical_cols)
dump_path = os.path.join(cache_dir, 'train_matrix.csv')
bfp.fit_transform(train_data[sl.selected_cols], train_data['label'], dump_path)

dump_path = os.path.join(cache_dir, 'test_matrix.csv')
bfp.fit_transform(test_data[sl.selected_cols], test_data['label'], dump_path)


train_set = pd.read_csv(os.path.join(cache_dir,'train_matrix.csv'))
