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
cl = pickle.load(open(os.path.join(cache_dir,"user_invest_feat_2017-12-16_2018-01-14.pkl"),'rb'))

cols = [col for col in res.columns.values if col not in ['user_id', 'product_group', 'label']]
sl = FeatureSelection(self.args)
sl.fit(res[cols], res['label'])
bfp = FeatureEncoder(None, sl.numerical_cols, sl.categorical_cols)
dump_path = os.path.join(self.cache_dir, 'train_matrix.csv')
bfp.fit_transform(res[sl.selected_cols], res['label'], dump_path)