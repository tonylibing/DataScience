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
from importlib import reload
sys.path.append("../..")
import feature.processor
reload(feature.processor)
from feature.processor import *

data_dir = "E:/dataset/kycdata"
cache_dir = "E:/dataset/kycdata/cache"
#data_dir = "/home/tanglek/dataset/kycdata"
#cache_dir = "/home/tanglek/dataset/kycdata/cache"

data = pickle.load(open(os.path.join(cache_dir,'merge_million_train_set_2017-12-01_2018-03-01_window30_step5.pkl'),'rb')).reset_index(drop=True)


cols = [col for col in data.columns.values if col not in ['user_id','product_group','label']]
sl = FeatureSelection()
sl.fit(data[cols],data['label'])

bfp = FeatureEncoder(None,sl.numerical_cols,sl.categorical_cols)

dump_path = os.path.join(cache_dir,'train_matrix.csv')
bfp.fit_transform(data[sl.selected_cols],data['label'],dump_path)


import matplotlib.pyplot as plt
feat_imp = pd.Series(gbm.booster().get_fscore()).sort_values(ascending=False)
import pandas as pd
feat_imp = pd.Series(gbm.booster().get_fscore()).sort_values(ascending=False)
feat_imp = pd.Series(gbm.get_fscore()).sort_values(ascending=False)

feat_imp = pd.Series(gbm.booster().get_fscore()).sort_values(ascending=False)
gbm.booster().get_fscore()
gbm.booster.get_fscore()
feat_imp = pd.Series(gbm.get_booster().get_fscore()).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')