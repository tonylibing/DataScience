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
data_dir = "E:/dataset/alimama"
cache_dir = "E:/dataset/alimama/cache"
train_data = pd.read_csv(os.path.join(data_dir,'round1_ijcai_18_train_20180301.txt'),sep=' ')
test_data = pd.read_csv(os.path.join(data_dir,'round1_ijcai_18_test_a_20180301.txt'),sep=' ')



import pickle
import lightgbm as lgb
import seaborn as sns
lg
lgb.plot_importance(lgbmodel,height=0.8,importance_type='split',max_num_features=50,figsize=(8,10),grid=False)
lgb.plot_tree(lgbmodel,tree_index=0)