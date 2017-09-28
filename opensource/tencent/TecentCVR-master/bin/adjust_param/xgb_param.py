#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import feature as fea
os.environ["OMP_NUM_THREADS"] = "8" # 并行训练
rng = np.random.RandomState(4315)

import warnings
warnings.filterwarnings("ignore")
param_grid = {
    'max_depth': [3, 4, 5, 6, 7,8],
    'n_estimators': [10, 50, 100, 400, 800, 1000, 2000],
    'leraning_rate': [0.1, 0.2, 0.3],
    'gamma': [0, 0.2],
    'subsample': [0.8, 1],
    'colsample._bylevel': [0.8, 1]
}

xgb_model = xgb.XGBClassifiler()
rgs= GridSearchCV(xgb_model, param_grid, n_jobs=-1)

rgs.fit(fea.x_user_ad_app, fea.y_user_ad_app.reshape(fea.y_user_ad_app.shape[0], ))

print(rgs.best_score_)
print(rgs.best_params_)


