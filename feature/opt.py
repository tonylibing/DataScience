# --bayes_opt
import gc
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from bayes_opt import BayesianOptimization

data_path = "/wls/xxx/data/rec_xg_data_2017_ft.csv"
data = pd.read_csv(data_path)
train = data.loc[(data['rd']>='2017-02-01') & (data['rd']<='2017-03-15')]
test = data.loc[data['rd']>'2017-03-15']
features = [col for col in data.columns if col not in ['user_id','rd','stat_date','invest_dt','invest_mob','invest_prdcate','target']]
# len(features)


def map_at_n(real,pred,n=2):
    rg = [-x for x in reversed(range(n+1)) if x!=0]
    pred_df = pd.DataFrame(data=pred[:,rg])
    right_ac = 0
    for i in range(pred_df.columns.size):
        right = (real==pred_df[i]).sum()
        right_ac = right_ac + right
    mapAtN = right_ac*1.0/real.shape[0]*1.0
    return mapAtN

def xgbcv(max_depth,n_estimators,learning_rate,subsample,colsample_bytree,min_child_weight,gamma):
    gbm = xgb.XGBClassifier(max_depth=int(max_depth), n_estimators=int(n_estimators), learning_rate=learning_rate,
                            subsample=subsample, colsample_bytree=colsample_bytree, 
                            min_child_weight = min_child_weight, gamma = gamma, 
                            objective="multi:softprob", seed=999,nthread=7)
    gbm.fit(train[features],train['invest_prdcate'])
    # gbm.fit(train[features],train['invest_prdcate'],eval_metric='logloss',eval_set=[(test[features],test['invest_prdcate'])],early_stopping_rounds=25)
    test_prob = gbm.predict_proba(test[features])
    test_order = np.argsort(test_prob)
    mapn = map_at_n(test['invest_prdcate'],test_order,1)
    with open('xgb_bayes_opt_results.txt','a') as f: f.write("max_depth:%f,n_estimators:%f,learning_rate:%f,subsample:%f,colsample_bytree:%f,min_child_weight:%f,gamma:%f,mapn:%f\n"%(max_depth,n_estimators,learning_rate,subsample,colsample_bytree,min_child_weight,gamma,mapn))
    return mapn

xgbBO = BayesianOptimization(xgbcv,
    {
    'max_depth':(int(10),int(12)),
    'n_estimators':(int(20),int(100)),
    'learning_rate':(0.05,0.1),
    'subsample':(0.3,0.5),
    'colsample_bytree':(0.6,0.8),
    'min_child_weight':(1,40),
    'gamma':(0.05,1)
    })

xgbBO.maximize(niter=20)
print('-'*53)
print('Final Result')
print('xgboost:%f' % xgbBO.res['max']['max_val'])
print('xgboost:%s' % xgbBO.res['max']['max_params'])


