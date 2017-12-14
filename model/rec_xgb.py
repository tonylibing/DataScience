# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:47:09 2017

@author: win7
"""

import sys
import gc
import pickle
import operator
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
# from bayes_opt import BayesianOptimization

# data_path = "/wls/personal/tangning593/workspace/dev/train/data/rec_ex_model_data_small"
# data_path = "/wls/personal/tangning593/workspace/data/rec_xg_data_20170522.csv"
# data_path = "/wls/personal/tangning593/workspace/data/rec_train_data_2017-06-09_sm.csv"
data_path = "/wls/personal/tangning593/workspace/data/rec_train_data_2017-07-05_sm.csv"
# data_path = "/wls/personal/tangning593/workspace/data/rec_data_34fill.csv"
# data_path = "/wls/personal/yanxuewen543/data/rec_train_data_20170601.csv"
# data_path = "/wls/personal/tangning593/workspace/data/rec_data_20170601.csv"
data = pd.read_csv(data_path)
data.drop(['stat_date','invest_mob'],inplace=True,axis=1)
# data = data.loc[data['invest_dt']>='2017-02-10']
# gc.collect()
# print(data.columns.values)

def check_miss(df):
	return df.isnull().sum()/df.shape[0]

cnt_cols = [col for col in data.columns if '_cnt' in col]
gap_cols = [col for col in data.columns if '_gap' in col]
viewcnt_cols = [col for col in data.columns if 'viewcnt_3' in col or 'viewcnt_7' in col or 'viewcnt_14' in col]
viewgap_cols = [col for col in data.columns if 'viewgap' in col]


data[cnt_cols]=data[cnt_cols].fillna(0)
data[gap_cols]=data[gap_cols].fillna(3650)
data[viewcnt_cols]=data[viewcnt_cols].fillna(0)
data[viewgap_cols]=data[viewgap_cols].fillna(3650)

# mis = check_miss(data)
# with pd.option_context('display.max_rows',None,'display.max_columns',3):
# 	print mis

data.shape[0]
# data2=data.loc[~(data['invest_prdcate'].isnull() & data['last_product_category_rec'].isnull()) & data["invest_prdcate"]=='ljb']
# data=data.dropna()
# gc.collect()
data=data.loc[data['invest_prdcate']!='ljb']
gc.collect()
# data=data.loc[~((data['invest_prdcate'].isnull()) | (data['last_product_category_rec'].isnull()) | (data["invest_prdcate"]=='ljb'))]
# mis = check_miss(data)
# with pd.option_context('display.max_rows',None,'display.max_columns',3):
# 	print mis
# data.rename(columns={"invest_prdcate":"target"},inplace=True)
encoders_dict = defaultdict(LabelEncoder)
# categorical = ['last_product_category_rec','cust_level','invest_prdcate']
categorical = ['last_product_category_rec','cust_level','invest_prdcate','last_view_product_category_rec']
data2=data.apply(lambda x:encoders_dict[x.name].fit_transform(x) if x.name in categorical else x)

# data2.to_csv("/wls/personal/tangning593/workspace/data/rec_data_20170601_ft.csv",index=False)

data2.to_csv("/wls/personal/tangning593/workspace/data/rec_data_20170706_ft.csv",index=False)
# data.apply(lambda x:encoders_dict[x.name].fit_transform(x.astype(str)) if x.name in categorical else x)
# data_dm = pd.get_dummies(data=data,columns=cols_to_tran)
features = [col for col in data.columns if col not in ['user_id','rd','stat_date','invest_dt','invest_mob','invest_prdcate','target','loan_request_id']]
len(features)
# /wls/personal/prod_deploy/tangning593/data
# data_path = "/wls/personal/tangning593/workspace/data/rec_xg_data_2017_ft.csv"

import sys
import gc
import pickle
import operator
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from collections import defaultdict

#data_path = "/wls/personal/tangning593/workspace/data/rec_data_20170609_ft.csv"
data_path = "/wls/personal/tangning593/workspace/data/rec_data_20170706_ft.csv"
data = pd.read_csv(data_path)
# train = data.loc[(data['rd']>='2017-02-01') & (data['rd']<='2017-04-10')]
# test = data.loc[data['rd']>'2017-04-10']
sp_wts =(1 - data.groupby('invest_prdcate').size()/data.shape[0]).reset_index()
sp_wts.columns=['invest_prdcate','sample_weight']
data2=pd.merge(data,sp_wts,how='left',on='invest_prdcate')
data=data2
del data2
# train = data.loc[(data['invest_dt']>='2017-03-01') & (data['invest_dt']<='2017-03-31')]
# test = data.loc[(data['invest_dt']>='2017-04-01') & (data['invest_dt']<='2017-04-15')]
train = data.loc[(data['invest_dt']>='2017-05-04') & (data['invest_dt']<='2017-06-18')]
test = data.loc[(data['invest_dt']>='2017-06-19') & (data['invest_dt']<='2017-07-04')]
ccs =["last_product_category_rec","bxt_cnt_60","min_bxt_gap","ljb_cnt_60","min_ljb_gap","xj_1k_cnt_60","min_xj_1k_gap","b2c_5w_10w_cate2_cnt_60","min_b2c_5w_10w_cate2_gap","p2p_others_cnt_60","min_p2p_others_gap","jj_cnt_60","min_jj_gap","b2c_non_fix_rate_cnt_60","min_b2c_non_fix_rate_gap","b2c_30w_cnt_60","min_b2c_30w_gap","xt_cnt_60","min_xt_gap","xj_1w_cnt_60","min_xj_1w_gap","pef_cnt_60","min_pef_gap","others_cnt_60","min_others_gap","b2c_10w_30w_cate1_cnt_60","min_b2c_10w_30w_cate1_gap","bx_cnt_60","min_bx_gap","b2c_10w_30w_cate2_cnt_60","min_b2c_10w_30w_cate2_gap","xj_100_cnt_60","min_xj_100_gap","lxplus_cnt_60","min_lxplus_gap","aed_cnt_60","min_aed_gap","b2c_5w_cate2_cnt_60","min_b2c_5w_cate2_gap","hw_cnt_60","min_hw_gap","b2c_5w_10w_cate3_cnt_60","min_b2c_5w_10w_cate3_gap","b2c_5w_cate1_cnt_60","min_b2c_5w_cate1_gap","b2c_5w_10w_cate1_cnt_60","min_b2c_5w_10w_cate1_gap","aedx_cnt_60","min_aedx_gap","bxt_amt_inc_30","ljb_amt_inc_30","xj_1k_amt_inc_30","b2c_5w_10w_cate2_amt_inc_30","p2p_others_amt_inc_30","jj_amt_inc_30","b2c_non_fix_rate_amt_inc_30","b2c_30w_amt_inc_30","xt_amt_inc_30","xj_1w_amt_inc_30","pef_amt_inc_30","others_amt_inc_30","b2c_10w_30w_cate1_amt_inc_30","bx_amt_inc_30","b2c_10w_30w_cate2_amt_inc_30","xj_100_amt_inc_30","lxplus_amt_inc_30","aed_amt_inc_30","b2c_5w_cate2_amt_inc_30","hw_amt_inc_30","b2c_5w_10w_cate3_amt_inc_30","b2c_5w_cate1_amt_inc_30","b2c_5w_10w_cate1_amt_inc_30","aedx_amt_inc_30","total_balance","curr_aum_amt","highest_asset_amt","age","cust_level","nationality","fst_invest_days","invest_dt","invest_prdcate","bxt_viewcnt_3","bxt_viewcnt_7","bxt_viewcnt_14","ljb_viewcnt_3","ljb_viewcnt_7","ljb_viewcnt_14","xj_1k_viewcnt_3","xj_1k_viewcnt_7","xj_1k_viewcnt_14","b2c_5w_10w_cate2_viewcnt_3","b2c_5w_10w_cate2_viewcnt_7","b2c_5w_10w_cate2_viewcnt_14","p2p_others_viewcnt_3","p2p_others_viewcnt_7","p2p_others_viewcnt_14","jj_viewcnt_3","jj_viewcnt_7","jj_viewcnt_14","b2c_non_fix_rate_viewcnt_3","b2c_non_fix_rate_viewcnt_7","b2c_non_fix_rate_viewcnt_14","b2c_30w_viewcnt_3","b2c_30w_viewcnt_7","b2c_30w_viewcnt_14","xt_viewcnt_3","xt_viewcnt_7","xt_viewcnt_14","xj_1w_viewcnt_3","xj_1w_viewcnt_7","xj_1w_viewcnt_14","pef_viewcnt_3","pef_viewcnt_7","pef_viewcnt_14","others_viewcnt_3","others_viewcnt_7","others_viewcnt_14","b2c_10w_30w_cate1_viewcnt_3","b2c_10w_30w_cate1_viewcnt_7","b2c_10w_30w_cate1_viewcnt_14","bx_viewcnt_3","bx_viewcnt_7","bx_viewcnt_14","b2c_10w_30w_cate2_viewcnt_3","b2c_10w_30w_cate2_viewcnt_7","b2c_10w_30w_cate2_viewcnt_14","xj_100_viewcnt_3","xj_100_viewcnt_7","xj_100_viewcnt_14","lxplus_viewcnt_3","lxplus_viewcnt_7","lxplus_viewcnt_14","aed_viewcnt_3","aed_viewcnt_7","aed_viewcnt_14","b2c_5w_cate2_viewcnt_3","b2c_5w_cate2_viewcnt_7","b2c_5w_cate2_viewcnt_14","hw_viewcnt_3","hw_viewcnt_7","hw_viewcnt_14","b2c_5w_10w_cate3_viewcnt_3","b2c_5w_10w_cate3_viewcnt_7","b2c_5w_10w_cate3_viewcnt_14","b2c_5w_cate1_viewcnt_3","b2c_5w_cate1_viewcnt_7","b2c_5w_cate1_viewcnt_14","b2c_5w_10w_cate1_viewcnt_3","b2c_5w_10w_cate1_viewcnt_7","b2c_5w_10w_cate1_viewcnt_14","aedx_viewcnt_3","aedx_viewcnt_7","aedx_viewcnt_14","last_view_product_category_rec"]
features = [col for col in ccs if col not in ['user_id','rd','stat_date','invest_dt','invest_mob','invest_prdcate','target','loan_request_id']]
print("features:"+str(len(features)))



# , show_progress=False
# def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
#     if useTrainCV:
#         xgb_param = alg.get_xgb_params()
#         xgb_param['num_class'] = np.unique(data['invest_prdcate']).size
#         xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
#         cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
#             metrics='auc', early_stopping_rounds=early_stopping_rounds)
#         alg.set_params(n_estimators=cvresult.shape[0])
    
#     #Fit the algorithm on the data
#     alg.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='auc')   
#     #Predict training set:
#     dtrain_predictions = alg.predict(dtrain[predictors])
#     dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
#     #Print model report:
#     print "\nModel Report"
#     print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
#     print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)
#     feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
#     feat_imp.plot(kind='bar', title='Feature Importances')
#     plt.ylabel('Feature Importance Score')

gbm = xgb.XGBClassifier(max_depth=12, n_estimators=30, learning_rate=0.08,
                              subsample=0.8, colsample_bytree=0.7,
                              objective="multi:softprob", seed=999)

# gbm = xgb.XGBClassifier(max_depth=11, n_estimators=80, learning_rate=0.08,
#                               subsample=0.4, colsample_bytree=0.7, max_delta_step=37,
#                               objective="multi:softprob", seed=999)

# gbm.fit(train[features],train['invest_prdcate'],eval_metric='log_loss',eval_set=[(test[features],test['invest_prdcate'])],early_stopping_rounds=25)
#gbm.fit(train[features],train['invest_prdcate'])
gbm.fit(train[features],train['invest_prdcate'],sample_weight=train['sample_weight'])

def map_at_n(real,pred,n=2):
    rg = [-x for x in reversed(range(n+1)) if x!=0]
    pred_df = pd.DataFrame(data=pred[:,rg])
    right_ac = 0
    for i in range(pred_df.columns.size):
        right = (real==pred_df[i]).sum()
        right_ac = right_ac + right
    mapAtN = right_ac*1.0/real.shape[0]*1.0
    return mapAtN

def wrong_class_pct(real,pred,n=2):
    rg = [-x for x in reversed(range(n+1)) if x!=0]
    pred_df = pd.DataFrame(data=pred[:,rg])
    real!=pred_df[0]


test_prob = gbm.predict_proba(test[features])
test_order = np.argsort(test_prob)
map_at_n(test['invest_prdcate'],test_order,1)

#error analysis1 
a = test['invest_prdcate'].reset_index(drop=True)
b = pd.DataFrame(data = test_order[:,-1])
c = pd.concat([a,b],axis=1)
c.rename(columns={0:'pred_prdcate'},inplace=True)
d = c.loc[c['invest_prdcate']!=c['pred_prdcate']].groupby(['invest_prdcate','pred_prdcate']).size()
e = d.loc[d>3000]

#error analysis2 
a = test.reset_index(drop=True)
b = pd.DataFrame(data = test_order[:,-1])
c = pd.concat([a,b],axis=1)
c.rename(columns={0:'pred_prdcate'},inplace=True)
d = c.loc[c['invest_prdcate']!=c['pred_prdcate']].groupby(['invest_prdcate','pred_prdcate']).size()
e = d.loc[d>3000]


#hxjh wrong
d.iloc[d.index.get_level_values('invest_prdcate')==12]
#b2c wrong
d.iloc[(d.index.get_level_values('invest_prdcate')>=1) & (d.index.get_level_values('invest_prdcate')<=8)]

le = encoders_dict['invest_prdcate']
le_name_mapping = dict(zip(le.classes_,le.transform(le.classes_)))
lm = sorted(le_name_mapping.items(),key=operator.itemgetter(1),reverse=False)
for i in lm:
    print i

le = encoders_dict['invest_prdcate']
le_name_mapping = dict(zip(le.classes_,le.transform(le.classes_)))
lm = sorted(le_name_mapping.items(),key=operator.itemgetter(1),reverse=False)
for i in lm:
    print i[0]



le2 = encoders_dict['last_product_category_rec']
le_name_mapping2 = dict(zip(le.classes_,le.transform(le.classes_)))
lm2 = sorted(le_name_mapping2.items(),key=operator.itemgetter(1),reverse=False)
for i in lm2:
    print i

le3 = encoders_dict['last_view_product_category_rec']
le_name_mapping3 = dict(zip(le.classes_,le.transform(le.classes_)))
lm3 = sorted(le_name_mapping3.items(),key=operator.itemgetter(1),reverse=False)
for i in lm3:
    print i

for k,v in encoders_dict.iteritems():
    pickle.dump(v,open('/wls/personal/tangning593/workspace/data/model/le_'+k,'wb'))


mapn = map_at_n(test['invest_prdcate'],test_order)
print(mapn)



pickle.dump(gbm,open('/wls/personal/tangning593/workspace/data/model/xg20170407.pickle.dat','wb'))

gbm = pickle.load(open('/wls/personal/tangning593/workspace/data/model/xg20170407.pickle.dat','rb'))

fs_dict = gbm._Booster.get_fscore()
import operator
fs = sorted(fs_dict.items(),key=operator.itemgetter(1),reverse=True)
for i in fs:
    print i
    
modelfit(gbm,data,features)

def xgbcv(max_depth,n_estimators,learning_rate,subsample,colsample_bytree,min_child_weight,gamma):
    gbm = xgb.XGBClassifier(max_depth=int(max_depth), n_estimators=int(n_estimators), learning_rate=learning_rate,
                            subsample=subsample, colsample_bytree=colsample_bytree, 
                            min_child_weight = min_child_weight, gamma = gamma, 
                            objective="multi:softprob", seed=999,nthread=6)
    gbm.fit(train[features],train['invest_prdcate'])
    # gbm.fit(train[features],train['invest_prdcate'],eval_metric='logloss',eval_set=[(test[features],test['invest_prdcate'])],early_stopping_rounds=25)
    test_prob = gbm.predict_proba(test[features])
    test_order = np.argsort(test_prob)
    mapn = map_at_n(test['invest_prdcate'],test_order,1)
    print("max_depth:%f,n_estimators:%f,learning_rate:%f,subsample:%f,colsample_bytree:%f,min_child_weight:%f,gamma:%f,mapn:%f"%(max_depth,n_estimators,learning_rate,subsample,colsample_bytree,min_child_weight,gamma,mapn))
    return mapn

xgbBO = BayesianOptimization(xgbcv,
    {
    'max_depth':(int(10),int(14)),
    'n_estimators':(int(20),int(100)),
    'learning_rate':(0.01,0.1),
    'subsample':(0.3,0.9),
    'colsample_bytree':(0.3,0.9),
    'min_child_weight':(1,40),
    'gamma':(0.01,1)
    })

xgbBO.maximize(niter=20)
print('-'*53)
print('Final Result')
print('xgboost:%f' % xgbBO.res['max']['max_val'])
print('xgboost:%s' % xgbBO.res['max']['max_params'])

#outlier analysis
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    for num in [10, 50, 100, 1000]:
        # Generate some data
        x = np.random.normal(0, 0.5, num-3)

        # Add three outliers...
        x = np.r_[x, -3, -10, 12]
        plot(x)

    plt.show()

def mad_based_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def percentile_based_outlier(data, threshold=95):
    diff = (100 - threshold) / 2.0
    minval, maxval = np.percentile(data, [diff, 100 - diff])
    return (data < minval) | (data > maxval)

def plot(x):
    fig, axes = plt.subplots(nrows=2)
    for ax, func in zip(axes, [percentile_based_outlier, mad_based_outlier]):
        sns.distplot(x, ax=ax, rug=True, hist=False)
        outliers = x[func(x)]
        ax.plot(outliers, np.zeros_like(outliers), 'ro', clip_on=False)

    kwargs = dict(y=0.95, x=0.05, ha='left', va='top')
    axes[0].set_title('Percentile-based Outliers', **kwargs)
    axes[1].set_title('MAD-based Outliers', **kwargs)
    fig.suptitle('Comparing Outlier Tests with n={}'.format(len(x)), size=14)

main()


#b2c split

def split_b2c(product):
    b2c_type = 'b2c_others'
    if(product["product_price"]<=50000 and product["invest_period_by_days"]<=180):
        b2c_type = 'b2c_5w_6m'
    elif(product["product_price"]<=50000 and product["invest_period_by_days"]>180 and product["invest_period_by_days"]<=360):
        b2c_type = 'b2c_5w_6m_12m'
    elif(product["product_price"]<=50000 and product["invest_period_by_days"]>360):
        b2c_type='b2c_5w_12m'
    elif(product["product_price"]>50000 and product["product_price"]<=100000 and product["invest_period_by_days"]<=180):
        b2c_type = 'b2c_5w_10w_6m'
    elif(product["product_price"]>50000 and product["product_price"]<=100000 and product["invest_period_by_days"]>180 and product["invest_period_by_days"]<=360):
        b2c_type = 'b2c_5w_10w_6m_12m'
    elif(product["product_price"]>50000 and product["product_price"]<=100000 and product["invest_period_by_days"]>360):
        b2c_type='b2c_5w_10w_12m'
    elif(product["product_price"]>100000 and product["product_price"]<=200000 and product["invest_period_by_days"]<=180):
        b2c_type = 'b2c_10w_20w_6m'
    elif(product["product_price"]>100000 and product["product_price"]<=200000 and product["invest_period_by_days"]>180 and product["invest_period_by_days"]<=360):
        b2c_type = 'b2c_10w_20w_6m_12m'
    elif(product["product_price"]>100000 and product["product_price"]<=200000 and product["invest_period_by_days"]>360):
        b2c_type='b2c_10w_20w_12m'
    elif(product["product_price"]>200000 and product["product_price"]<=300000 and product["invest_period_by_days"]<=180):
        b2c_type = 'b2c_20w_30w_6m'
    elif(product["product_price"]>200000 and product["product_price"]<=300000 and product["invest_period_by_days"]>180 and product["invest_period_by_days"]<=360):
        b2c_type = 'b2c_20w_30w_6m_12m'
    elif(product["product_price"]>200000 and product["product_price"]<=300000 and product["invest_period_by_days"]>360):
        b2c_type='b2c_20w_30w_12m'
    elif(product["product_price"]>30000 and product["invest_period_by_days"]<=180):
        b2c_type = 'b2c_30w_6m'
    elif(product["product_price"]>30000 and product["invest_period_by_days"]>180 and product["invest_period_by_days"]<=360):
        b2c_type = 'b2c_30w_6m_12m'
    elif(product["product_price"]>30000 and product["invest_period_by_days"]>360):
        b2c_type='b2c_30w_12m'

    return b2c_type



select 
    b2c_type
    ,count(*)
from 
(
select 
    id
    ,case when product_price<=50000 and invest_period_by_days<=91 then 'b2c_5w_3m'
         when product_price<=50000 and invest_period_by_days>91 and invest_period_by_days<=180 then 'b2c_5w_3m_6m'
         when product_price<=50000 and invest_period_by_days>180 and invest_period_by_days<=360 then 'b2c_5w_6m_12m'
         when product_price<=50000 and invest_period_by_days>360 then 'b2c_5w_12m'
         ---
         when product_price>50000 and product_price<=100000 and invest_period_by_days<=91 then 'b2c_5w_10w_3m'
         when product_price>50000 and product_price<=100000 and invest_period_by_days>91 and invest_period_by_days<=180 then 'b2c_5w_10w_3m_6m'
         when product_price>50000 and product_price<=100000 and invest_period_by_days>180 and invest_period_by_days<=360 then 'b2c_5w_10w_6m_12m'
         when product_price>50000 and product_price<=100000 and invest_period_by_days>360 then 'b2c_5w_10w_12m'
         ---
         when product_price>100000 and product_price<=200000 and invest_period_by_days<=91 then 'b2c_10w_20w_3m'
         when product_price>100000 and product_price<=200000 and invest_period_by_days>91 and invest_period_by_days<=180 then 'b2c_10w_20w_3m_6m'
         when product_price>100000 and product_price<=200000 and invest_period_by_days>180 and invest_period_by_days<=360 then 'b2c_10w_20w_6m_12m'
         when product_price>100000 and product_price<=200000 and invest_period_by_days>360 then 'b2c_10w_20w_12m'
         ---
         when product_price>200000 and product_price<=300000 and invest_period_by_days<=91 then 'b2c_20w_30w_3m'
         when product_price>200000 and product_price<=300000 and invest_period_by_days>91 and invest_period_by_days<=180 then 'b2c_20w_30w_3m_6m'
         when product_price>200000 and product_price<=300000 and invest_period_by_days>180 and invest_period_by_days<=360 then 'b2c_20w_30w_6m_12m'
         when product_price>200000 and product_price<=300000 and invest_period_by_days>360 then 'b2c_20w_30w_12m'
         ---
         when product_price>300000 and product_price<=500000 and invest_period_by_days<=91 then 'b2c_30w_50w_3m'
         when product_price>300000 and product_price<=500000 and invest_period_by_days>91 and invest_period_by_days<=180 then 'b2c_30w_50w_3m_6m'
         when product_price>300000 and product_price<=500000 and invest_period_by_days>180 and invest_period_by_days<=360 then 'b2c_30w_50w_6m_12m'
         when product_price>300000 and product_price<=500000 and invest_period_by_days>360 then 'b2c_30w_50w_12m'
         ---
         when product_price>500000 and invest_period_by_days<=91 then 'b2c_50w_3m'
         when product_price>500000 and invest_period_by_days>91 and invest_period_by_days<=180 then 'b2c_50w_3m_6m'
         when product_price>500000 and invest_period_by_days>180 and invest_period_by_days<=360 then 'b2c_50w_6m_12m'
         when product_price>500000 and invest_period_by_days>360 then 'b2c_50w_12m'
    end b2c_type
from 
    bdtanz_dw.rec_cmn_products where item='B2C'
) a
group by 
    b2c_type
order by b2c_type