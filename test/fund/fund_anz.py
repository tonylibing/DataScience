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
from sklearn.preprocessing import LabelEncoder
import multiprocessing
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score, f1_score, accuracy_score, \
average_precision_score, log_loss

data_dir = "/wls/personal/tangning593/workspace/ail/data/fund/data"
cache_dir = "/wls/personal/tangning593/workspace/ail/data/fund/cache"

#data_dir = "/wls/personal/tangning593/workspace/ail/data/fund/data"
#cache_dir = "/wls/personal/tangning593/workspace/ail/data/fund/cache"

fund_data = pd.read_csv(os.path.join(data_dir,"fund_data.csv"),sep='\001',header=None,names=['user_id','product_id','invest_amount','invest_date','fund_code','fund_brand','fund_name','fund_type','fund_risk_level','buy_daily_limit','buy_fee_rate_desc','buy_fee_discount_desc','min_invest_amount','redemption_fee_rate_desc','charge_type','redemption_arrival_day','fund_opening_type','established_date','dividend_type','trustee','collection_mode','age','gender','is_mob_attr_region','mobile_no_attribution','nationality_cn','jijin_invest_amt','model_jijin_risk_pref','score_level','fund_question_no1','fund_question_no2','fund_question_no3','fund_question_no4','fund_question_no5','fund_question_no6','fund_question_no7','fund_question_no8','fund_question_no9','fund_question_no10','fund_question_no11','fund_question_no12','fund_question_no13','fund_risk_verify_status_cn','survey_risk_question_no1','survey_risk_question_no2','survey_risk_question_no3','survey_risk_question_no4','survey_risk_question_no5','survey_risk_question_no6','survey_risk_question_no7','survey_risk_question_no8','survey_risk_question_no9','survey_risk_question_no10','survey_risk_question_no11','survey_risk_question_no12','survey_risk_question_no13','prd_trav_cfund','prd_trav_bfund','prd_trav_sfund','prd_trav_mfund','fs_gbd_score','fs_trav_potential','prd_period_pref','prd_threshold_pref','prd_fix_flexible_pref','kyc_financial_strength','kyc_risk_appetite','aum','historical_max_aum','template_id','template_version_no','class_level','score','q1','q2','q3','q4','q5','q6','q7','q8','q9','q10','q11','q12','q13','q14','q15','q16','q17','q18','q19','q20','marriage_status_cd','education_cd','family_member_quantiny','industry','prof','life_cycle','age_range','cust_aum_flag','group_vip_level','group_vip_flag','toa_pa_act_assets_amt','toa_pa_act_debts_amt','posses_house_auto_flag','series_prod_type_count','pc_insu_vip_flag','pc_insu_vip_level','hold_auto_prod_flag','hold_moto_prod_flag','pc_prod_type_count','pnc_aum_flag','vehicle_loss_insured_value','vehicle_quantity','vip_flag','wealth_score','invest_exp','risk_sensitive','ml_model_level','risk_level'])
fund_data.drop(['buy_fee_rate_desc','buy_fee_discount_desc','redemption_fee_rate_desc','family_member_quantiny','life_cycle','group_vip_level','pc_insu_vip_level','hold_auto_prod_flag','pc_prod_type_count','pnc_aum_flag'],axis=1,inplace=True)
print(fund_data.columns.values)
cs=ColumnSummary(fund_data)
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
print(cs)

mod = fund_data['education_cd'].mode()[0]
fund_data['education_cd'].replace('‰', mod, inplace=True)
mod = fund_data['marriage_status_cd'].mode()[0]
fund_data['marriage_status_cd'].replace('@', mod, inplace=True)

features = [col for col in fund_data.columns.values if (('fund_' not in col) and ((col in ['gender','mobile_no_attribution','nationality_cn','model_jijin_risk_pref','score_level','fund_risk_verify_status_cn']) or ('age' in col) or ('industry' in col) or ('prof' in col) or ('_cd' in col) or ('risk' in col) or ('q' in col) or ('fund_question' in col) or ('survey_risk_question' in col) or ('trav' in col) or ('level' in col) or ('flag' in col)))]


sl = FeatureSelection()
sl.fit(fund_data[features], fund_data['fund_type'])
bfp = FeatureEncoder(None, sl.numerical_cols, sl.categorical_cols)
le = LabelEncoder()
le.fit(np.unique(fund_data['fund_type'].values))
fund_data['label']=fund_data[['fund_type']].apply(le.transform)

dump_path = os.path.join(cache_dir, 'train_matrix.csv')
data_to_save = bfp.fit_transform(fund_data[sl.selected_cols], fund_data['label'], dump_path)
with open(dump_path, 'w') as gf:
data_to_save.to_csv(gf, header=True, index=False)


data_path = os.path.join(cache_dir, "train_matrix.csv")
with open(data_path, 'rb') as gf:
data = pd.read_csv(gf)

cols = [col for col in data.columns.values if col not in ['label']]
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(data.loc[:, cols], y, test_size=0.2, random_state=999,stratify=y)

threads = int(0.9 * multiprocessing.cpu_count())
lgbm = xgb.XGBClassifier(n_estimators=30, learning_rate=0.3, max_depth=10, min_child_weight=6, gamma=0.3,
subsample=0.7,
colsample_bytree=0.7, objective='multi:softprob', nthread=threads,reg_alpha=1e-05, reg_lambda=1, seed=27)
print('training set...')
print(y_train.value_counts())
lgbm.fit(X_train, y_train)
print('training...')
del data
model_version = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
dump_model_path = os.path.join(data_dir, 'fund_model_%s.pkl' % model_version)
with open(dump_model_path, 'wb') as gf:
pickle.dump(lgbm, gf)

print("Features importance...")
feat_imp = pd.Series(lgbm.get_booster().get_score()).sort_values(ascending=False)
# feat_imp = pd.Series(gbm.booster().get_fscore()).sort_values(ascending=False)
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
print(feat_imp)

y_pre = lgbm.predict(X_test)
y_pro = lgbm.predict_proba(X_test)[:, 1]
# 0 is the final test data
print(confusion_matrix(y_test, y_pre))