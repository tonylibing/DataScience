# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import sys
import multiprocessing
import psutil
import argparse
import pickle
import scipy
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score, f1_score, accuracy_score, \
    average_precision_score, log_loss
import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime
from dateutil.parser import parse
import xgboost as xgb
import lightgbm as lgb
from tpot import TPOTClassifier
# import xlearn as xl
from tqdm import tqdm
from collections import Counter
from importlib import reload
from imblearn.over_sampling import SMOTE
sys.path.append("../..")
import feature.processor

reload(feature.processor)
from feature.processor import *
from model.GBDTLRClassifier import *

# from wordbatch.models import FTRL, FM_FTRL

FLAGS = None


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

def plot_confusion_matrix(cm, genre_list, name, title):
    pylab.clf()
    pylab.matshow(cm, fignum=False, cmap='Blues', vmin=0, vmax=1.0)
    ax = pylab.axes()
    ax.set_xticks(range(len(genre_list)))
    ax.set_xticklabels(genre_list)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks(range(len(genre_list)))
    ax.set_yticklabels(genre_list)
    pylab.title(title)
    pylab.colorbar()
    pylab.grid(False)
    pylab.show()

def check_col(user_feats, cols):
    for col in cols:
        s = user_feats[user_feats[col].astype(str).str.contains('‰')]
        if s.size > 0:
            print(col)


class Rec():
    def __init__(self, args):
        self.args = args
        self.data_dir = self.args.data_dir
        self.cache_dir = self.args.cache_dir
        self.model_dir = self.args.model_dir
        self.model_type = self.args.model_type
        self.csv_header = False
        # self.csv_header = True

    def user_feat(self,train_end_date):
        print('gen user feat %s'%train_end_date)
        dump_path = os.path.join(self.cache_dir, 'user_feat_%s.pkl'%train_end_date)
        if os.path.exists(dump_path):
            with open(dump_path, 'rb') as gf:
                user_feats = pickle.load(gf)
        else:
            dt = datetime.strptime(train_end_date,'%Y-%m-%d').strftime('%Y%m%d')
            user_feat_dir = os.path.join(self.data_dir, "al_kyc_tg_%s.csv" % dt)
            # user_feat_dir = os.path.join(self.data_dir, "value_customer_features_%s.csv" % dt)
            if not os.path.exists(user_feat_dir):
                return  None
            with open(user_feat_dir, 'rb') as gf:
                if self.csv_header:
                    user_feats = pd.read_csv(gf)
                else:
                    user_feats = pd.read_csv(gf,  header=None, sep='\001',names=['user_id','aum','historical_max_aum','balance_amount','frozen_amount','applying_amount','template_id','template_version_no','class_level','score','q1','q2','q3','q4','q5','q6','q7','q8','q9','q10','q11','q12','q13','q14','q15','q16','q17','q18','q19','q20','bank_3m_in_amt','bank_24m_in_amt','bank_3m_out_amt','bank_24m_out_amt','credit_3m_out_cnt','credit_24m_out_cnt','bank_3m_in_cnt','bank_24m_in_cnt','bank_3m_out_cnt','bank_24m_out_cnt','bank_3m_invest_cnt','bank_24m_invest_cnt','bank_3m_in_max_amt','bank_24m_in_max_amt','spend_amt_rn_rank','spend_cnt_rn_rank','income_per_month_predict','spend_amt_per_month','spend_cnt_per_month','spend_age','cons_tot_m3_num','cons_tot_m3_pay','cons_tot_m12_num','cons_tot_m12_pay','cons_max_m3_pay','cons_max_m12_pay','is_cdm','marriage_status_cd','education_cd','family_member_quantiny','industry','prof','life_cycle','age_range','cust_aum_flag','group_vip_level','group_vip_flag','toa_pa_act_assets_amt','toa_pa_act_debts_amt','posses_house_auto_flag','series_prod_type_count','pc_insu_vip_flag','pc_insu_vip_level','hold_auto_prod_flag','hold_moto_prod_flag','pc_prod_type_count','pnc_aum_flag','vehicle_loss_insured_value','vehicle_quantity','year_activity_level','quar_activity_level','year_trade_level','quar_trade_level','is_sx','elis_aum_flag2','vip_flag','wealth_score','invest_exp','risk_sensitive','invest_lately_12m_avg_aum','invest_lately_24m_max_aum','invest_lately_3m_b2c_amt','invest_lately_3m_b2c_cnt','invest_lately_3m_amt','invest_lately_3m_cnt','invest_lately_3m_avg_aum','invest_lately_3m_max_aum','inv_100w_cnt','inv_10w_cnt','inv_30w_cnt','inv_50w_cnt','inv_5w_cnt','invest_amt','invest_b2c_amt','invest_b2c_cnt','invest_cnt','invest_risklevel_unique','invest_most_min_invest_amt','invest_most_period_by_days','invest_most_rate','invest_most_risk_level','invest_total_max_amt','ml_model_level','risk_level','media1_audit_status','media2_audit_status','media3_audit_status'])

            mod = user_feats['education_cd'].mode()[0]
            user_feats['education_cd'].replace('‰', mod, inplace=True)
            mod = user_feats['marriage_status_cd'].mode()[0]
            user_feats['marriage_status_cd'].replace('@', mod, inplace=True)
            # subscribe_status = self.user_subscribe_status()
            # user_feats = pd.merge(user_feats, subscribe_status, how='left', on='user_id')
            # del (subscribe_status)
            # gc.collect()
            # audit_status = self.user_audit_status()
            # user_feats = pd.merge(user_feats, audit_status, how='left', on='user_id')
            # del (audit_status)
            # gc.collect()
            with open(dump_path, 'wb') as gf:
                pickle.dump(user_feats, gf)

        return user_feats

    def product_feat(self):
        print('gen product features')
        dump_path = os.path.join(self.cache_dir, 'product_feat.pkl')
        if os.path.exists(dump_path):
            with open(dump_path, 'rb') as gf:
                products = pickle.load(gf)
        else:
            product_dir = os.path.join(self.data_dir, "products.csv")
            with open(product_dir, 'rb') as gf:
                if self.csv_header:
                    products = pd.read_csv(gf)
                else:
                    products = pd.read_csv(gf,header=None,names=['product_id','product_category','item','product_price','invest_period_by_days'])
            products.dropna(inplace=True)
            products['product_id']=products['product_id'].astype(int)
            products['product_price']=products['product_price'].astype(float)
            # products.rename(columns={'id': 'product_id'}, inplace=True)
            products.loc[products['product_id'] == 147566049, 'invest_period_by_days'] = 28
            products.loc[products['product_id'] == 157269050, 'invest_period_by_days'] = 7
            products['product_price'].fillna(0.01,inplace=True)
            products.loc[products['product_price']<0.0]=0.0
            products['invest_period_by_days'].fillna(0,inplace=True)
            # products = products[(products['product_id']==147566049) | (products['product_id']==157269050)]
            # seg price
            amt_grp_names = ['less_1w', '1w', '5w', '10w', '30w', '50w', '100w']
            amt_bins = [ -0.01, 10000, 50000, 100000, 300000, 500000, 1000000, 10000000000]
            amt = pd.cut(products['product_price'], amt_bins, labels=amt_grp_names)
            amt.rename('price_group', inplace=True)
            # amt2 = pd.get_dummies(amt, prefix='amt')
            # seg period
            period_grp = ['1week', '1m', '2m', '3m', '6m', '9m', '12m', '15m', '18m', '24m', '30m', '36m']
            period_bins = [1, 7, 30, 60, 90, 180, 360, 450, 540, 720, 900, 1080, 2160]
            period = pd.cut(products['invest_period_by_days'], period_bins, labels=period_grp)
            period.rename('period_group', inplace=True)
            # period2 = pd.get_dummies(period, prefix='invest_period')
            products['product_category'] = products['product_category'].astype(str)
            product_category_one_hot = pd.get_dummies(products['product_category'], prefix='cat')
            # cmn_product_category_one_hot = pd.get_dummies(products['cmn_product_category'], prefix='cmn_cat')
            item_one_hot = pd.get_dummies(products['item'], prefix='item')
            products = pd.concat(
                [products, product_category_one_hot, item_one_hot, amt, period], axis=1)
            # products =  pd.concat([products, product_category_one_hot,cmn_product_category_one_hot,item_one_hot,amt,period,amt2,period2], axis=1)
            products['product_group'] = products['price_group'].astype(str)
            # products['product_group'] = products['price_group'].astype(str) + '_'  + products['item'].astype(str)
            # products.drop(['product_category', 'cmn_product_category', 'item', 'product_price', 'invest_period_by_days'],axis=1, inplace=True)
            # add product group cvr features
            with open(dump_path, 'wb') as gf:
                pickle.dump(products[['product_id','product_group','product_category','item','product_price','invest_period_by_days']], gf)

        return products

    def get_browse(self, start_date, end_date):
        print('get browse', start_date, end_date)
        spec_browse_data = os.path.join(self.data_dir, "browse.csv")
        dump_path = os.path.join(self.cache_dir, 'user_browse_%s_%s.pkl' % (start_date, end_date))
        if os.path.exists(dump_path):
            with open(dump_path, 'rb') as gf:
                browse = pickle.load(gf)
        else:
            with open(spec_browse_data, 'rb') as gf:
                if self.csv_header:
                    browse = pd.read_csv(gf)
                else:
                    browse = pd.read_csv(gf,header=None,names=['user_id','product_id','date','duraction'])
            browse.dropna(inplace=True)
            # browse['user_id'] = browse['user_id'].apply(lambda x:int(float(x)))
            browse['user_id'] = browse['user_id'].astype(int)
            # browse['date'] = browse['request_time'].apply(lambda x: x[:10])
            # del browse['request_time']
            browse = browse[(browse['date'] >= start_date) & (browse['date'] <= end_date)]
            products = self.product_feat()
            browse = pd.merge(browse, products[['product_id', 'product_group']], how='left', on='product_id')
            # browse = browse[(browse['product_id']==147566049) | (browse['product_id']==157269050)]
            with open(dump_path, 'wb') as gf:
                pickle.dump(browse, gf)

        return browse

    def user_browse_feature(self, start_date, end_date):
        print('gen user browse features', start_date, end_date)
        browse_feat = None
        dump_path = os.path.join(self.cache_dir, 'user_browse_feat_%s_%s.pkl' % (start_date, end_date))
        if os.path.exists(dump_path):
            with open(dump_path, 'rb') as gf:
                browse_feat = pickle.load(gf)
        else:
            interval_browse = self.get_browse(start_date, end_date)
            spans = [30, 15, 7, 3, 1]
            for span in spans:
                span_start_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=span - 1)
                span_start_date = span_start_date.strftime('%Y-%m-%d')
                print(span_start_date, end_date)
                if span_start_date < start_date:
                    print('warning pass date:', span_start_date, start_date)
                    span_start_date = start_date
                    # continue
                browse = interval_browse[
                    (interval_browse['date'] <= end_date) & (interval_browse['date'] >= span_start_date)]
                # browse times
                browse_product_times = browse.groupby(['user_id', 'product_group']).size().reset_index()
                browse_product_times.rename(columns={0: '%d_day_browse_id_times' % (span)}, inplace=True)
                # browse duration
                browse_product_duration = browse.groupby(['user_id', 'product_group'])['duraction'].sum().reset_index()
                browse_product_duration.rename(columns={'duraction': '%d_day_browse_id_duration' % (span)},
                                               inplace=True)
                # active days
                active_days = browse.groupby(['user_id', 'date']).size().reset_index()
                active_days = active_days.groupby('user_id').size().reset_index()
                active_days.rename(columns={0: '%d_day_user_active_days' % (span)}, inplace=True)
                active_duration = browse.groupby(['user_id'])['duraction'].sum().reset_index()
                active_duration.rename(columns={'duraction': '%d_day_user_active_duration' % (span)}, inplace=True)

                browse_feat_tmp = pd.merge(browse_product_times, browse_product_duration, how='left',
                                           on=['user_id', 'product_group'])
                browse_feat_tmp.fillna(0, inplace=True)
                # browse_feat_tmp = browse_feat_tmp[(browse_feat_tmp['product_group'].str.contains('100w'))]
                # browse_feat_tmp = browse_feat_tmp[(browse_feat_tmp['product_group'].str.contains('100w_C01')) | (browse_feat_tmp['product_group'].str.contains('100w_A04'))]
                browse_feat_tmp = browse_feat_tmp.set_index(['user_id', 'product_group']).unstack(fill_value=0).reset_index()
                browse_feat_tmp.columns = ['_'.join(t) for t in browse_feat_tmp.columns.values]
                browse_feat_tmp.rename(columns={'user_id_': 'user_id'}, inplace=True)

                browse_feat_tmp = pd.merge(browse_feat_tmp, active_days, how='left', on='user_id')
                browse_feat_tmp = pd.merge(browse_feat_tmp, active_duration, how='left', on='user_id')
                browse_feat_tmp.fillna(0,inplace=True)

                if browse_feat is not None:
                    browse_feat = pd.merge(browse_feat, browse_feat_tmp, how='left', on='user_id')
                else:
                    browse_feat = browse_feat_tmp

            print("browse feature columns:")
            print(browse_feat.columns.values)
            browse_feat.fillna(0,inplace=True)
            with open(dump_path, 'wb') as gf:
                pickle.dump(browse_feat, gf)

        return browse_feat

    def get_collection(self, start_date, end_date):
        print('get actual collection', start_date, end_date)
        collect_amt_data = os.path.join(self.data_dir, "collect_amt.csv")
        dump_path = os.path.join(self.cache_dir, 'user_collection_%s_%s.pkl' % (start_date, end_date))
        if os.path.exists(dump_path):
            with open(dump_path, 'rb') as gf:
                collection = pickle.load(gf)
        else:
            with open(collect_amt_data, 'rb') as gf:
                if self.csv_header:
                    collection = pd.read_csv(gf)
                else:
                    collection = pd.read_csv(gf,header=None,names=['user_id','actual_collection_amt','actual_collection_time'])

            collection.dropna(inplace=True)
            collection['user_id'] = collection['user_id'].astype(int)
            collection['date'] = collection['actual_collection_time'].apply(lambda x: x[:10])
            del collection['actual_collection_time']
            collection = collection[(collection['date'] >= start_date) & (collection['date'] <= end_date)]
            with open(dump_path, 'wb') as gf:
                pickle.dump(collection, gf)

        return collection

    def user_collection_feature(self, start_date, end_date):
        print('gen user collection features', start_date, end_date)
        collection_feat = None
        dump_path = os.path.join(self.cache_dir, 'user_collection_feat_%s_%s.pkl' % (start_date, end_date))
        if os.path.exists(dump_path):
            with open(dump_path, 'rb') as gf:
                collection_feat = pickle.load(gf)
        else:
            interval_collection = self.get_collection(start_date, end_date)
            spans = [30, 15, 7, 3, 1]
            for span in spans:
                span_start_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=span - 1)
                span_start_date = span_start_date.strftime('%Y-%m-%d')
                print(span_start_date, end_date)
                if span_start_date < start_date:
                    print('warning pass date:', span_start_date, start_date)
                    span_start_date = start_date
                    # continue
                collection = interval_collection[
                    (interval_collection['date'] <= end_date) & (interval_collection['date'] >= span_start_date)]
                # browse times
                collect_amt = collection.groupby('user_id')['actual_collection_amt'].sum().reset_index()
                collect_amt.rename(columns={'actual_collection_amt': '%d_day_collect_amt' % (span)}, inplace=True)

                if collection_feat is not None:
                    collection_feat = pd.merge(collection_feat, collect_amt, how='left', on='user_id')
                else:
                    collection_feat = collect_amt

            collection_feat.fillna(0, inplace=True)
            with open(dump_path, 'wb') as gf:
                pickle.dump(collection_feat, gf)

        return collection_feat

    def user_audit_status(self):
        print('get user audit status')
        dump_path = os.path.join(self.cache_dir, 'user_audit_status.pkl')
        if os.path.exists(dump_path):
            with open(dump_path, 'rb') as gf:
                audit_status = pickle.load(gf)
        else:
            audit_data = os.path.join(self.data_dir, "audit_status.csv")
            with open(audit_data, 'rb') as gf:
                if self.csv_header:
                    audit_status = pd.read_csv(gf)
                else:
                    audit_status = pd.read_csv(gf,header=None,names=['user_id','media1_audit_status','media2_audit_status','media3_audit_status'])
                audit_status['user_id'] = audit_status['user_id'].astype(int)
            with open(dump_path, 'wb') as gf:
                pickle.dump(audit_status, gf)

        return audit_status

    def user_subscribe_status(self):
        print('get user subscribe status')
        dump_path = os.path.join(self.cache_dir, 'user_subscribe_status.pkl')
        if os.path.exists(dump_path):
            with open(dump_path, 'rb') as gf:
                subscribe_status = pickle.load(gf)
        else:
            audit_data = os.path.join(self.data_dir, "subscribe_status.csv")
            with open(audit_data, 'rb') as gf:
                if self.csv_header:
                    subscribe_status = pd.read_csv(gf)
                else:
                    subscribe_status = pd.read_csv(gf,header=None,names=['user_id','is_email_unsubscribe','is_sms_unsubscribe'])
                subscribe_status['user_id'] = subscribe_status['user_id'].astype(int)
            with open(dump_path, 'wb') as gf:
                pickle.dump(subscribe_status, gf)

        del (subscribe_status['stat_date'])
        return subscribe_status

    def get_invest(self, start_date, end_date):
        print('get invests', start_date, end_date)
        dump_path = os.path.join(self.cache_dir, 'user_invest_%s_%s.pkl' % (start_date, end_date))
        if os.path.exists(dump_path):
            with open(dump_path, 'rb') as gf:
                invest = pickle.load(gf)
        else:
            products = self.product_feat()
            spec_invest_data = os.path.join(self.data_dir, "invests.csv")
            with open(spec_invest_data, 'rb') as gf:
                if self.csv_header:
                    invest = pd.read_csv(gf)
                else:
                    invest = pd.read_csv(gf,header=None,names=['user_id','product_id','investment_amount','invest_dt'])
            invest.dropna(inplace=True)
            # invest.rename(columns={'loaner_user_id': 'user_id'}, inplace=True)
            # invest['date']=invest['request_time'].apply(lambda x:x[:10])
            invest = invest[(invest['invest_dt'] >= start_date) & (invest['invest_dt'] <= end_date)]
            invest = pd.merge(invest, products[['product_id', 'product_group']], how='left', on='product_id')
            #        invest = invest[(invest['product_id']==147566049) |(invest['product_id']==157269050)]
            with open(dump_path, 'wb') as gf:
                pickle.dump(invest, gf)

        return invest

    def user_invest_feature(self, start_date, end_date):
        print('gen user invest features', start_date, end_date)
        invest_feat = None
        dump_path = os.path.join(self.cache_dir, 'user_invest_feat_%s_%s.pkl' % (start_date, end_date))
        if os.path.exists(dump_path):
            with open(dump_path, 'rb') as gf:
                invest_feat = pickle.load(gf)
        else:
            interval_invest = self.get_invest(start_date, end_date)
            spans = [30, 15, 7, 3, 1]
            for span in spans:
                span_start_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=span - 1)
                span_start_date = span_start_date.strftime('%Y-%m-%d')
                print(span_start_date, end_date)
                if span_start_date < start_date:
                    print('pass:', span_start_date, start_date)
                    span_start_date = start_date
                    # continue
                invest = interval_invest[
                    (interval_invest['invest_dt'] <= end_date) & (interval_invest['invest_dt'] >= span_start_date)]
                # invest times
                #            invest_product_times = invest.groupby(['user_id', 'product_id']).size().reset_index()
                invest_product_times = invest.groupby(['user_id', 'product_group']).size().reset_index()
                invest_product_times.rename(columns={0: '%d_day_invest_id_times' % (span)}, inplace=True)
                # invest product id amount
                invest_product_amt = invest.groupby(['user_id', 'product_group'])[
                    'investment_amount'].sum().reset_index()
                invest_product_amt.rename(columns={'investment_amount': '%d_day_invest_id_amt' % (span)}, inplace=True)
                # invest times
                invest_times = invest.groupby(['user_id', 'invest_dt']).size().reset_index()
                invest_times.rename(columns={0: 'invest_times'}, inplace=True)
                invest_times = invest_times.groupby(['user_id'])['invest_times'].sum().reset_index()
                invest_times.rename(columns={'invest_times': '%d_day_invest_times' % (span)}, inplace=True)
                # invest amt
                invest_amt = invest.groupby(['user_id'])['investment_amount'].sum().reset_index()
                invest_amt.rename(columns={'investment_amount': '%d_day_invest_amt' % (span)}, inplace=True)


                invest_feat_tmp = pd.merge(invest_product_times, invest_product_amt, how='left',
                                           on=['user_id', 'product_group'])


                invest_feat_tmp.fillna(0, inplace=True)
                # invest_feat_tmp = invest_feat_tmp[(invest_feat_tmp['product_group'].str.contains('100w'))]
                # invest_feat_tmp = invest_feat_tmp[(invest_feat_tmp['product_group'].str.contains('100w_C01')) | (invest_feat_tmp['product_group'].str.contains('100w_A04'))]
                invest_feat_tmp = invest_feat_tmp.set_index(['user_id', 'product_group']).unstack(fill_value=0).reset_index()
                invest_feat_tmp.columns = ['_'.join(t) for t in invest_feat_tmp.columns.values]
                invest_feat_tmp.rename(columns={'user_id_': 'user_id'}, inplace=True)

                invest_feat_tmp = pd.merge(invest_feat_tmp, invest_times, how='left', on='user_id')
                invest_feat_tmp = pd.merge(invest_feat_tmp, invest_amt, how='left', on='user_id')
                invest_feat_tmp.fillna(0,inplace=True)
                if invest_feat is not None:
                    invest_feat = pd.merge(invest_feat, invest_feat_tmp, how='left', on='user_id')
                else:
                    invest_feat = invest_feat_tmp

            print("user invest features:")
            print(invest_feat.columns.values)
            invest_feat.fillna(0, inplace=True)
            with open(dump_path, 'wb') as gf:
                pickle.dump(invest_feat, gf)

        return invest_feat

    def gen_labels(self, start_date, end_date):
        print('gen lables', start_date, end_date)
        dump_path = os.path.join(self.cache_dir, 'labels_%s_%s.pkl' % (start_date, end_date))
        if os.path.exists(dump_path):
            with open(dump_path, 'rb') as gf:
                invests = pickle.load(gf)
        else:
            invests = self.get_invest(start_date, end_date)
            invests = invests.groupby(['user_id', 'product_group'], as_index=False).sum()
            invests['label'] = 1
            invests= invests[(invests['product_group'].str.contains('100w'))]
            invests.reset_index(drop=True,inplace=True)
            with open(dump_path, 'wb') as gf:
                pickle.dump(invests[['user_id', 'label']], gf)

        return invests[['user_id', 'label']]

    def get_collect_plan(self, start_date, end_date):
        print('get collect plan detail', start_date, end_date)
        dump_path = os.path.join(self.cache_dir, 'user_collect_plan_%s_%s.pkl' % (start_date, end_date))
        if os.path.exists(dump_path):
            with open(dump_path, 'rb') as gf:
                collect_amt = pickle.load(gf)
        else:
            collect_plan_data = os.path.join(self.data_dir, "al_collect_plan.csv")
            with open(collect_plan_data, 'rb') as gf:
                if self.csv_header:
                    collect_amt = pd.read_csv(gf)
                else:
                    collect_amt = pd.read_csv(gf,header=None,names=['user_id','future_collect_amount','collection_date'])
            collect_amt.dropna(inplace=True)
            collect_amt = collect_amt[(collect_amt['collection_date'] >= start_date) & (collect_amt['collection_date'] <= end_date)]
            with open(dump_path, 'wb') as gf:
                pickle.dump(collect_amt, gf)

        return collect_amt

    def user_collect_plan_feat(self, start_date, end_date):
        print('gen user future collect amt:', start_date, end_date)
        dump_path = os.path.join(self.cache_dir, 'user_collect_plan_feat_%s_%s.pkl' % (start_date, end_date))
        if os.path.exists(dump_path):
            with open(dump_path, 'rb') as gf:
                collect_amt = pickle.load(gf)
        else:
            collect_amt = self.get_collect_plan(start_date, end_date)
            collect_amt = collect_amt.groupby('user_id')['future_collect_amount'].sum().reset_index()
            with open(dump_path, 'wb') as gf:
                pickle.dump(collect_amt, gf)

        return collect_amt

    def make_million_train_set(self, train_start_date, train_end_date, test_start_date, test_end_date):
        print('make train set', train_start_date, train_end_date, test_start_date, test_end_date)
        dump_path = os.path.join(self.cache_dir, 'million_train_set_%s_%s_%s_%s.pkl' % (
            train_start_date, train_end_date, test_start_date, test_end_date))
        if os.path.exists(dump_path):
            return
            # with open(dump_path, 'rb') as gf:
            #     train_set = pickle.load(gf)
        else:
            browse_feat = self.user_browse_feature(train_start_date, train_end_date)
            # invest_feat = self.user_invest_feature(train_start_date, train_end_date)
            print(browse_feat.columns.values)
            # print(invest_feat.columns.values)
            # train_set = pd.merge(browse_feat, invest_feat, how='outer', on='user_id')
            # train_set.fillna(0,inplace=True)
            # pickle.dump(train_set, open(os.path.join(self.cache_dir,'browse_invest_{0}_{1}.pkl'.format(train_start_date,train_end_date)),'wb'))
            # del (browse_feat)
            # del (invest_feat)
            # gc.collect()
            user_feat = self.user_feat(train_end_date)
            if user_feat is None:
                return
            print(user_feat.columns.values)
            train_set = pd.merge(user_feat,browse_feat, how='left', on='user_id')
            del browse_feat
            del (user_feat)
            gc.collect()
            collect_feat = self.user_collect_plan_feat(test_start_date, test_end_date)
            print(collect_feat.columns.values)
            train_set = pd.merge(train_set, collect_feat, how='left', on='user_id')
            del (collect_feat)
            gc.collect()
            labels = self.gen_labels(test_start_date, test_end_date)
            print(labels.columns.values)
            train_set = pd.merge(train_set, labels, how='left', on='user_id')
            # pickle.dump(train_set, open(
            #     os.path.join(self.cache_dir, 'trainset_labels_{0}_{1}.pkl'.format(train_start_date, train_end_date)),
            #     'wb'))
            del (labels)
            gc.collect()
            train_set['label'].fillna(0, inplace=True)
            num_cols = [i for i in train_set.columns.values if
                        (('future_collect_amount' in i) or ('_times' in i) or ('_day_' in i) or ('duration' in i) or ('_id_' in i) or ('days' in i))]
            train_set[num_cols] = train_set[num_cols].fillna(0)
            # train_set = train_set[train_set['product_group'].str.contains("100w")]
            with open(dump_path, 'wb') as gf:
                pickle.dump(train_set, gf)
            del (train_set)
            gc.collect()

            # labels = train_set['label'].copy()
            # del train_set['user_id']
            # del train_set['product_group']
            # del train_set['label']
            #
            # print('train set cols:')
            # print(train_set.columns)
            # return train_set, labels

    def sampling_data(self,start_date, end_date, window=30, step=5):
        res = None
        whole = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
        l = int(whole.days / step)
        print("merge train set with step", step)
        for i in tqdm(range(l), desc='train matrix:'):
            s = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=i * step)
            train_start_date = datetime.strftime(s, '%Y-%m-%d')
            s = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=(i * step) + window - 1)
            train_end_date = datetime.strftime(s, '%Y-%m-%d')
            e = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=(i * step) + window)
            act_start_date = datetime.strftime(e, '%Y-%m-%d')
            e = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=(i * step) + 2 * window - 1)
            act_end_date = datetime.strftime(e, '%Y-%m-%d')
            if act_end_date <= end_date:
                print('sampling set', train_start_date, train_end_date, act_start_date, act_end_date)
                dump_path = os.path.join(self.cache_dir, 'sampled_million_train_set_%s_%s_%s_%s.pkl' % (
                    train_start_date, train_end_date, act_start_date, act_end_date))
                if os.path.exists(dump_path):
                    train_set = pickle.load(open(dump_path, 'rb'))
                else:
                    dump_path = os.path.join(self.cache_dir, 'million_train_set_%s_%s_%s_%s.pkl' % (train_start_date,train_end_date,act_start_date, act_end_date))
                    if not os.path.exists(dump_path):
                        continue
                    train_set = pickle.load(open(dump_path, 'rb'))
                    print("train set n/y:")
                    print(train_set.groupby('label').size())
                    print("sampling negative samples:")
                    # rows = random.sample(train_data[train_data['label']==0].index,10)
                    train_data_n = train_set[train_set['label'] == 0]
                    train_data_y = train_set[train_set['label'] == 1]
                    sample_rate = 0.2
                    # sample_rate = 100.0/(train_data_n.shape[0]/train_data_y.shape[1])
                    train_data_n = train_data_n.sample(frac=sample_rate, random_state=999)
                    del train_set
                    gc.collect()
                    train_set = pd.concat([train_data_n, train_data_y], axis=0, ignore_index=True)
                    dump_path = os.path.join(self.cache_dir, 'sampled_million_train_set_%s_%s_%s_%s.pkl' % (
                    train_start_date, train_end_date, act_start_date, act_end_date))
                    with open(dump_path, 'wb') as gf:
                        pickle.dump(train_set, gf)
                    del train_data_n
                    del train_data_y
                    gc.collect()
                    print("train set n/y:")
                    print(train_set.groupby('label').size())

                if res is not None:
                    res = pd.concat([res, train_set], axis=0, ignore_index=True)
                    del train_set
                    gc.collect()
                else:
                    res = train_set

        dump_path= os.path.join(self.cache_dir, 'sampled_set_%s_%s.pkl' % (start_date,end_date))
        with open(dump_path, 'wb') as gf:
            pickle.dump(res, gf)


    def make_train_libsvm_matrix(self, train_start_date, train_end_date, act_start_date, act_end_date):
        dump_path = os.path.join(self.cache_dir, 'train_matrix_%s_%s_%s_%s.libsvm' % (
            train_start_date, train_end_date, act_start_date, act_end_date))
        if os.path.exists(dump_path):
            return
        print("making train libsvm matrix:{0}-{1}-{2}-{3}".format(train_start_date, train_end_date, act_start_date,
                                                                  act_end_date))
        feat_sel_path = os.path.join(self.cache_dir, 'feat_sel.pkl')
        feat_encoder_path = os.path.join(self.cache_dir, 'feat_encoder.pkl')

        X_train, y_train = self.make_train_set(train_start_date, train_end_date, act_start_date, act_end_date)
        print("train set y=0:{0}".format(y_train[y_train == 0].shape[0]))
        print("train set y=1:{0}".format(y_train[y_train == 1].shape[0]))

        if os.path.exists(feat_sel_path) and os.path.exists(feat_encoder_path):
            with open(feat_sel_path, 'rb') as gf:
                sl = pickle.load(gf)
            with open(feat_encoder_path, 'rb') as gf:
                bfp = pickle.load(gf)
        else:
            sl = FeatureSelection(self.args)
            sl.fit(X_train, y_train)
            bfp = FeatureEncoder(self.args, sl.numerical_cols, sl.categorical_cols)
            bfp.fit(X_train[sl.selected_cols], y_train, dump_path)

        sl.transform(X_train)
        bfp.transform(X_train[sl.selected_cols], y_train, dump_path)

        if os.path.exists(feat_sel_path) and os.path.exists(feat_encoder_path):
            pass
        else:
            with open(feat_sel_path, 'wb') as gf:
                pickle.dump(sl, gf)
            with open(feat_encoder_path, 'wb') as gf:
                pickle.dump(bfp, gf)

    def make_sliding_train_matrix(self, start_date, end_date, window=30, step=5):
        whole = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
        l = int(whole.days / step)
        print("gen train matrix with step", step)
        for i in tqdm(range(l), desc='train matrix:'):
            s = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=i * step)
            train_start_date = datetime.strftime(s, '%Y-%m-%d')
            s = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=(i * step) + window - 1)
            train_end_date = datetime.strftime(s, '%Y-%m-%d')
            e = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=(i * step) + window)
            act_start_date = datetime.strftime(e, '%Y-%m-%d')
            e = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=(i * step) + 2 * window - 1)
            act_end_date = datetime.strftime(e, '%Y-%m-%d')
            if act_end_date <= end_date:
                self.make_train_libsvm_matrix(train_start_date, train_end_date, act_start_date, act_end_date)

    def make_sliding_train_set(self, start_date, end_date, window=30, step=5):
        whole = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
        l = int(whole.days / step)
        print("gen train matrix with step", step)
        for i in tqdm(range(l), desc='train matrix:'):
            s = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=i * step)
            train_start_date = datetime.strftime(s, '%Y-%m-%d')
            s = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=(i * step) + window - 1)
            train_end_date = datetime.strftime(s, '%Y-%m-%d')
            e = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=(i * step) + window)
            act_start_date = datetime.strftime(e, '%Y-%m-%d')
            e = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=(i * step) + 2 * window - 1)
            act_end_date = datetime.strftime(e, '%Y-%m-%d')
            if act_end_date <= end_date:
                # self.make_million_only_data(train_start_date, train_end_date, act_start_date, act_end_date)
                self.make_million_train_set(train_start_date, train_end_date, act_start_date, act_end_date)

    def merge_sliding_train_set(self, start_date, end_date, window=30, step=5):
        res = None
        dump_path = os.path.join(self.cache_dir, "merge_million_train_set_%s_%s_window%s_step%s.pkl" % (start_date, end_date, window, step))
        if os.path.exists(dump_path):
            return
            # with open(dump_path, 'rb') as gf:
            #     res = pickle.load(gf)
        else:
            whole = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
            l = int(whole.days / step)
            print("merge train set with step", step)
            for i in tqdm(range(l), desc='train matrix:'):
                s = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=i * step)
                train_start_date = datetime.strftime(s, '%Y-%m-%d')
                s = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=(i * step) + window - 1)
                train_end_date = datetime.strftime(s, '%Y-%m-%d')
                e = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=(i * step) + window)
                act_start_date = datetime.strftime(e, '%Y-%m-%d')
                e = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=(i * step) + 2 * window - 1)
                act_end_date = datetime.strftime(e, '%Y-%m-%d')
                if act_end_date <= end_date:
                    print('make train set', train_start_date, train_end_date, act_start_date, act_end_date)
                    dump_path = os.path.join(self.cache_dir, 'million_train_set_%s_%s_%s_%s.pkl' % (
                        train_start_date, train_end_date, act_start_date, act_end_date))
                    if not os.path.exists(dump_path):
                        continue
                    train_set = pickle.load(open(dump_path, 'rb'))
                    if res is not None:
                        tmp = pd.concat([res, train_set], axis=0,ignore_index=True)
                        del (res)
                        del (train_set)
                        gc.collect()
                        res = tmp
                    else:
                        res = train_set
            # this should be handled case by case
            num_cols = [i for i in res.columns.values if
                        (('browse_id_times' in i) or ('browse_id_amt' in i) or ('invest_id_times' in i) or ('invest_id_amt' in i) or ('duraction' in i) or ('collect_amt' in i))]
            res[num_cols].fillna(0,inplace=True)
            dump_path = os.path.join(self.cache_dir, "merge_million_train_set_%s_%s_window%s_step%s.pkl" % (start_date, end_date, window, step))
            if res is not None:
                print("dump train matrix to %s"%dump_path)
                with open(dump_path, 'wb') as gf:
                    pickle.dump(res, gf,protocol=4)


    def merge_label_user_ids(self, start_date, end_date, window=30, step=5):
        res = None
        dump_path = os.path.join(self.cache_dir, "merge_label_userids_%s_%s_window%s_step%s.csv" % (start_date, end_date, window, step))
        if os.path.exists(dump_path):
            return
            # with open(dump_path, 'rb') as gf:
            #     res = pickle.load(gf)
        else:
            whole = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
            l = int(whole.days / step)
            # print("merge train set with step", step)
            for i in tqdm(range(l), desc='train matrix:'):
                s = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=i * step)
                train_start_date = datetime.strftime(s, '%Y-%m-%d')
                s = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=(i * step) + window - 1)
                train_end_date = datetime.strftime(s, '%Y-%m-%d')
                e = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=(i * step) + window)
                act_start_date = datetime.strftime(e, '%Y-%m-%d')
                e = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=(i * step) + 2 * window - 1)
                act_end_date = datetime.strftime(e, '%Y-%m-%d')
                if act_end_date <= end_date:
                    print('make train set', train_start_date, train_end_date, act_start_date, act_end_date)
                    dump_path = os.path.join(self.cache_dir, 'labels_%s_%s.pkl' % (act_start_date, act_end_date))
                    print("merge file %s"%dump_path)
                    if not os.path.exists(dump_path):
                        continue
                    train_set = pickle.load(open(dump_path, 'rb'))
                    if res is not None:
                        tmp = pd.concat([res, train_set], axis=0,ignore_index=True)
                        del (res)
                        del (train_set)
                        gc.collect()
                        res = tmp
                    else:
                        res = train_set
            res.drop_duplicates(subset='user_id',keep='first',inplace=True)
            res['potential_score']=0
            # res.rename(columns={'': ''}, inplace=True)
            dump_path = os.path.join(self.cache_dir, "merge_label_userids_%s_%s_window%s_step%s.csv" % (start_date, end_date, window, step))
            if res is not None:
                print("dump mergeed label userids to %s"%dump_path)
                with open(dump_path, 'w') as gf:
                    res[['user_id','potential_score']].to_csv(gf,header=True,index=False)

    def train_xgb_month(self):
        print("=" * 60)
        print(self.cache_dir)
        start_time = time.time()
        data_path = os.path.join(self.cache_dir, "train_matrix.csv")
        # data_path = os.path.join(self.cache_dir, "train_matrix.csv")
        with open(data_path, 'r') as gf:
            data = pd.read_csv(gf)

        # cols = [col for col in data.columns.values if col not in ['label']]
        # ex_cols = [col for col in data.columns.values if (('is_' in col) or ('flag' in col))]
        # cols = list(set(data.columns.values)-set(ex_cols))
        # ('future_collect' in col) or
        # or (col.startswith('q'))
        # cols=['invest_amt','invest_cnt','invest_most_min_invest_amt','historical_max_aum','aum','invest_total_max_amt','inv_5w_cnt','invest_lately_12m_avg_aum','invest_b2c_cnt','invest_b2c_amt','invest_lately_24m_max_aum','invest_lately_3m_b2c_amt','inv_50w_cnt','inv_30w_cnt','inv_100w_cnt','invest_lately_3m_max_aum','invest_lately_3m_avg_aum','invest_lately_3m_cnt','invest_lately_3m_amt','invest_lately_3m_b2c_cnt','inv_10w_cnt']
        cols = [col for col in data.columns.values if ((('1_day' not in col) and  ('3_day' not in col) and  ('label' not in col) and ('is_' not in col) and (
                    'flag' not in col) and ('user_id' not in col) and ('invest_id' not in col)) and (
                                                                   ('future_collect' in col) or (col.startswith('q')) or ('amt' in col) or (
                                                                               'cnt' in col) or ('aum' in col) or (
                                                                               'rank' in col) or ('income' in col) or (
                                                                               'media1' in col) or (
                                                                               'collect_amt' in col) or (
                                                                               'browse_id' in col) or (
                                                                               'duration' in col) or (
                                                                               'times' in col) or (
                                                                               'user_active_days' in col)))]
        # cols = ['aum','historical_max_aum','balance_amount','frozen_amount','applying_amount','invest_lately_12m_avg_aum','invest_lately_24m_max_aum','invest_lately_3m_b2c_amt','invest_lately_3m_b2c_cnt','invest_lately_3m_amt','invest_lately_3m_cnt','invest_lately_3m_avg_aum','invest_lately_3m_max_aum','inv_100w_cnt','inv_10w_cnt','inv_30w_cnt','inv_50w_cnt','inv_5w_cnt','invest_amt','invest_b2c_amt','invest_b2c_cnt','invest_cnt','invest_risklevel_unique','invest_most_min_invest_amt','invest_most_period_by_days','invest_most_rate','invest_most_risk_level','invest_total_max_amt']
        # cols =list(set(cls) + set(['aum','historical_max_aum','balance_amount','invest_lately_12m_avg_aum','invest_lately_24m_max_aum','invest_lately_3m_b2c_amt','invest_lately_3m_b2c_cnt','invest_lately_3m_amt','invest_lately_3m_cnt','invest_lately_3m_avg_aum','invest_lately_3m_max_aum','inv_100w_cnt','inv_10w_cnt','inv_30w_cnt','inv_50w_cnt','inv_5w_cnt','invest_amt','invest_b2c_amt','invest_b2c_cnt','invest_cnt','invest_risklevel_unique','invest_most_min_invest_amt','invest_most_period_by_days','invest_most_rate','invest_most_risk_level','invest_total_max_amt']))
        print("{0} input features:".format(len(cols)))
        print(cols)
        y = data['label']
        scale_pos_weight = (y[y == 0].shape[0]) * 1.0 / (y[y == 1].shape[0])
        print("scale_pos_weight:", scale_pos_weight)
        X_train = data[cols]
        y_train = data['label']
        # X_train, X_test, y_train, y_test = train_test_split(data.loc[:, cols], y, test_size=0.2, random_state=999,
        #                                                     stratify=y)
        threads = int(0.9 * multiprocessing.cpu_count())
        lgbm = xgb.XGBClassifier(n_estimators=30, learning_rate=0.3, max_depth=10, min_child_weight=6, gamma=0.3,
                                subsample=0.7,
                                colsample_bytree=0.7, objective='rank:pairwise', nthread=threads,
                                scale_pos_weight=scale_pos_weight, reg_alpha=1e-05, reg_lambda=1, seed=27)
        print('training set...')
        print(y_train.value_counts())
        lgbm.fit(X_train, y_train)
        print('training...')
        print('[{}] Train lightgbm completed'.format(time.time() - start_time))
        del data
        model_version = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        dump_model_path = os.path.join(self.model_dir, 'lgbm_model_%s.pkl' % model_version)
        with open(dump_model_path, 'wb') as gf:
            pickle.dump(lgbm, gf)

        print("Features importance...")
        feat_imp = pd.Series(lgbm.get_booster().get_score()).sort_values(ascending=False)
        # feat_imp = pd.Series(gbm.booster().get_fscore()).sort_values(ascending=False)
        with pd.option_context('display.max_rows',None,'display.max_columns',3):
            print(feat_imp)

        print('predicting...')
        # test set
        for i in range(0, 7):
            data_path = os.path.join(self.cache_dir, "test_matrix%s.csv" % i)
            if not os.path.exists(data_path):
                continue
            with open(data_path, 'r') as gf:
                data = pd.read_csv(gf)

            X_test = data[cols]
            y_test = data['label']
            print('test set %s...' % i)
            print(y_test.value_counts())
            y_pre = lgbm.predict(X_test)
            y_pro = lgbm.predict_proba(X_test)[:, 1]
            #0 is the final test data
            if i==0:
                pass
            else:
                print("xgboost model Test AUC Score: {0}".format(roc_auc_score(y_test, y_pro)))
                print("xgboost model Test Precision: {0}".format(precision_score(y_test, y_pre)))
                print("xgboost model Test   Recall : {0}".format(recall_score(y_test, y_pre)))
                print("xgboost model Test F1 Score: {0}".format(f1_score(y_test, y_pre)))
                print("xgboost model Test AUC of PR-curve: {0}".format(average_precision_score(y_test, y_pro)))
                print("xgboost model Test logloss: {0}".format(log_loss(y_test, y_pro)))
                print("xgboost Test confusion_matrix :")
                print(confusion_matrix(y_test, y_pre))
            # print(y_pro)
            data['potential_score'] = y_pro
            res = data[['user_id','potential_score']].sort_values(by='potential_score', ascending=False)
            # print(res)
            dump_path = os.path.join(self.cache_dir,  'xgb_potential_results_%s.csv'%i)
            with open(dump_path, 'w') as gf:
                res.to_csv(gf,index=False)


    def train_lgb_month(self):
        print("=" * 60)
        print(self.cache_dir)
        start_time = time.time()
        data_path = os.path.join(self.cache_dir, "train_matrix.csv")
        with open(data_path, 'r') as gf:
            data = pd.read_csv(gf)

        #cols = [col for col in data.columns.values if col not in ['label']]
        # ex_cols = [col for col in data.columns.values if (('is_' in col) or ('flag' in col))]
        # cols = list(set(data.columns.values)-set(ex_cols))
        # ('future_collect' in col) or
        cols = [col for col in data.columns.values if ((('label' not in col) and ('is_' not in col) and (
                    'flag' not in col) and ('user_id' not in col) and ('invest_id' not in col)) and (
                                                                   ('future_collect' in col) or (col.startswith('q')) or ('amt' in col) or (
                                                                               'cnt' in col) or ('aum' in col) or (
                                                                               'rank' in col) or ('income' in col) or (
                                                                               'media1' in col) or (
                                                                               'collect_amt' in col) or (
                                                                               'browse_id' in col) or (
                                                                               'duration' in col) or (
                                                                               'times' in col) or (
                                                                               'user_active_days' in col)))]
        # cols=['invest_amt','invest_cnt','invest_most_min_invest_amt','historical_max_aum','aum','invest_total_max_amt','inv_5w_cnt','invest_lately_12m_avg_aum','invest_b2c_cnt','invest_b2c_amt','invest_lately_24m_max_aum','invest_lately_3m_b2c_amt','inv_50w_cnt','inv_30w_cnt','inv_100w_cnt','invest_lately_3m_max_aum','invest_lately_3m_avg_aum','invest_lately_3m_cnt','invest_lately_3m_amt','invest_lately_3m_b2c_cnt','inv_10w_cnt']
        # cols = [col for col in data.columns.values if ((('label' not in col) and ('is_' not in col) and ('flag' not in col) and ('user_id' not in col) and ('invest_id' not in col)) and (('future_collect' in col) or (col.startswith('q')) or  ('amt' in col) or ('cnt' in col) or ('aum' in col)  or ('rank' in col) or ('income' in col)  or ('media1' in col) or ('collect_amt' in col) or ('browse_id' in col)  or ('duration' in col) or ('times' in col) or ('user_active_days' in col)))]
        # cols = ['aum','historical_max_aum','balance_amount','frozen_amount','applying_amount','invest_lately_12m_avg_aum','invest_lately_24m_max_aum','invest_lately_3m_b2c_amt','invest_lately_3m_b2c_cnt','invest_lately_3m_amt','invest_lately_3m_cnt','invest_lately_3m_avg_aum','invest_lately_3m_max_aum','inv_100w_cnt','inv_10w_cnt','inv_30w_cnt','inv_50w_cnt','inv_5w_cnt','invest_amt','invest_b2c_amt','invest_b2c_cnt','invest_cnt','invest_risklevel_unique','invest_most_min_invest_amt','invest_most_period_by_days','invest_most_rate','invest_most_risk_level','invest_total_max_amt']
        # cols =list(set(cls) + set(['aum','historical_max_aum','balance_amount','invest_lately_12m_avg_aum','invest_lately_24m_max_aum','invest_lately_3m_b2c_amt','invest_lately_3m_b2c_cnt','invest_lately_3m_amt','invest_lately_3m_cnt','invest_lately_3m_avg_aum','invest_lately_3m_max_aum','inv_100w_cnt','inv_10w_cnt','inv_30w_cnt','inv_50w_cnt','inv_5w_cnt','invest_amt','invest_b2c_amt','invest_b2c_cnt','invest_cnt','invest_risklevel_unique','invest_most_min_invest_amt','invest_most_period_by_days','invest_most_rate','invest_most_risk_level','invest_total_max_amt']))
        print("{0} input features:".format(len(cols)))
        print(cols)
        y = data['label']
        # scale_pos_weight = 1000
        scale_pos_weight = (y[y == 0].shape[0]) * 1.0 / (y[y == 1].shape[0])
        print("scale_pos_weight:",scale_pos_weight)
        X_train = data[cols]
        y_train = data['label']
        # X_train, X_test, y_train, y_test = train_test_split(data.loc[:, cols], y, test_size=0.2, random_state=999,
        #                                                     stratify=y)
        # lgbm = RandomForestClassifier()
        lgbm = lgb.LGBMClassifier(boosting_type='gbdt', max_depth=10, learning_rate=0.3, n_estimators=30,
                                  scale_pos_weight=scale_pos_weight, min_child_weight=1, subsample=0.7,
                                  colsample_bytree=0.7,
                                  reg_alpha=1e-05, reg_lambda=1)
        print('training set...')
        print(y_train.value_counts())
        lgbm.fit(X_train, y_train)
        print('training...')
        print('[{}] Train lightgbm completed'.format(time.time() - start_time))
        del data
        model_version = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        dump_model_path = os.path.join(self.model_dir,'lgbm_model_%s.pkl'%model_version)
        with open(dump_model_path, 'wb') as gf:
            pickle.dump(lgbm, gf)

        print("Features importance...")
        gain = lgbm.booster_.feature_importance('gain')
        ft = pd.DataFrame({'feature': lgbm.booster_.feature_name(), 'split': lgbm.booster_.feature_importance('split'),
                           'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
        with pd.option_context('display.max_rows',None,'display.max_columns',3):
            print(ft)

        print('predicting...')
        #test set
        for i in range(0,7):
            data_path = os.path.join(self.cache_dir, "test_matrix%s.csv"%i)
            if not os.path.exists(data_path):
                continue
            with open(data_path, 'r') as gf:
                data = pd.read_csv(gf)

            X_test = data[cols]
            y_test = data['label']
            print('test set %s...'%i)
            print(y_test.value_counts())
            y_pre = lgbm.predict(X_test)
            y_pro = lgbm.predict_proba(X_test)[:, 1]
            if i==0:
                pass
            else:
                print("lightgbm model Test AUC Score: {0}".format(roc_auc_score(y_test, y_pro)))
                print("lightgbm model Test Precision: {0}".format(precision_score(y_test, y_pre)))
                print("lightgbm model Test   Recall : {0}".format(recall_score(y_test, y_pre)))
                print("lightgbm model Test F1 Score: {0}".format(f1_score(y_test, y_pre)))
                print("lightgbm model Test AUC of PR-curve: {0}".format(average_precision_score(y_test, y_pro)))
                print("lightgbm model Test logloss: {0}".format(log_loss(y_test, y_pro)))
                print("lightgbm Test confusion_matrix :")
                print(confusion_matrix(y_test, y_pre))
            # print(y_pro)
            data['potential_score'] = y_pro
            res = data[['user_id','potential_score']].sort_values(by='potential_score', ascending=False)
            # print(res)
            dump_path = os.path.join(self.cache_dir,  'lgbm_potential_results_%s.csv'%i)
            with open(dump_path, 'w') as gf:
                res.to_csv(gf,index=False)

    def train_gbdtlr_month(self):
        print("=" * 60)
        print(self.cache_dir)
        start_time = time.time()
        data_path = os.path.join(self.cache_dir, "train_matrix.csv")
        with open(data_path, 'r') as gf:
            data = pd.read_csv(gf)

        #cols = [col for col in data.columns.values if col not in ['label']]
        # ex_cols = [col for col in data.columns.values if (('is_' in col) or ('flag' in col))]
        # cols = list(set(data.columns.values)-set(ex_cols))
        # ('future_collect' in col) or
        cols = [col for col in data.columns.values if ((('label' not in col) and ('is_' not in col) and (
                    'flag' not in col) and ('user_id' not in col) and ('invest_id' not in col)) and (
                                                                   ('future_collect' in col) or (col.startswith('q')) or ('amt' in col) or (
                                                                               'cnt' in col) or ('aum' in col) or (
                                                                               'rank' in col) or ('income' in col) or (
                                                                               'media1' in col) or (
                                                                               'collect_amt' in col) or (
                                                                               'browse_id' in col) or (
                                                                               'duration' in col) or (
                                                                               'times' in col) or (
                                                                               'user_active_days' in col)))]
        # cols=['invest_amt','invest_cnt','invest_most_min_invest_amt','historical_max_aum','aum','invest_total_max_amt','inv_5w_cnt','invest_lately_12m_avg_aum','invest_b2c_cnt','invest_b2c_amt','invest_lately_24m_max_aum','invest_lately_3m_b2c_amt','inv_50w_cnt','inv_30w_cnt','inv_100w_cnt','invest_lately_3m_max_aum','invest_lately_3m_avg_aum','invest_lately_3m_cnt','invest_lately_3m_amt','invest_lately_3m_b2c_cnt','inv_10w_cnt']
        # cols = [col for col in data.columns.values if ((('label' not in col) and ('is_' not in col) and ('flag' not in col) and ('user_id' not in col) and ('invest_id' not in col)) and (('future_collect' in col) or (col.startswith('q')) or  ('amt' in col) or ('cnt' in col) or ('aum' in col)  or ('rank' in col) or ('income' in col)  or ('media1' in col) or ('collect_amt' in col) or ('browse_id' in col)  or ('duration' in col) or ('times' in col) or ('user_active_days' in col)))]
        # cols = ['aum','historical_max_aum','balance_amount','frozen_amount','applying_amount','invest_lately_12m_avg_aum','invest_lately_24m_max_aum','invest_lately_3m_b2c_amt','invest_lately_3m_b2c_cnt','invest_lately_3m_amt','invest_lately_3m_cnt','invest_lately_3m_avg_aum','invest_lately_3m_max_aum','inv_100w_cnt','inv_10w_cnt','inv_30w_cnt','inv_50w_cnt','inv_5w_cnt','invest_amt','invest_b2c_amt','invest_b2c_cnt','invest_cnt','invest_risklevel_unique','invest_most_min_invest_amt','invest_most_period_by_days','invest_most_rate','invest_most_risk_level','invest_total_max_amt']
        # cols =list(set(cls) + set(['aum','historical_max_aum','balance_amount','invest_lately_12m_avg_aum','invest_lately_24m_max_aum','invest_lately_3m_b2c_amt','invest_lately_3m_b2c_cnt','invest_lately_3m_amt','invest_lately_3m_cnt','invest_lately_3m_avg_aum','invest_lately_3m_max_aum','inv_100w_cnt','inv_10w_cnt','inv_30w_cnt','inv_50w_cnt','inv_5w_cnt','invest_amt','invest_b2c_amt','invest_b2c_cnt','invest_cnt','invest_risklevel_unique','invest_most_min_invest_amt','invest_most_period_by_days','invest_most_rate','invest_most_risk_level','invest_total_max_amt']))
        print("{0} input features:".format(len(cols)))
        print(cols)
        y = data['label']
        # scale_pos_weight = 1000
        scale_pos_weight = (y[y == 0].shape[0]) * 1.0 / (y[y == 1].shape[0])
        print("scale_pos_weight:",scale_pos_weight)
        X_train = data[cols]
        y_train = data['label']
        # X_train, X_test, y_train, y_test = train_test_split(data.loc[:, cols], y, test_size=0.2, random_state=999,
        #                                                     stratify=y)
        # lgbm = RandomForestClassifier()
        lgbm = XgboostLRClassifier()
        print('training set...')
        print(y_train.value_counts())
        lgbm.fit(X_train, y_train)
        print('training...')
        print('[{}] Train lightgbm completed'.format(time.time() - start_time))
        del data
        model_version = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        dump_model_path = os.path.join(self.model_dir,'gbdtlr_model_%s.pkl'%model_version)
        with open(dump_model_path, 'wb') as gf:
            pickle.dump(lgbm, gf)

        # print("Features importance...")
        # gain = lgbm.booster_.feature_importance('gain')
        # ft = pd.DataFrame({'feature': lgbm.booster_.feature_name(), 'split': lgbm.booster_.feature_importance('split'),
        #                    'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
        # with pd.option_context('display.max_rows',None,'display.max_columns',3):
        #     print(ft)

        print('predicting...')
        #test set
        for i in range(0,7):
            data_path = os.path.join(self.cache_dir, "test_matrix%s.csv"%i)
            if not os.path.exists(data_path):
                continue
            with open(data_path, 'r') as gf:
                data = pd.read_csv(gf)

            X_test = data[cols]
            y_test = data['label']
            print('test set %s...'%i)
            print(y_test.value_counts())
            y_pre = lgbm.predict(X_test)
            y_pro = lgbm.predict_proba(X_test)[:, 1]
            if i==0:
                pass
            else:
                print("gbdt+lr model Test AUC Score: {0}".format(roc_auc_score(y_test, y_pro)))
                print("gbdt+lr model Test Precision: {0}".format(precision_score(y_test, y_pre)))
                print("gbdt+lr model Test   Recall : {0}".format(recall_score(y_test, y_pre)))
                print("gbdt+lr model Test F1 Score: {0}".format(f1_score(y_test, y_pre)))
                print("gbdt+lr model Test AUC of PR-curve: {0}".format(average_precision_score(y_test, y_pro)))
                print("gbdt+lr model Test logloss: {0}".format(log_loss(y_test, y_pro)))
                print("gbdt+lr Test confusion_matrix :")
                print(confusion_matrix(y_test, y_pre))
            # print(y_pro)
            data['potential_score'] = y_pro
            res = data[['user_id','potential_score']].sort_values(by='potential_score', ascending=False)
            # print(res)
            dump_path = os.path.join(self.cache_dir,  'gbdtlr_potential_results_%s.csv'%i)
            with open(dump_path, 'w') as gf:
                res.to_csv(gf,index=False)

    def auto_train(self):
        print("=" * 60)
        start_time = time.time()
        data_path = os.path.join(self.cache_dir, "train_matrix.csv")
        with open(data_path, 'rb') as gf:
            data = pd.read_csv(gf)

        cols = [col for col in data.columns.values if col not in ['label']]
        y = data['label']
        X_train, X_test, y_train, y_test = train_test_split(data.loc[:, cols], y, test_size=0.2, random_state=999,
                                                            stratify=y)
        tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
        print('training...')
        tpot.fit(X_train, y_train)
        print('[{}] Auto train completed'.format(time.time() - start_time))
        print(tpot.score(X_test, y_test))
        tpot.export(os.path.join(self.cache_dir, 'tpot_kycrec_pipeline.py'))

    def gen_train_test_csvs(self):
        print("gen train test data:")
        train_data = pickle.load(
            open(os.path.join(self.cache_dir, "sampled_set_2017-10-01_2018-02-02.pkl"), 'rb'))

        print("train set n/y:")
        print(train_data.groupby('label').size())
        # sm = SMOTE(ratio='minority',random_state=999)
        # X_sm,y_sm = sm.fit_sample(train_data[],train_data['label'])
        # ('1_day_' not in col) and ('3_day_' not in col) and
        # cols = [col for col in train_data.columns.values if
        #         (('is_' not in col) and ('flag' not in col) and ('invest_id' not in col)) and ((
        #                 ('user_id' in col) or (col.startswith('q')) or ('future_collect' in col) or ('amt' in col) or (
        #                 'cnt' in col) or ('aum' in col) or ('rank' in col) or ('income' in col) or (
        #                         'media1' in col) or ('collect_amt' in col) or ('browse_id' in col) or (
        #                         'duration' in col) or ('times' in col) or ('user_active_days' in col)))]
        cols = [col for col in train_data.columns.values if (('1_day_' not in col) and ('3_day_' not in col) and ('label' not in col) and ('user_id' not in col))]
        print(cols)
        sl = FeatureSelection()
        sl.fit(train_data[cols], train_data['label'])
        print("selected cols:",sl.selected_cols)
        bfp = FeatureEncoder(self.args, sl.numerical_cols, sl.categorical_cols)
        dump_path = os.path.join(self.cache_dir, 'train_matrix.csv')
        feats = bfp.fit_transform(train_data[sl.selected_cols], train_data['label'], dump_path)
        print("train data shape:", train_data.shape)
        print("feats shape:",feats.shape)
        data_to_save = pd.concat([train_data[['user_id']], feats], axis=1)
        with open(dump_path, 'w') as gf:
            data_to_save.to_csv(gf, header=True, index=False)

        test_files = ['million_train_set_2018-03-04_2018-04-02_2018-04-03_2018-05-02.pkl','million_train_set_2018-01-29_2018-02-27_2018-02-28_2018-03-29.pkl','million_train_set_2018-01-24_2018-02-22_2018-02-23_2018-03-24.pkl','million_train_set_2018-01-19_2018-02-17_2018-02-18_2018-03-19.pkl','million_train_set_2018-01-14_2018-02-12_2018-02-13_2018-03-14.pkl','million_train_set_2018-01-09_2018-02-07_2018-02-08_2018-03-09.pkl','million_train_set_2018-01-04_2018-02-02_2018-02-03_2018-03-04.pkl']
        for i,f in enumerate(test_files):
            print(f)
            test_data = pickle.load(
                open(os.path.join(self.cache_dir, f), 'rb'))
            dump_path = os.path.join(self.cache_dir, 'test_matrix%s.csv'%i)
            feats = bfp.transform(test_data[sl.selected_cols], test_data['label'], dump_path)
            print("test data shape:", test_data.shape)
            print("feats shape:", feats.shape)
            data_to_save = pd.concat([test_data[['user_id']], feats], axis=1)
            with open(dump_path, 'w') as gf:
                data_to_save.to_csv(gf, header=True, index=False)


    def train(self):
        train_start_date = '2017-12-01'
        train_end_date = '2017-12-31'
        act_start_date = '2018-01-01'
        act_end_date = '2018-01-31'

        test_start_date = '2018-01-01'
        test_end_date = '2018-01-31'
        test_act_start_date = '2018-02-01'
        test_act_end_date = '2018-02-29'
        feat_sel_path = os.path.join(self.cache_dir, 'feat_sel.pkl')
        feat_encoder_path = os.path.join(self.cache_dir, 'feat_encoder.pkl')
        if os.path.exists(feat_sel_path) and os.path.exists(feat_encoder_path):
            with open(feat_sel_path, 'rb') as gf:
                sl = pickle.load(gf)
            with open(feat_encoder_path, 'rb') as gf:
                bfp = pickle.load(gf)
        else:
            sl = FeatureSelection(self.args)
            bfp = FeatureEncoder(self.args)

        train_path = os.path.join(self.cache_dir, 'feature_matrix.libsvm')
        test_path = os.path.join(self.cache_dir, 'feature_matrix_test.libsvm')
        if os.path.exists(train_path) and os.path.exists(test_path):
            pass
        else:
            X_train, y_train = self.make_train_set(train_start_date, train_end_date, act_start_date, act_end_date)
            print("train set y=0:{0}".format(y_train[y_train == 0].shape[0]))
            print("train set y=1:{0}".format(y_train[y_train == 1].shape[0]))
            info = psutil.virtual_memory()
            print('=' * 60)
            print(info)
            print('=' * 60)
            print(psutil.Process(os.getpid()).memory_info().rss)
            scale_pos_weight = (y_train[y_train == 0].shape[0]) * 1.0 / (y_train[y_train == 1].shape[0])

            sl.fit(X_train, y_train)
            X = sl.transform(X_train)
            del (X_train)

            if not os.path.exists(train_path):
                bfp.fit_transform(X, y_train, 'feature_matrix.libsvm')
                del (X)
                gc.collect()

            with open(feat_sel_path, 'wb') as gf:
                pickle.dump(sl, gf)
            with open(feat_encoder_path, 'wb') as gf:
                pickle.dump(bfp, gf)

            X_test, y_test = self.make_test_set(test_start_date, test_end_date, test_act_start_date, test_act_end_date)
            X = sl.transform(X_test)
            del (X_test)
            if not os.path.exists(test_path):
                bfp.transform(X, y_test, 'feature_matrix_test.libsvm')
                del (X)
                gc.collect()

        cores = multiprocessing.cpu_count()
        threads = int(cores * 0.8)
        if self.model_type == 'xgb':
            print("=" * 60)
            start_time = time.time()
            dtrain = xgb.DMatrix(train_path)
            dtest = xgb.DMatrix(test_path)

            gbm = xgb.XGBClassifier(n_estimators=30, learning_rate=0.3, max_depth=4, min_child_weight=6, gamma=0.3,
                                    subsample=0.7,
                                    colsample_bytree=0.7, objective='binary:logistic', nthread=threads,
                                    scale_pos_weight=scale_pos_weight, reg_alpha=1e-05, reg_lambda=1, seed=27)
            print('training...')
            gbm.fit(dtrain, label=y_train)
            print('[{}] Train xgboost completed'.format(time.time() - start_time))
            y_pre = gbm.predict(dtest)
            y_pro = gbm.predict_proba(dtest)
            print("Xgboost model Test AUC Score: {0}".format(roc_auc_score(y_test, y_pro)))
            print("Xgboost model Test Precision: {0}".format(precision_score(y_test, y_pre)))
            print("Xgboost model Test   Recall : {0}".format(recall_score(y_test, y_pre)))
            print("Xgboost model Test F1 Score: {0}".format(f1_score(y_test, y_pre)))
            print("Xgboost model Test AUC of PR-curve: {0}".format(average_precision_score(y_test, y_pro)))
            print("Xgboost model Test logloss: {0}".format(log_loss(y_test, y_pro)))
            print("Xgboost Test confusion_matrix :")
            print(confusion_matrix(y_test, y_pre))
            del (gbm)
            del (y_pre)
            del (y_pro)
        elif self.model_type == 'lgb':
            print("=" * 60)
            start_time = time.time()
            dtrain = lgb.Dataset(train_path)
            dtest = lgb.Dataset(test_path)
            lgbm = lgb.LGBMClassifier(boosting_type='gbdt', max_depth=4, learning_rate=0.3, n_estimators=30,
                                      scale_pos_weight=scale_pos_weight, min_child_weight=1, subsample=0.7,
                                      colsample_bytree=0.7,
                                      reg_alpha=1e-05, reg_lambda=1)
            lgbm.fit(dtrain, y_train)
            print('training...')
            lgbm.fit(dtrain, label=y_train)
            print('[{}] Train xgboost completed'.format(time.time() - start_time))
            y_pre = lgbm.predict(dtest)
            y_pro = lgbm.predict_proba(dtest)
            print("Xgboost model Test AUC Score: {0}".format(roc_auc_score(y_test, y_pro)))
            print("Xgboost model Test Precision: {0}".format(precision_score(y_test, y_pre)))
            print("Xgboost model Test   Recall : {0}".format(recall_score(y_test, y_pre)))
            print("Xgboost model Test F1 Score: {0}".format(f1_score(y_test, y_pre)))
            print("Xgboost model Test AUC of PR-curve: {0}".format(average_precision_score(y_test, y_pro)))
            print("Xgboost model Test logloss: {0}".format(log_loss(y_test, y_pro)))
            print("Xgboost Test confusion_matrix :")
            print(confusion_matrix(y_test, y_pre))
            del (lgbm)
            del (y_pre)
            del (y_pro)
            # elif self.model_type=='fm':
            #     print("=" * 60)
            #     start_time = time.time()
            #     fm = xl.FMModel()
            #     fm.fit(feature_matrix,y_train)
            #     print('[{}] Train xl FM completed'.format(time.time() - start_time))
            #     y_pro = fm.predict(X_test)
            #     print("xl FM model Test AUC Score: {0}".format(roc_auc_score(y_test, y_pro)))
            #     print("xl FM model Test AUC of PR-curve: {0}".format(average_precision_score(y_test, y_pro)))
            #     print("xl FM model Test logloss: {0}".format(log_loss(y_test, y_pro)))
            #     print("xl FM Test confusion_matrix :")
            # elif self.model_type=='wb':
            #     print("=" * 60)
            #     start_time = time.time()
            #     fm_ftrl = FM_FTRL(alpha=0.012, beta=0.01, L1=0.00001, L2=0.1, alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
            #                       D_fm=20, e_noise=0.0001, iters=3000, inv_link="identity", threads=4)
            #     fm_ftrl.fit(feature_matrix, y_train)
            #     print('[{}] Train FM_FTRL completed'.format(time.time() - start_time))
            #     y_pro = fm_ftrl.predict(X_test)
            #     print("FM_FTRL model Test AUC Score: {0}".format(roc_auc_score(y_test, y_pro)))
            #     print("FM_FTRL model Test AUC of PR-curve: {0}".format(average_precision_score(y_test, y_pro)))
            #     print("FM_FTRL model Test logloss: {0}".format(log_loss(y_test, y_pro)))
            #     print("FM_FTRL Test confusion_matrix :")


if __name__ == '__main__':
    args_in = sys.argv[1:]
    print(args_in)
    parser = argparse.ArgumentParser()
    mtyunArgs = parser.add_argument_group('cloud option')
    mtyunArgs.add_argument('--data_dir', type=str, default='',
                           help='input data path')
    mtyunArgs.add_argument('--cache_dir', type=str, default='',
                           help='cache data path')
    mtyunArgs.add_argument('--task', type=str, default='',
                           help='task type')
    mtyunArgs.add_argument('--model_dir', type=str, default='',
                           help='output model path')
    mtyunArgs.add_argument('--model_type', type=str, default='lgb',
                           help='model type str')
    mtyunArgs.add_argument('--tf_fs', type=str, default='', help='output model path')
    mtyunArgs.add_argument('--tf_prefix', type=str, default='', help='output model path')
    mtyunArgs.add_argument('--default_fs', type=str, default='', help='output model path')
    mtyunArgs.add_argument('--worker_num', type=str, default='', help='output model path')
    mtyunArgs.add_argument('--num_gpus', type=str, default='', help='output model path')
    mtyunArgs.add_argument('--num_ps', type=str, default='', help='output model path')
    mtyunArgs.add_argument('--num_worker', type=str, default='', help='output model path')
    mtyunArgs.add_argument('--tensorboard_dir', type=str, default='', help='output model path')
    mtyunArgs.add_argument('--tb_dir', type=str, default='local_tensorbord_dir_0', help='output model path')
    FLAGS, _ = parser.parse_known_args()
    print('FLAGS')
    print(FLAGS)
    args = parser.parse_args(args_in)

    r = Rec(args)
    if args.task == 'sliding':
        r.make_sliding_train_set('2017-10-01', '2018-04-01')
    elif args.task == 'sampling':
        r.sampling_data('2017-10-01', '2018-02-02')
    elif args.task=='gencsv':
        r.gen_train_test_csvs()
    elif args.task=='xgb':
        r.train_xgb_month()
    elif args.task == 'lgb':
        r.train_lgb_month()
    elif args.task == 'xgb,lgb':
        r.train_xgb_month()
        r.train_lgb_month()
    elif args.task == 'all':
        r.make_sliding_train_set('2017-10-01', '2018-04-01')
        r.make_sliding_train_set('2018-03-04', '2018-05-02')
        r.sampling_data('2017-10-01', '2018-02-02')
        r.gen_train_test_csvs()
        r.train_xgb_month()
        r.train_lgb_month()
    elif args.task == 'gen,train':
        r.gen_train_test_csvs()
        r.train_xgb_month()
        r.train_lgb_month()
    elif args.task=='gbdtlr':
        r.train_gbdtlr_month()

# r.merge_label_user_ids('2017-10-01', '2018-05-29')
    # r.merge_sliding_train_set('2018-02-03', '2018-03-29')
    # r.auto_train()
    #nohup /wls/personal/tangning593/software/anaconda3/bin/python targeting.py --data_dir=/wls/personal/tangning593/workspace/ail/data/targeting/ --cache_dir=/wls/personal/tangning593/workspace/ail/data/targeting/cache2 --model_dir=/wls/personal/tangning593/workspace/ail/data/targeting/model2 --task=all  > targeting_all.log 2>&1 &
