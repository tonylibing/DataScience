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
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score, f1_score, accuracy_score, average_precision_score, log_loss
import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime
from dateutil.parser import parse
import xgboost as xgb
import lightgbm as lgb
from tpot import TPOTClassifier
#import xlearn as xl
import tensorflow as tf
from tqdm import tqdm
from collections import Counter
from importlib import reload
sys.path.append("../..")
import feature.processor
reload(feature.processor)
from feature.processor import *
#from wordbatch.models import FTRL, FM_FTRL

FLAGS = None

def check_col(user_feats,cols):
    for col in cols:
        s = user_feats[user_feats[col].astype(str).str.contains('‰')]
        if s.size>0:
            print(col)
        
class Rec():
    def __init__(self, args):
        self.args = args
        self.data_dir = self.args.data_dir
        self.cache_dir = os.path.join(self.data_dir, 'cache')
        self.model_type = self.args.model_type

    def value_customer(self):
        product_dir = os.path.join(self.data_dir, "value_customer.csv")
        with tf.gfile.FastGFile(product_dir, 'rb') as gf:
            value_customers = pd.read_csv(gf)
        value_customers['user_id']=value_customers['user_id'].astype(int)
        return value_customers

    def user_feat(self):
        print('gen user feat')
        dump_path = os.path.join(self.cache_dir, 'user_feat.pkl')
        if tf.gfile.Exists(dump_path):
            with tf.gfile.FastGFile(dump_path, 'rb') as gf:
                user_feats = pickle.load(gf)
        else:
            user_feat_dir = os.path.join(self.data_dir, "value_customer_features.csv")
            with tf.gfile.FastGFile(user_feat_dir, 'rb') as gf:
                user_feats = pd.read_csv(gf)

            mod = user_feats['education_cd'].mode()[0]
            user_feats['education_cd'].replace('‰', mod, inplace=True)
            mod = user_feats['marriage_status_cd'].mode()[0]
            user_feats['marriage_status_cd'].replace('@', mod, inplace=True)
            subscribe_status = self.user_subscribe_status()
            user_feats = pd.merge(user_feats,subscribe_status,how='left',on='user_id')
            del(subscribe_status)
            gc.collect()
            audit_status = self.user_audit_status()
            user_feats = pd.merge(user_feats,audit_status,how='left',on='user_id')
            del(audit_status)
            gc.collect()
            with tf.gfile.FastGFile(dump_path, 'wb') as gf:
                pickle.dump(user_feats, gf)

        return user_feats

    def product_feat(self):
        print('gen product features')
        dump_path = os.path.join(self.cache_dir, 'product_feat.pkl')
        if tf.gfile.Exists(dump_path):
            with tf.gfile.FastGFile(dump_path, 'rb') as gf:
                products = pickle.load(gf)
        else:
            product_dir = os.path.join(self.data_dir, "products.csv")
            with tf.gfile.FastGFile(product_dir, 'rb') as gf:
                products = pd.read_csv(gf)
            products.dropna(inplace=True)
            products.rename(columns={'id': 'product_id'}, inplace=True)
            products.loc[products['product_id'] == 147566049, 'invest_period_by_days'] = 28
            products.loc[products['product_id'] == 157269050, 'invest_period_by_days'] = 7
            # products = products[(products['product_id']==147566049) | (products['product_id']==157269050)]
            # seg price
            amt_grp_names = ['1k', '1w', '5w', '10w', '30w', '50w', '100w', '1000w']
            amt_bins = [1, 1000, 10000, 50000, 100000, 300000, 500000, 1000000, 10000000]
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
            cmn_product_category_one_hot = pd.get_dummies(products['cmn_product_category'], prefix='cmn_cat')
            item_one_hot = pd.get_dummies(products['item'], prefix='item')
            products = pd.concat(
                [products, product_category_one_hot, cmn_product_category_one_hot, item_one_hot, amt, period], axis=1)
            # products =  pd.concat([products, product_category_one_hot,cmn_product_category_one_hot,item_one_hot,amt,period,amt2,period2], axis=1)
            products['product_group'] = products['price_group'].astype(str) + '_' + products['period_group'].astype(
                str) + '_' + products['product_category'].astype(str)
            # products.drop(['product_category', 'cmn_product_category', 'item', 'product_price', 'invest_period_by_days'],axis=1, inplace=True)
            # add product group cvr features
            with tf.gfile.FastGFile(dump_path, 'wb') as gf:
                pickle.dump(products, gf)

        return products

    def gen_sample(self, start_date, end_date):
        print('get interactive users', start_date, end_date)
        dump_path = os.path.join(self.cache_dir, 'samples_%s_%s.pkl' % (start_date, end_date))
        if tf.gfile.Exists(dump_path):
            with tf.gfile.FastGFile(dump_path, 'rb') as gf:
                samples = pickle.load(gf)
        else:
            actions = self.get_browse(start_date, end_date)
            samples = actions.groupby(['user_id', 'product_group']).size().reset_index()
            samples = samples[['user_id','product_group']]
            # samples = actions[['user_id']].drop_duplicates()
            print('samples num is:', samples.shape[0])
            with tf.gfile.FastGFile(dump_path, 'wb') as gf:
                pickle.dump(samples, gf)
        return samples

    def gen_million_sample(self, start_date, end_date):
        print('get interactive users', start_date, end_date)
        # start_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=span)
        # start_date = start_date.strftime('%Y-%m-%d')
        dump_path = os.path.join(self.cache_dir, 'samples_%s_%s.pkl' % (start_date, end_date))
        if tf.gfile.Exists(dump_path):
            with tf.gfile.FastGFile(dump_path, 'rb') as gf:
                samples = pickle.load(gf)
        else:
            actions = self.get_browse(start_date, end_date)
            samples = actions.groupby(['user_id', 'product_group']).size().reset_index()
            samples = samples[samples['product_group'].astype(str).str.contains('100w'),['user_id','product_group']]
            # samples = actions[['user_id']].drop_duplicates()
            print('million browse samples num is:', samples.shape[0])
            with tf.gfile.FastGFile(dump_path, 'wb') as gf:
                pickle.dump(samples, gf)
        return samples

    def get_browse(self, start_date, end_date):
        print('get browse', start_date, end_date)
        spec_browse_data = os.path.join(self.data_dir, "browse.csv")
        dump_path = os.path.join(self.cache_dir, 'user_browse_%s_%s.pkl' % (start_date, end_date))
        if tf.gfile.Exists(dump_path):
            with tf.gfile.FastGFile(dump_path, 'rb') as gf:
                browse = pickle.load(gf)
        else:
            with tf.gfile.FastGFile(spec_browse_data, 'rb') as gf:
                browse = pd.read_csv(gf)
            browse.dropna(inplace=True)
            browse['user_id'] = browse['user_id'].astype(int)
            browse['date'] = browse['request_time'].apply(lambda x: x[:10])
            del browse['request_time']
            browse = browse[(browse['date'] >= start_date) & (browse['date'] <= end_date)]
            products = self.product_feat()
            browse = pd.merge(browse, products[['product_id', 'product_group']], how='left', on='product_id')
            # browse = browse[(browse['product_id']==147566049) | (browse['product_id']==157269050)]
            with tf.gfile.FastGFile(dump_path, 'wb') as gf:
                pickle.dump(browse, gf)

        return browse

    def user_browse_feature(self, start_date, end_date):
        print('gen user browse features', start_date, end_date)
        browse_feat = None
        dump_path = os.path.join(self.cache_dir, 'user_browse_feat_%s_%s.pkl' % (start_date, end_date))
        if tf.gfile.Exists(dump_path):
            with tf.gfile.FastGFile(dump_path, 'rb') as gf:
                browse_feat = pickle.load(gf)
        else:
            interval_browse = self.get_browse(start_date, end_date)
            spans = [30, 15, 7, 3, 1]
            for span in spans:
                span_start_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=span-1)
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
                browse_feat_tmp = pd.merge(browse_feat_tmp, active_days, how='left', on='user_id')
                browse_feat_tmp = pd.merge(browse_feat_tmp, active_duration, how='left', on='user_id')

                if browse_feat is not None:
                    browse_feat = pd.merge(browse_feat, browse_feat_tmp, how='left', on=['user_id', 'product_group'])
                else:
                    browse_feat = browse_feat_tmp

            browse_feat.fillna(0, inplace=True)
            with tf.gfile.FastGFile(dump_path, 'wb') as gf:
                pickle.dump(browse_feat, gf)

        return browse_feat

    def get_collection(self, start_date, end_date):
        print('get actual collection', start_date, end_date)
        collect_amt_data = os.path.join(self.data_dir, "actual_collection.csv")
        dump_path = os.path.join(self.cache_dir, 'user_collection_%s_%s.pkl' % (start_date, end_date))
        if tf.gfile.Exists(dump_path):
            with tf.gfile.FastGFile(dump_path, 'rb') as gf:
                collection = pickle.load(gf)
        else:
            with tf.gfile.FastGFile(collect_amt_data, 'rb') as gf:
                collection = pd.read_csv(gf)

            collection.dropna(inplace=True)
            collection['user_id'] = collection['user_id'].astype(int)
            collection['date'] = collection['actual_collection_time'].apply(lambda x: x[:10])
            del collection['actual_collection_time']
            collection = collection[(collection['date'] >= start_date) & (collection['date'] <= end_date)]
            with tf.gfile.FastGFile(dump_path, 'wb') as gf:
                pickle.dump(collection, gf)

        return collection

    def user_collection_feature(self, start_date, end_date):
        print('gen user collection features', start_date, end_date)
        collection_feat = None
        dump_path = os.path.join(self.cache_dir, 'user_collection_feat_%s_%s.pkl' % (start_date, end_date))
        if tf.gfile.Exists(dump_path):
            with tf.gfile.FastGFile(dump_path, 'rb') as gf:
                collection_feat = pickle.load(gf)
        else:
            interval_collection = self.get_collection(start_date, end_date)
            spans = [30, 15, 7, 3, 1]
            for span in spans:
                span_start_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=span-1)
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
            with tf.gfile.FastGFile(dump_path, 'wb') as gf:
                pickle.dump(collection_feat, gf)

        return collection_feat

    def user_audit_status(self):
        print('get user audit status')
        dump_path = os.path.join(self.cache_dir, 'user_audit_status.pkl')
        if tf.gfile.Exists(dump_path):
            with tf.gfile.FastGFile(dump_path, 'rb') as gf:
                audit_status = pickle.load(gf)
        else:
            audit_data = os.path.join(self.data_dir, "audit_status.csv")
            with tf.gfile.FastGFile(audit_data, 'rb') as gf:
                audit_status = pd.read_csv(gf)
                audit_status['user_id'] = audit_status['user_id'].astype(int)
            with tf.gfile.FastGFile(dump_path, 'wb') as gf:
                pickle.dump(audit_status, gf)

        return audit_status

    def user_subscribe_status(self):
        print('get user subscribe status')
        dump_path = os.path.join(self.cache_dir, 'user_subscribe_status.pkl')
        if tf.gfile.Exists(dump_path):
            with tf.gfile.FastGFile(dump_path, 'rb') as gf:
                subscribe_status = pickle.load(gf)
        else:
            audit_data = os.path.join(self.data_dir, "subscribe_status.csv")
            with tf.gfile.FastGFile(audit_data, 'rb') as gf:
                subscribe_status = pd.read_csv(gf)
                subscribe_status['user_id'] = subscribe_status['user_id'].astype(int)
            with tf.gfile.FastGFile(dump_path, 'wb') as gf:
                pickle.dump(subscribe_status, gf)

        del(subscribe_status['stat_date'])
        return subscribe_status

    def get_invest(self, start_date, end_date):
        print('get invests', start_date, end_date)
        dump_path = os.path.join(self.cache_dir, 'user_invest_%s_%s.pkl' % (start_date, end_date))
        if tf.gfile.Exists(dump_path):
            with tf.gfile.FastGFile(dump_path, 'rb') as gf:
                invest = pickle.load(gf)
        else:
            products = self.product_feat()
            spec_invest_data = os.path.join(self.data_dir, "invests.csv")
            with tf.gfile.FastGFile(spec_invest_data, 'rb') as gf:
                invest = pd.read_csv(gf)
            invest.dropna(inplace=True)
            invest.rename(columns={'loaner_user_id': 'user_id'}, inplace=True)
            # invest['date']=invest['request_time'].apply(lambda x:x[:10])
            invest = invest[(invest['invest_dt'] >= start_date) & (invest['invest_dt'] <= end_date)]
            invest = pd.merge(invest, products[['product_id', 'product_group']], how='left', on='product_id')
            #        invest = invest[(invest['product_id']==147566049) |(invest['product_id']==157269050)]
            with tf.gfile.FastGFile(dump_path, 'wb') as gf:
                pickle.dump(invest, gf)

        return invest

    def user_invest_feature(self, start_date, end_date):
        print('gen user invest features', start_date, end_date)
        invest_feat = None
        dump_path = os.path.join(self.cache_dir, 'user_invest_feat_%s_%s.pkl' % (start_date, end_date))
        if tf.gfile.Exists(dump_path):
            with tf.gfile.FastGFile(dump_path, 'rb') as gf:
                invest_feat = pickle.load(gf)
        else:
            interval_invest = self.get_invest(start_date, end_date)
            spans = [30, 15, 7, 3, 1]
            for span in spans:
                span_start_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=span-1)
                span_start_date = span_start_date.strftime('%Y-%m-%d')
                print(span_start_date, end_date)
                if span_start_date < start_date:
                    print('pass:', span_start_date, start_date)
                    span_start_date=start_date
                    # continue
                invest = interval_invest[
                    (interval_invest['invest_dt'] <= end_date) & (interval_invest['invest_dt'] >=span_start_date)]
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
                invest_feat_tmp = pd.merge(invest_feat_tmp, invest_times, how='left', on='user_id')
                invest_feat_tmp = pd.merge(invest_feat_tmp, invest_amt, how='left', on='user_id')

                if invest_feat is not None:
                    invest_feat = pd.merge(invest_feat, invest_feat_tmp, how='left', on=['user_id', 'product_group'])
                else:
                    invest_feat = invest_feat_tmp

            invest_feat.fillna(0, inplace=True)
            with tf.gfile.FastGFile(dump_path, 'wb') as gf:
                pickle.dump(invest_feat, gf)

        return invest_feat

    def gen_labels(self, start_date, end_date):
        print('gen lables', start_date, end_date)
        dump_path = os.path.join(self.cache_dir, 'labels_%s_%s.pkl' % (start_date, end_date))
        if tf.gfile.Exists(dump_path):
            with tf.gfile.FastGFile(dump_path, 'rb') as gf:
                invests = pickle.load(gf)
        else:
            invests = self.get_invest(start_date, end_date)
            invests = invests.groupby(['user_id', 'product_group'], as_index=False).sum()
            invests['label'] = 1
            invests = invests[['user_id', 'product_group', 'label']]
            with tf.gfile.FastGFile(dump_path, 'wb') as gf:
                pickle.dump(invests, gf)

        return invests

    def gen_million_labels(self, start_date, end_date):
        print('gen lables', start_date, end_date)
        dump_path = os.path.join(self.cache_dir, 'labels_%s_%s.pkl' % (start_date, end_date))
        if tf.gfile.Exists(dump_path):
            with tf.gfile.FastGFile(dump_path, 'rb') as gf:
                invests = pickle.load(gf)
        else:
            invests = self.get_million_invest(start_date, end_date)
            invests = invests.groupby(['user_id', 'product_group'], as_index=False).sum()
            invests['label'] = 1
            invests = invests[['user_id', 'product_group', 'label']]
            with tf.gfile.FastGFile(dump_path, 'wb') as gf:
                pickle.dump(invests, gf)

        return invests

    def make_train_set(self, train_start_date, train_end_date, test_start_date, test_end_date):
        print('make train set', train_start_date, train_end_date, test_start_date, test_end_date)
        dump_path = os.path.join(self.cache_dir, 'train_set_%s_%s_%s_%s.pkl' % (
            train_start_date, train_end_date, test_start_date, test_end_date))
        if tf.gfile.Exists(dump_path):
            with tf.gfile.FastGFile(dump_path, 'rb') as gf:
                train_set = pickle.load(gf)
        else:
            browse_feat = self.user_browse_feature(train_start_date, train_end_date)
            invest_feat = self.user_invest_feature(train_start_date, train_end_date)
            print(browse_feat.columns)
            print(invest_feat.columns)
            train_set = pd.merge(browse_feat, invest_feat, how='outer', on=['user_id', 'product_group'])
            del(browse_feat)
            del(invest_feat)
            gc.collect()
            user_feat = self.user_feat()
            print(user_feat.columns)
            train_set = pd.merge(user_feat,train_set, how='left', on='user_id')
            del(user_feat)
            gc.collect()
            collect_feat = self.get_collection(train_start_date,train_end_date)
            train_set = pd.merge(train_set,collect_feat, how='left', on='user_id')
            del(collect_feat)
            gc.collect()
            labels = self.gen_labels(test_start_date, test_end_date)
            train_set = pd.merge(train_set, labels, how='outer', on=['user_id', 'product_group'])
            del(labels)
            gc.collect()
            train_set['label'].fillna(0, inplace=True)
            with tf.gfile.FastGFile(dump_path, 'wb') as gf:
                pickle.dump(train_set, gf)

        labels = train_set['label'].copy()
        del train_set['user_id']
        del train_set['product_group']
        del train_set['label']

        print('train set cols:')
        print(train_set.columns)
        return train_set, labels

    def make_million_train_set(self, train_start_date, train_end_date, test_start_date, test_end_date):
        print('make train set', train_start_date, train_end_date, test_start_date, test_end_date)
        dump_path = os.path.join(self.cache_dir, 'million_train_set_%s_%s_%s_%s.pkl' % (
            train_start_date, train_end_date, test_start_date, test_end_date))
        if tf.gfile.Exists(dump_path):
            return
            with tf.gfile.FastGFile(dump_path, 'rb') as gf:
                train_set = pickle.load(gf)
        else:
            browse_feat = self.user_browse_feature(train_start_date, train_end_date)
            invest_feat = self.user_invest_feature(train_start_date, train_end_date)
            print(browse_feat.columns)
            print(invest_feat.columns)
            train_set = pd.merge(browse_feat, invest_feat, how='outer', on=['user_id', 'product_group'])
            # pickle.dump(train_set, open(os.path.join(self.cache_dir,'browse_invest_{0}_{1}.pkl'.format(train_start_date,train_end_date)),'wb'))
            del(browse_feat)
            del(invest_feat)
            gc.collect()
            user_feat = self.user_feat()
            print(user_feat.columns)
            train_set = pd.merge(train_set,user_feat, how='left', on='user_id')
            del(user_feat)
            gc.collect()
            collect_feat = self.user_collection_feature(train_start_date,train_end_date)
            train_set = pd.merge(train_set,collect_feat, how='left', on='user_id')
            del(collect_feat)
            gc.collect()
            labels = self.gen_labels(test_start_date, test_end_date)
            train_set = pd.merge(train_set, labels, how='outer', on=['user_id', 'product_group'])
            # pickle.dump(train_set, open(
            #     os.path.join(self.cache_dir, 'trainset_labels_{0}_{1}.pkl'.format(train_start_date, train_end_date)),
            #     'wb'))
            del(labels)
            gc.collect()
            train_set['label'].fillna(0, inplace=True)
            train_set=train_set[train_set['product_group'].str.contains("100w")]
            with tf.gfile.FastGFile(dump_path, 'wb') as gf:
                pickle.dump(train_set, gf)
            del(train_set)
            gc.collect()

        # labels = train_set['label'].copy()
        # del train_set['user_id']
        # del train_set['product_group']
        # del train_set['label']
        #
        # print('train set cols:')
        # print(train_set.columns)
        # return train_set, labels

    def make_test_set(self, test_start_date, test_end_date,test_act_start_date,test_act_end_date):
        print('make test set{0}-{1}-{2}-{3}'.format( test_start_date, test_end_date,test_act_start_date,test_act_end_date))
        dump_path = os.path.join(self.cache_dir, 'test_set_%s_%s.pkl' % (test_start_date, test_end_date))
        if tf.gfile.Exists(dump_path):
            with tf.gfile.FastGFile(dump_path, 'rb') as gf:
                test_set = pickle.load(gf)
        else:
            test_set = self.gen_sample(test_start_date, test_end_date)
            # test_set = self.gen_million_sample(test_start_date, test_end_date)
            browse_feat = self.user_browse_feature(test_start_date, test_end_date)
            invest_feat = self.user_invest_feature(test_start_date, test_end_date)
            user_feat = self.user_feat()
            print(user_feat.columns)
            test_set = pd.merge(test_set, user_feat, how='left', on='user_id')
            labels = self.gen_labels(test_act_start_date,test_act_end_date)
            test_set = pd.merge(test_set, invest_feat, how='left', on=['user_id', 'product_group'])
            test_set = pd.merge(test_set, browse_feat, how='left', on=['user_id', 'product_group'])
            test_set = pd.merge(test_set, labels, how='left', on=['user_id', 'product_group'])
            del(browse_feat)
            del(invest_feat)
            del(user_feat)
            del(labels)
            gc.collect()
            test_set.fillna(0, inplace=True)
            with tf.gfile.FastGFile(dump_path, 'wb') as gf:
                pickle.dump(test_set, gf)

        labels = test_set['label'].copy()
        del test_set['user_id']
        del test_set['product_group']
        del test_set['label']
        print('test set cols:')
        print(test_set.columns)
        return test_set, labels
        # index = test_set[['user_id']].copy()
        # del test_set['user_id']
        # print('test set cols:')
        # print(test_set.columns)
        # return index, test_set

    def gen_test(self):
        train_start_date = '2017-12-01'
        train_end_date = '2017-12-31'
        act_start_date = '2018-01-01'
        act_end_date = '2018-01-31'

        test_start_date = '2017-12-31'
        test_end_date = '2018-01-29'
        test_act_start_date = '2018-01-30'
        test_act_end_date = '2018-02-28'

        feat_sel_path = os.path.join(self.cache_dir, 'feat_sel.pkl')
        feat_encoder_path = os.path.join(self.cache_dir, 'feat_encoder.pkl')
        if tf.gfile.Exists(feat_sel_path) and tf.gfile.Exists(feat_encoder_path):
            with tf.gfile.FastGFile(feat_sel_path, 'rb') as gf:
                sl = pickle.load(gf)
            with tf.gfile.FastGFile(feat_encoder_path, 'rb') as gf:
                bfp = pickle.load(gf)

        X_test, y_test = self.make_test_set(test_start_date, test_end_date,test_act_start_date,test_act_end_date)
        X = sl.transform(X_test)
        del (X_test)
        test_path = os.path.join(self.cache_dir, 'feature_matrix_test.libsvm')
        if not tf.gfile.Exists(test_path):
            bfp.transform(X, y_test, 'feature_matrix_test.libsvm')
            del(X)
            gc.collect()

    def make_train_libsvm_matrix(self,train_start_date,train_end_date,act_start_date,act_end_date):
        dump_path = os.path.join(self.cache_dir, 'train_matrix_%s_%s_%s_%s.libsvm' % (
            train_start_date, train_end_date, act_start_date, act_end_date))
        if tf.gfile.Exists(dump_path):
            return
        print("making train libsvm matrix:{0}-{1}-{2}-{3}".format(train_start_date,train_end_date,act_start_date,act_end_date))
        feat_sel_path = os.path.join(self.cache_dir, 'feat_sel.pkl')
        feat_encoder_path = os.path.join(self.cache_dir, 'feat_encoder.pkl')

        X_train, y_train = self.make_train_set(train_start_date, train_end_date, act_start_date, act_end_date)
        print("train set y=0:{0}".format(y_train[y_train == 0].shape[0]))
        print("train set y=1:{0}".format(y_train[y_train == 1].shape[0]))

        if tf.gfile.Exists(feat_sel_path) and tf.gfile.Exists(feat_encoder_path):
            with tf.gfile.FastGFile(feat_sel_path, 'rb') as gf:
                sl = pickle.load(gf)
            with tf.gfile.FastGFile(feat_encoder_path, 'rb') as gf:
                bfp = pickle.load(gf)
        else:
            sl = FeatureSelection(self.args)
            sl.fit(X_train, y_train)
            bfp = FeatureEncoder(self.args,sl.numerical_cols,sl.categorical_cols)
            bfp.fit(X_train[sl.selected_cols],y_train,dump_path)

        sl.transform(X_train)
        bfp.transform(X_train[sl.selected_cols],y_train, dump_path)

        if tf.gfile.Exists(feat_sel_path) and tf.gfile.Exists(feat_encoder_path):
            pass
        else:
            with tf.gfile.FastGFile(feat_sel_path, 'wb') as gf:
                pickle.dump(sl, gf)
            with tf.gfile.FastGFile(feat_encoder_path, 'wb') as gf:
                pickle.dump(bfp, gf)

    def make_sliding_train_matrix(self,start_date, end_date, window=30,step=5):
        whole = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
        l = int(whole.days / step)
        print("gen train matrix with step", step)
        for i in tqdm(range(l),desc='train matrix:'):
            s = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=i * step)
            train_start_date = datetime.strftime(s, '%Y-%m-%d')
            s = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=(i * step)+window-1)
            train_end_date = datetime.strftime(s, '%Y-%m-%d')
            e = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=(i *step)+window)
            act_start_date = datetime.strftime(e, '%Y-%m-%d')
            e = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=(i *step)+2*window-1)
            act_end_date = datetime.strftime(e, '%Y-%m-%d')
            if act_end_date <= end_date:
                self.make_train_libsvm_matrix(train_start_date,train_end_date,act_start_date,act_end_date)


    def make_sliding_train_set(self,start_date, end_date, window=30,step=5):
        whole = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
        l = int(whole.days / step)
        print("gen train matrix with step", step)
        for i in tqdm(range(l),desc='train matrix:'):
            s = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=i * step)
            train_start_date = datetime.strftime(s, '%Y-%m-%d')
            s = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=(i * step)+window-1)
            train_end_date = datetime.strftime(s, '%Y-%m-%d')
            e = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=(i *step)+window)
            act_start_date = datetime.strftime(e, '%Y-%m-%d')
            e = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=(i *step)+2*window-1)
            act_end_date = datetime.strftime(e, '%Y-%m-%d')
            if act_end_date <= end_date:
                self.make_million_train_set(train_start_date,train_end_date,act_start_date,act_end_date)

    def merge_sliding_train_set(self,start_date, end_date, window=30,step=5):
        whole = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')
        l = int(whole.days / step)
        print("merge train set with step", step)
        res = None
        for i in tqdm(range(l),desc='train matrix:'):
            s = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=i * step)
            train_start_date = datetime.strftime(s, '%Y-%m-%d')
            s = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=(i * step)+window-1)
            train_end_date = datetime.strftime(s, '%Y-%m-%d')
            e = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=(i *step)+window)
            act_start_date = datetime.strftime(e, '%Y-%m-%d')
            e = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=(i *step)+2*window-1)
            act_end_date = datetime.strftime(e, '%Y-%m-%d')
            if act_end_date <= end_date:
                print('make train set', train_start_date, train_end_date, act_start_date, act_end_date)
                dump_path = os.path.join(self.cache_dir, 'million_train_set_%s_%s_%s_%s.pkl' % (
                    train_start_date, train_end_date, act_start_date, act_end_date))
                train_set = pickle.load(open(dump_path,'rb'))
                if res is not None:
                    tmp = pd.concat([res,train_set],axis=0)
                    del(res)
                    res = tmp
                    del(train_set)
                    gc.collect()
                else:
                    res = train_set

        if res is not None:
            with tf.gfile.FastGFile(os.path.join(self.cache_dir,"merge_million_train_set_%s_%s_window%s_step%s.pkl" % (start_date, end_date,window,step)), 'wb') as gf:
                pickle.dump(res, gf)

        cols = [col for col in res.columns.values if col not in ['user_id', 'product_group', 'label']]
        sl = FeatureSelection()
        sl.fit(res[cols], res['label'])

        bfp = FeatureEncoder(None, sl.numerical_cols, sl.categorical_cols)

        dump_path = os.path.join(self.cache_dir, 'train_matrix.csv')
        bfp.fit_transform(res[sl.selected_cols], res['label'], dump_path)

    def train_xgb(self):
        print("=" * 60)
        start_time = time.time()
        data_path = os.path.join(self.cache_dir, "train_matrix.csv")
        with tf.gfile.FastGFile(data_path, 'rb') as gf:
            data = pd.read_csv(gf)

        cols = [col for col in data.columns.values if col not in ['label']]
        y =data['label']
        scale_pos_weight = (y[y == 0].shape[0]) * 1.0 / (y[y == 1].shape[0])
        X_train, X_test, y_train, y_test = train_test_split(data.loc[:,cols], y, test_size=0.3, random_state=999,stratify=y)

        threads = int(0.8*multiprocessing.cpu_count())
        gbm = xgb.XGBClassifier(n_estimators=30, learning_rate=0.3, max_depth=4, min_child_weight=6, gamma=0.3,
                                subsample=0.7,
                                colsample_bytree=0.7, objective='binary:logistic', nthread=threads,
                                scale_pos_weight=scale_pos_weight, reg_alpha=1e-05, reg_lambda=1, seed=27)
        print('training...')
        gbm.fit(X_train, y_train)
        print('[{}] Train xgboost completed'.format(time.time() - start_time))
        print('predicting...')
        print('test set...',y_test.value_counts())
        print("feature importance:")
        # plot
        feat_imp = pd.Series(gbm.get_booster().get_fscore()).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')

        y_pre = gbm.predict(X_test)
        y_pro = gbm.predict_proba(X_test)[:,1]
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

    def train_lgb(self):
        print("=" * 60)
        start_time = time.time()
        data_path = os.path.join(self.cache_dir, "train_matrix.csv")
        with tf.gfile.FastGFile(data_path, 'rb') as gf:
            data = pd.read_csv(gf)

        cols = [col for col in data.columns.values if col not in ['label']]
        y =data['label']
        scale_pos_weight = (y[y == 0].shape[0]) * 1.0 / (y[y == 1].shape[0])
        X_train, X_test, y_train, y_test = train_test_split(data.loc[:,cols], y, test_size=0.3, random_state=999,stratify=y)
        lgbm = lgb.LGBMClassifier(boosting_type='gbdt', max_depth=4, learning_rate=0.3, n_estimators=30,
                                  scale_pos_weight=scale_pos_weight, min_child_weight=1, subsample=0.7,
                                  colsample_bytree=0.7,
                                  reg_alpha=1e-05, reg_lambda=1)
        lgbm.fit(X_train, y_train)
        print('training...')
        lgbm.fit(X_train,y_train)
        print('[{}] Train xgboost completed'.format(time.time() - start_time))
        print('predicting...')
        print('test set...',y_test.value_counts())
        y_pre = lgbm.predict(X_test)
        y_pro = lgbm.predict_proba(X_test)[:,1]
        print("lightgbm model Test AUC Score: {0}".format(roc_auc_score(y_test, y_pro)))
        print("lightgbm model Test Precision: {0}".format(precision_score(y_test, y_pre)))
        print("lightgbm model Test   Recall : {0}".format(recall_score(y_test, y_pre)))
        print("lightgbm model Test F1 Score: {0}".format(f1_score(y_test, y_pre)))
        print("lightgbm model Test AUC of PR-curve: {0}".format(average_precision_score(y_test, y_pro)))
        print("lightgbm model Test logloss: {0}".format(log_loss(y_test, y_pro)))
        print("lightgbm Test confusion_matrix :")
        print(confusion_matrix(y_test, y_pre))
        del (lgbm)
        del (y_pre)
        del (y_pro)


    def auto_train(self):
        print("=" * 60)
        start_time = time.time()
        data_path = os.path.join(self.cache_dir, "train_matrix.csv")
        with tf.gfile.FastGFile(data_path, 'rb') as gf:
            data = pd.read_csv(gf)

        cols = [col for col in data.columns.values if col not in ['label']]
        y =data['label']
        X_train, X_test, y_train, y_test = train_test_split(data.loc[:,cols], y, test_size=0.3, random_state=999,stratify=y)
        tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
        print('training...')
        tpot.fit(X_train, y_train)
        print('[{}] Auto train completed'.format(time.time() - start_time))
        print(tpot.score(X_test, y_test))
        tpot.export(os.path.join(self.cache_dir,'tpot_kycrec_pipeline.py'))

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
        if tf.gfile.Exists(feat_sel_path) and tf.gfile.Exists(feat_encoder_path):
            with tf.gfile.FastGFile(feat_sel_path, 'rb') as gf:
                sl = pickle.load(gf)
            with tf.gfile.FastGFile(feat_encoder_path, 'rb') as gf:
                bfp = pickle.load(gf)
        else:
            sl = FeatureSelection(self.args)
            bfp = FeatureEncoder(self.args)

        train_path = os.path.join(self.cache_dir, 'feature_matrix.libsvm')
        test_path = os.path.join(self.cache_dir, 'feature_matrix_test.libsvm')
        if tf.gfile.Exists(train_path) and tf.gfile.Exists(test_path):
            pass
        else:
            X_train, y_train = self.make_train_set(train_start_date, train_end_date, act_start_date, act_end_date)
            print("train set y=0:{0}".format(y_train[y_train == 0].shape[0]))
            print("train set y=1:{0}".format(y_train[y_train == 1].shape[0]))
            info = psutil.virtual_memory()
            print('='*60)
            print(info)
            print('='*60)
            print(psutil.Process(os.getpid()).memory_info().rss)
            scale_pos_weight = (y_train[y_train == 0].shape[0]) * 1.0 / (y_train[y_train == 1].shape[0])

            sl.fit(X_train, y_train)
            X = sl.transform(X_train)
            del(X_train)

            if not tf.gfile.Exists(train_path):
                bfp.fit_transform(X,y_train,'feature_matrix.libsvm')
                del(X)
                gc.collect()

            with tf.gfile.FastGFile(feat_sel_path, 'wb') as gf:
                pickle.dump(sl,gf)
            with tf.gfile.FastGFile(feat_encoder_path, 'wb') as gf:
                pickle.dump(bfp,gf)

            X_test, y_test = self.make_test_set(test_start_date, test_end_date,test_act_start_date,test_act_end_date)
            X = sl.transform(X_test)
            del(X_test)
            if not tf.gfile.Exists(test_path):
                bfp.transform(X,y_test,'feature_matrix_test.libsvm')
                del(X)
                gc.collect()

        cores = multiprocessing.cpu_count()
        threads = int(cores*0.8)
        if self.model_type=='xgb':
            print("=" * 60)
            start_time = time.time()
            dtrain = xgb.DMatrix(train_path)
            dtest = xgb.DMatrix(test_path)

            gbm = xgb.XGBClassifier(n_estimators=30, learning_rate=0.3, max_depth=4, min_child_weight=6, gamma=0.3,subsample=0.7,
                                colsample_bytree=0.7, objective='binary:logistic', nthread=threads,
                                scale_pos_weight=scale_pos_weight, reg_alpha=1e-05, reg_lambda=1, seed=27)
            print('training...')
            gbm.fit(dtrain,label=y_train)
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
            del(gbm)
            del(y_pre)
            del(y_pro)
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
            lgbm.fit(dtrain,label=y_train)
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
            del(lgbm)
            del(y_pre)
            del(y_pro)
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

def main(_):
    args_in = sys.argv[1:]
    print(args_in)
    parser = argparse.ArgumentParser()
    mtyunArgs = parser.add_argument_group('美团云选项')
    mtyunArgs.add_argument('--data_dir', type=str, default='',
                           help='input data path')
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
    # r.make_million_train_set('2017-12-01','2017-12-30','2017-12-31','2018-01-29')
    r.make_sliding_train_set('2017-12-01','2018-03-01')
    r.merge_sliding_train_set('2017-12-01','2018-03-01')
    # r.train_lgb()
    # r.train_xgb()
    # r.auto_train()
    # r.train()

if __name__ == '__main__':
    tf.app.run(main=main)
