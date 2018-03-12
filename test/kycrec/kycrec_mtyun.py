import numpy as np
import pandas as pd
import os
import sys
import argparse
import pickle
from sklearn.datasets import dump_svmlight_file
from datetime import datetime
from datetime import timedelta
from dateutil.parser import parse
import xgboost
import lightgbm
import tensorflow as tf

from xgboost import XGBClassifier
from collections import Counter

# data_dir = "E:\\dataset\\kycdata"
# cache_dir = "E:\\dataset\\kycdata\\cache"

FLAGS = None

class Rec():
    def __init__(self,args):
        self.args = args
        self.data_dir =  self.args.data_dir
        self.cache_dir = os.path.join(self.data_dir,'cache')
    
    def value_customer(self):
        product_dir = os.path.join(self.data_dir, "value_customer.csv")
        with tf.gfile.FastGFile(product_dir, 'rb') as gf:
            value_customers = pd.read_csv(gf)
        value_customers
        return value_customers
    
    
    def user_feat(self):
        pass
    
    
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
            #products = products[(products['product_id']==147566049) | (products['product_id']==157269050)]
            # seg price
            amt_grp_names = ['1k', '1w', '5w', '10w', '30w', '50w', '100w', '1000w']
            amt_bins = [1, 1000, 10000, 50000, 100000, 300000, 500000, 1000000, 10000000]
            amt = pd.cut(products['product_price'], amt_bins, labels=amt_grp_names)
            amt.rename('price_group', inplace=True)
            amt2 = pd.get_dummies(amt, prefix='amt')
            # seg period
            period_grp = ['1week', '1m', '2m', '3m', '6m', '9m', '12m', '15m', '18m', '24m', '30m', '36m']
            period_bins = [1, 7, 30, 60, 90, 180, 360, 450, 540, 720, 900, 1080, 2160]
            period = pd.cut(products['invest_period_by_days'], period_bins, labels=period_grp)
            period.rename('period_group', inplace=True)
            period2 = pd.get_dummies(period, prefix='invest_period')
            products['product_category'] = products['product_category'].astype(str)
            product_category_one_hot = pd.get_dummies(products['product_category'], prefix='cat')
            cmn_product_category_one_hot = pd.get_dummies(products['cmn_product_category'], prefix='cmn_cat')
            item_one_hot = pd.get_dummies(products['item'], prefix='item')
            products = pd.concat(
                [products, product_category_one_hot, cmn_product_category_one_hot, item_one_hot, amt, period, amt2,
                 period2], axis=1)
            # products =  pd.concat([products, product_category_one_hot,cmn_product_category_one_hot,item_one_hot,amt,period,amt2,period2], axis=1)
            products['product_group'] = products['price_group'].astype(str) + '_' + products['period_group'].astype(str)
            products.drop(['product_category', 'cmn_product_category', 'item', 'product_price', 'invest_period_by_days'],
                          axis=1, inplace=True)
            with tf.gfile.FastGFile(dump_path, 'wb') as gf:
                pickle.dump(products, gf)

        return products

    def get_browse(self,start_date, end_date):
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
            browse = browse[(browse['date'] >= start_date) & (browse['date'] < end_date)]
            products = self.product_feat()
            browse = pd.merge(browse, products[['product_id', 'product_group']], how='left', on='product_id')
            #browse = browse[(browse['product_id']==147566049) | (browse['product_id']==157269050)]
            with tf.gfile.FastGFile(dump_path, 'wb') as gf:
                pickle.dump(browse, gf)
    
        return browse
    
    
    def gen_sample(self,start_date, end_date):
        print('get interactive users', start_date, end_date)
        # start_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=span)
        # start_date = start_date.strftime('%Y-%m-%d')
        dump_path = os.path.join(self.cache_dir, 'samples_%s_%s.pkl' % (start_date, end_date))
        if tf.gfile.Exists(dump_path):
            with tf.gfile.FastGFile(dump_path, 'rb') as gf:
                samples = pickle.load(gf)
        else:
            actions = self.get_browse(start_date, end_date)
            samples = actions[['user_id']].drop_duplicates()
            print('samples num is:', samples.shape[0])
            with tf.gfile.FastGFile(dump_path, 'wb') as gf:
                pickle.dump(samples, gf)
        return samples
    
    
    def user_browse_feature(self,start_date, end_date):
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
                span_start_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=span)
                span_start_date = span_start_date.strftime('%Y-%m-%d')
                print(span_start_date, end_date)
                if span_start_date < start_date:
                    print('pass:', span_start_date, start_date)
                    continue
                browse = interval_browse[
                    (interval_browse['date'] <= end_date) & (interval_browse['date'] > span_start_date)]
                # browse times
                browse_product_times = browse.groupby(['user_id', 'product_group']).size().reset_index()
                browse_product_times.rename(columns={0: '%d_day_browse_id_times' % (span)}, inplace=True)
                # browse duration
                browse_product_duration = browse.groupby(['user_id', 'product_group'])['duraction'].sum().reset_index()
                browse_product_duration.rename(columns={'duraction': '%d_day_browse_id_duration' % (span)}, inplace=True)
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
    
    
    def get_invest(self,start_date, end_date):
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
            invest = invest[(invest['invest_dt'] >= start_date) & (invest['invest_dt'] < end_date)]
            invest = pd.merge(invest, products[['product_id', 'product_group']], how='left', on='product_id')
            #        invest = invest[(invest['product_id']==147566049) |(invest['product_id']==157269050)]
            with tf.gfile.FastGFile(dump_path, 'wb') as gf:
                pickle.dump(invest, gf)
    
        return invest
    
    
    def user_invest_feature(self,start_date, end_date):
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
                span_start_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=span)
                span_start_date = span_start_date.strftime('%Y-%m-%d')
                print(span_start_date, end_date)
                if span_start_date < start_date:
                    print('pass:', span_start_date, start_date)
                    continue
                invest = interval_invest[
                    (interval_invest['invest_dt'] <= end_date) & (interval_invest['invest_dt'] > span_start_date)]
                # invest times
                #            invest_product_times = invest.groupby(['user_id', 'product_id']).size().reset_index()
                invest_product_times = invest.groupby(['user_id', 'product_group']).size().reset_index()
                invest_product_times.rename(columns={0: '%d_day_invest_id_times' % (span)}, inplace=True)
                # invest product id amount
                invest_product_amt = invest.groupby(['user_id', 'product_group'])['investment_amount'].sum().reset_index()
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
    
    
    def gen_labels(self,start_date, end_date):
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
    
    
    def make_train_set(self,train_start_date, train_end_date, test_start_date, test_end_date):
        print('make train set', train_start_date, train_end_date, test_start_date, test_end_date)
        dump_path = os.path.join(self.cache_dir, 'train_set_%s_%s_%s_%s.pkl' % (
        train_start_date, train_end_date, test_start_date, test_end_date))
        if tf.gfile.Exists(dump_path):
            with tf.gfile.FastGFile(dump_path, 'rb') as gf:
                actions = pickle.load(gf)
        else:
            # user = get_basic_user_feat()
            # product = get_basic_product_feat()
            browse_feat = self.user_browse_feature(train_start_date, train_end_date)
            invest_feat = self.user_invest_feature(train_start_date, train_end_date)
            # user_acc = get_accumulate_user_feat(start_days, train_end_date)
            # product_acc = get_accumulate_product_feat(start_days, train_end_date)
            # comment_acc = get_comments_product_feat(train_start_date, train_end_date)
            labels = self.gen_labels(test_start_date, test_end_date)
            print(browse_feat.columns,invest_feat.columns)
            actions = pd.merge(browse_feat, invest_feat, how='left', on=['user_id', 'product_group'])
            # actions = pd.merge(actions, user, how='left', on='user_id')
            # actions = pd.merge(actions, user_acc, how='left', on='product_id')
            # product_features = self.product_feat()
            # actions = pd.merge(actions, product_features, how='left', on='product_group')
            # actions = pd.merge(actions, product_acc, how='left', on='product_id')
            actions = pd.merge(actions, labels, how='left', on=['user_id', 'product_group'])
            actions['label'].fillna(0, inplace=True)
            with tf.gfile.FastGFile(dump_path, 'wb') as gf:
                pickle.dump(actions, gf)
    
        labels = actions['label'].copy()
        del actions['user_id']
        del actions['product_group']
        del actions['label']
    
        print(actions.columns)
        return actions, labels
    
    
    def make_test_set(self,test_start_date, test_end_date):
        print('make test set', test_start_date, test_end_date)
        dump_path = os.path.join(self.cache_dir, 'test_set_%s_%s.pkl' % (test_start_date, test_end_date))
        if tf.gfile.Exists(dump_path):
            with tf.gfile.FastGFile(dump_path, 'rb') as gf:
                test_set = pickle.load(gf)
        else:
            test_set = self.gen_sample(test_end_date, test_end_date)
            browse_feat = self.user_browse_feature(test_start_date, test_end_date)
            invest_feat = self.user_invest_feature(test_start_date, test_end_date)
            # basic_user_feat = gen_basic_user_feat()
            # acc_user_feat = gen_accumulate_user_feat("2016-02-01", test_end_date)
            # acc_user_feat_all_cate = gen_accumulate_user_feat_all_cate("2016-02-01", test_end_date)
    
            #        window = [1, 2, 3, 5, 7, 10, 15, 21, 30]
            #        for i in window:
            #            start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(days=i)
            #            start_date = start_date.strftime('%Y-%m-%d')
            #            user_action_feat = gen_user_action_feat(start_date, test_end_date, i)
            #            test_set = pd.merge(test_set, user_action_feat, how='left', on='user_id')
            #
            #        window = [4, 8, 16]
            #        for i in window:
            #            start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(hours=i)
            #            user_action_feat = gen_user_action_feat_hour(start_date, test_end_date, i)
            #            test_set = pd.merge(test_set, user_action_feat, how='left', on='user_id')
    
            test_set = pd.merge(browse_feat, invest_feat, how='left', on=['user_id', 'product_group'])
            product_features = self.product_feat()
            test_set = pd.merge(test_set, product_features, how='left', on='product_group')
            test_set.fillna(0, inplace=True)
            with tf.gfile.FastGFile(dump_path, 'wb') as gf:
                pickle.dump(test_set, gf)
    
        index = test_set[['user_id']].copy()
        del test_set['user_id']
        print(test_set.columns)
        return index, test_set
    
    
    def train(self):
        train_start_date = '2017-12-01'
        train_end_date = '2017-12-31'
        act_start_date = '2018-01-01'
        act_end_date = '2018-01-31'
    
        test_start_date = '2018-01-01'
        test_end_date = '2018-01-31'
        test_act_start_date = '2018-02-01'
        test_act_end_date = '2018-02-29'
    
        train_X, train_Y = self.make_train_set(train_start_date, train_end_date, act_start_date, act_end_date)
        # train_X, train_Y = make_train_set_slide(train_start_date, train_end_date, act_start_date, act_end_date)
        test_index, test_X = self.make_test_set(test_start_date, test_end_date)
    
        print('training...')
        c = Counter(train_Y.values)
        clf = XGBClassifier(max_depth=5, min_child_weight=6, scale_pos_weight=c[0] / 16 / c[1], nthread=12, seed=0)
        clf.fit(train_X.values, train_Y.values)
    
        pre_y = clf.predict_proba(test_X.values)[:, 1]
        res = test_index.copy()
        res['prob'] = pre_y
    
    
    def train_sm(self):
        train_start_date = '2017-12-01'
        train_end_date = '2017-12-15'
        act_start_date = '2017-12-16'
        act_end_date = '2017-12-31'
    
        test_start_date = '2018-01-01'
        test_end_date = '2018-01-31'
        test_act_start_date = '2018-02-01'
        test_act_end_date = '2018-02-29'
    
        train_X, train_Y = self.make_train_set(train_start_date, train_end_date, act_start_date, act_end_date)
        # train_X, train_Y = make_train_set_slide(train_start_date, train_end_date, act_start_date, act_end_date)
        test_index, test_X = self.make_test_set(test_start_date, test_end_date)
    
        print('training...')
        c = Counter(train_Y.values)
        clf = XGBClassifier(max_depth=5, min_child_weight=6, scale_pos_weight=c[0] / 16 / c[1], nthread=12, seed=0)
        clf.fit(train_X.values, train_Y.values)
    
        pre_y = clf.predict_proba(test_X.values)[:, 1]
        res = test_index.copy()
        res['prob'] = pre_y
    
def main(_):
    args_in = sys.argv[1:]
    print(args_in)
    parser = argparse.ArgumentParser()
    mtyunArgs = parser.add_argument_group('美团云选项')
    mtyunArgs.add_argument('--data_dir', type=str, default='',
                           help='input data path')
    mtyunArgs.add_argument('--model_dir', type=str, default='',
                           help='output model path')
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
    r.train()
    
if __name__ == '__main__':
    tf.app.run(main=main)