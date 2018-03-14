# -*- coding: utf-8 -*-
import datetime
import json
import numbers
import random
import time
import os
import sys
import gc
import operator
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
import scipy.stats.stats as stats
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from pandas.io.json import json_normalize
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC
from statsmodels.stats.outliers_influence import variance_inflation_factor
from collections import Counter

def ColumnStats(df,cols):
    for col in cols:
        values = list(df[col].fillna('missing').value_counts().loc[lambda x: x.index != 'missing'].index)
        type_list = [type(i) for i in values]
        type_cnt = Counter(type_list)
        if len(type_cnt)>1:
            print('{} has multiple data types'.format(col))
            print(type_cnt)

def ColumnInfo(df, col):
    print(col)
    col_type = ''
    missing_pct = 0.0
    # rows = np.random.choice(df.index.values, 1000)
    # df_sample = df.ix[rows][col]
    df_sample = df[col].dropna()
    uniq_vals = list(set(df_sample))
    del(df_sample)
    if np.nan in uniq_vals:
        uniq_vals.remove(np.nan)

    if col.endswith('time'):
        col_type = 'time'
    elif col.endswith('_id'):
        col_type = 'id'
        # col_type = 'categorical'
        # or ('level' in col)
    elif col.startswith('dayOf') or col.startswith('hourOf') or col.startswith('is_') or ('category' in col)  or (
                'flag' in col) or ('version' in col):
        col_type = 'categorical'
    elif isinstance(uniq_vals[0], float):
        col_type = 'numerical'
    elif len(uniq_vals) >= 10 and isinstance(uniq_vals[0], numbers.Real):
        col_type = 'numerical'
    else:
        col_type = 'categorical'

    if (col_type == 'numerical'):
        missing_vals = df[col].map(lambda x: int(pd.isnull(x)))
        missing_pct = sum(missing_vals) * 1.0 / df.shape[0]
    elif (col_type == 'categorical'):
        missing_vals = df[col].map(lambda x: int(x != x))
        missing_pct = sum(missing_vals) * 1.0 / df.shape[0]

    return col, col_type, df[col].dtype, missing_pct


def ColumnSummary(df, label_col='label', id_cols=None,dump_path=None):
    column_info = pd.DataFrame([(ColumnInfo(df, col)) for col in tqdm(df.columns.values, desc='Columns Info')])
    column_info.columns = ['col_name', 'ColumnType', 'dtype', 'missing_pct']
    summary = df.describe(include='all').transpose()
    summary = summary.reset_index()
    # print(summary.columns)
    all = pd.merge(summary, column_info, left_on='index', right_on='col_name')
    del(summary)
    del(column_info)
    gc.collect()
    all.drop('col_name', axis=1)
    print("Column Summary")
    print(all)
    if dump_path is not None:
        with tf.gfile.FastGFile(dump_path, 'wb') as gf:
            all.to_csv(gf,index=False,header=True)

    return all


class FeatureSelection(TransformerMixin):
    def __init__(self,args=None):
        self.column_type = None
        self.variance_selector = VarianceThreshold()
        # self.variance_selector = VarianceThreshold(threshold=.001)
        self.scaler = RobustScaler()
        # self.scaler = StandardScaler()
        self.numerical_cols = []
        self.categorical_cols = []
        self.selected_cols = []
        self.args = None
        self.data_dir = None
        self.model_dir = None
        if args is not None:
            self.args = args
            self.data_dir =  self.args.data_dir
            self.model_dir = self.args.model_dir

    def fit(self, X, y):
        print("X shape:")
        print(X.shape)
        print("before feature selection")
        print(X.columns.values)
        origin_features = set(X.columns.values)
        print(len(X.columns.values))
        # drop missing too much columns
        summary_path = os.path.join(self.args.data_dir,'column_summary.csv')
        self.column_summary = ColumnSummary(X,dump_path=summary_path)
        self.column_type = self.column_summary.set_index('col_name')['ColumnType'].to_dict()
        drop_cols = self.column_summary[self.column_summary['missing_pct'] > 0.99]['col_name']

        if len(drop_cols) > 0:
            print("drop missing too much columns:{0}".format(drop_cols))
            # df.drop(drop_cols,axis=0,inplace=True)

        for col, col_type in self.column_type.items():
            if col_type == 'numerical':
                self.numerical_cols.append(col)
            elif col_type == 'categorical':
                self.categorical_cols.append(col)

        # feature selection
        print("numerical features:")
        print(self.numerical_cols)
        print("categorical features:")
        print(self.categorical_cols)
        df_cat = X[self.categorical_cols]
        df_nonna = X[self.numerical_cols].fillna(0)
        # df_nonna = X[self.numerical_cols].dropna()
        print("df_nonna shape:")
        print(df_nonna.shape)
        y_tmp = y[df_nonna.index]
        # print(df_nonna.index.values)
        # normalize numerical features
        # df_norm = pd.DataFrame(normalize(df_nonna,axis=0),columns=self.numerical_cols)
        df_norm = pd.DataFrame(self.scaler.fit_transform(df_nonna), columns=self.numerical_cols)
        df_numerical = self.variance_selector.fit_transform(df_norm)
        idxs_selected = self.variance_selector.get_support(indices=True)
        print("feature selected by > variance threshold:")
        print(df_nonna.columns[idxs_selected])
        print(len(df_nonna.columns[idxs_selected]))
        self.selected_cols = list(df_nonna.columns[idxs_selected]) + self.categorical_cols
        # chi2
        # df2 = df_nonna[df_nonna.columns[idxs_selected]]
        # selector = SelectKBest(chi2, k=int(len(df2.columns)*0.95))
        # df_numerical = selector.fit_transform(df2,y_tmp)
        # idxs_selected = selector.get_support(indices=True)
        # print("ch2 selection")
        # print(df2.columns[idxs_selected])
        # print(len(df2.columns[idxs_selected]))
        #
        # self.selected_cols=list(df2.columns[idxs_selected]) + self.categorical_cols
        print("after feature selection")
        print(self.selected_cols)
        print(len(self.selected_cols))
        print("dropped columns:")
        print(origin_features - set(self.selected_cols))

        return self

    def transform(self, X):
        return X[self.selected_cols]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class FeatureEncoder(BaseEstimator):
    """
    1. drop missing columns according to missing percentage, filter outliers
    2. detect column type
    3. binning numerical type feature and genrating new feature names
    4. one-hot encoding or woe encoding of categorical feature
    5. do 3,4 in parallel using FeatureUnion
    6. feature union
    7. to sparse feature matrix or dense
    """
    def __init__(self,args=None):
        self.column_type = None
        self.feature_matrix = None
        self.feature_processors = []
        self.variance_selector = VarianceThreshold(threshold=.99 * (1 - .99))
        self.scaler = StandardScaler()
        self.feature_names = {}
        self.numerical_cols = []
        self.categorical_cols = []
        self.args = None
        self.data_dir = None
        self.model_dir = None
        if args is not None:
            self.args = args
            self.data_dir =  self.args.data_dir
            self.model_dir = self.args.model_dir

    def fit(self, df):
        # column summary
        summary_path = os.path.join(self.args.data_dir,'column_summary.csv')
        self.column_summary = ColumnSummary(df,dump_path=summary_path)
        self.column_type = self.column_summary.set_index('col_name')['ColumnType'].to_dict()

        for col in df.columns.values:
            if self.column_type[col] == 'numerical':
                fp = ContinuousFeatureTransformer(col, self.column_type[col], {})
                self.feature_processors.append(fp)
            elif self.column_type[col] == 'categorical':
                fp = CategoricalFeatureTransformer(col, self.column_type[col], {})
                self.feature_processors.append(fp)

        print("=" * 60)
        for fp in self.feature_processors:
            print('col:{0},type:{1},params:{2}'.format(fp.col_name, fp.col_type, fp.params))
        print("=" * 60)

        self.feature_offset = {}
        self.feature_name = []
        self.length = 0
        for fp in self.feature_processors:
            fp.fit(df)
            self.feature_name.append(fp.col_name)
            self.feature_offset[fp.col_name] = self.length
            self.length += fp.dimension

        # StandardScaler
        # print("normalize numerical features:{0}".format(self.numerical_cols))
        # self.scaler.fit(self.df[self.numerical_cols])
        # df_numerical = self.scaler.transform(self.df[self.numerical_cols])
        # self.df = pd.concat([self.df[self.categorical_cols],df_numerical],axis=1)

        print("=" * 60)
        print("feature_offset:{0}".format(self.feature_offset))
        print("=" * 60)

        return self

    def transform(self, df):
        print("in transform")
        print(len(df.columns.values))
        print(ColumnSummary(df))
        for fp in self.feature_processors:
            fp.transform(df)
        df_tmp = df[self.feature_name]
        data = []
        row_idx = []
        col_idx = []
        for i, v in enumerate(df_tmp.values):
            for k, vv in enumerate(v):
                fp = self.feature_processors[k]
                if 'categorical' == fp.col_type:
                    if pd.isnull(vv) == False:
                        data.append(1.0)
                        row_idx.append(i)
                        col_idx.append(int(vv) + self.feature_offset[fp.col_name])
                        self.feature_names["{0}={1}".format(fp.col_name, fp.id2feature[int(vv)])] = int(vv) + \
                                                                                                    self.feature_offset[
                                                                                                        fp.col_name]
                else:
                    data.append(vv)
                    row_idx.append(i)
                    col_idx.append(self.feature_offset[fp.col_name])
                    self.feature_names[fp.col_name] = self.feature_offset[fp.col_name]
        data.append(0.0)
        row_idx.append(len(df_tmp) - 1)
        col_idx.append(self.length - 1)
        print(len(data))
        print(len(row_idx))
        print(len(col_idx))
        return csr_matrix((data, (row_idx, col_idx)))

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def persist(self):
        pass

    def load(self):
        pass

    def __str__(self):
        ps = ['col:{0},type:{1},params:{2}'.format(fp.col_name, fp.col_type, fp.params) for fp in
              self.feature_processors]
        processors = "\n".join(ps)
        sorted_x = sorted(self.feature_names.items(), key=operator.itemgetter(1))
        info = processors + "\n" + json.dumps(sorted_x)
        return info


class ContinuousFeatureGenerator:
    """
    Normalize the integer features to [0, 1] by min-max normalization
    """

    def __init__(self, num_feature):
        self.num_feature = num_feature
        self.min = [sys.maxsize] * num_feature
        self.max = [-sys.maxsize] * num_feature
        #gen continous clip
        self.continous_clip = []

    def build(self, datafile, continous_features):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_feature):
                    val = features[continous_features[i]]
                    if val != '':
                        val = int(val)
                        if val > self.continous_clip[i]:
                            val = self.continous_clip[i]
                        self.min[i] = min(self.min[i], val)
                        self.max[i] = max(self.max[i], val)

    def gen(self, idx, val):
        if val == '':
            return 0.0
        val = float(val)
        return (val - self.min[idx]) / (self.max[idx] - self.min[idx])


class OutliersFilter(TransformerMixin):
    def __init__(self, column, method="percentile", threshold=95):
        self.column = column
        self.method = method
        self.modified_z_score = None
        self.threshold = threshold

    def fit(self, X):
        if self.method == "percentile":
            diff = (100 - self.threshold) / 2.0
            minval, maxval = np.percentile(X[self.column], [diff, 100 - diff])
            self.minval = minval
            self.maxval = maxval
        elif self.method == "mad":
            if len(X[self.column].shape) == 1:
                points = X[self.column][:, None]
            median = np.median(points, axis=0)
            diff = np.sum((points - median) ** 2, axis=-1)
            diff = np.sqrt(diff)
            med_abs_deviation = np.median(diff)
            self.modified_z_score = 0.6745 * diff / med_abs_deviation

        return self

    def transform(self, X):
        if self.method == "percentile":
            return X.loc[(X[self.column] < self.minval) | (X[self.column] > self.maxval)]
        elif self.method == "mad":
            return X.loc[self.modified_z_score > self.threshold]

    def mad_based_outlier(points, thresh=3.5):
        if len(points.shape) == 1:
            points = points[:, None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median) ** 2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh

    def percentile_based_outlier(data, threshold=95):
        diff = (100 - threshold) / 2.0
        minval, maxval = np.percentile(data, [diff, 100 - diff])
        return (data < minval) | (data > maxval)

    def iqr_based_outlier(data):
        """
        For outlier detection, upper and lower fences (Q3 + 1.5IQR and Q1 − 1.5IQR) of the differences are often used in statistics, where Q1 is the lower 25% quantile, Q3 is the upper 25% quantile and IQR = Q3 − Q1.
        :param threshold:
        :return:
        """
        Q1, Q3 = np.percentile(data, [0.25, 0.75])
        IQR = Q3 - Q1
        minval = Q3 + 1.5 * IQR
        maxval = Q1 - 1.5 * IQR
        return (data < minval) | (data > maxval)


class MissingImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
                              index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill, inplace=True)

    def fit_transform(self, df, y=None):
        return self.fit(df, y).transform(df, y)


class ContinuousFeatureTransformer(TransformerMixin):
    def __init__(self, col_name, col_type, params):
        self.col_name = col_name
        self.col_type = col_type
        self.params = params
        self.fillmethod = 'median'
        if 'fillmethod' in self.params:
            self.fillmethod = self.params['fillmethod']

    def fit(self, df):
        if self.fillmethod == 'value':
            self.value = self.params['value']
        elif self.fillmethod == 'mean':
            self.value = df[self.col_name].mean()
        elif self.fillmethod == 'median':
            self.value = df[self.col_name].median()

        self.dimension = 1
        return self

    def transform(self, df):
        if self.fillmethod == 'random':
            missing_cnt = df.loc[pd.isnull(df[self.col_name])][self.col_name].size
            not_missing = df.loc[~pd.isnull(df[self.col_name])][self.col_name]
            rnd_value = not_missing.sample(n=missing_cnt)
            df.loc[pd.isnull(df[self.col_name])][self.col_name] = rnd_value
        else:
            df[self.col_name] = df[self.col_name].fillna(self.value)
        return df[self.col_name]

    def fit_transform(self, df):
        return self.fit(df).transform(df)


class CategoricalFeatureTransformer(TransformerMixin):
    def __init__(self, col_name, col_type, params):
        self.col_name = col_name
        self.col_type = col_type
        self.params = params
        self.feature2id = {}
        self.id2feature = {}
        # self.imputer = MissingImputer()

    def fit(self, df):
        # fill missing value with mod
        # mod_value = df[self.col_name].value_counts().index[0]
        # print("{0} mod value:{1}".format(self.col_name,mod_value))
        # df[self.col_name].fillna(mod_value,inplace=True)
        # generate feature index mapping
        idx = 0
        self.feature2id = {}
        self.id2feature = {}
        for item in df[self.col_name].astype(str).unique():
            self.id2feature[idx] = item
            self.feature2id[item] = idx
            idx += 1

        self.dimension = idx
        print("col_name:{0},col_type:{1},feature2id:{2}".format(self.col_name, self.col_type, self.feature2id))
        return self

    def transform(self, df):
        df[self.col_name] = df[self.col_name].astype(str).map(self.feature2id)
        return df[self.col_name]

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def __str__(self):
        return "col_name:{0},col_type:{1},feature2id:{2}".format(self.col_name, self.col_type, self.feature2id)


class QuantileBinarizer(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y):
        return self

    def transform(self, X):
        X2 = X.fillna(np.median(X))
        r = 0
        while np.abs(r) < 1:
            d1 = pd.DataFrame({"X": X2, "Y": Y, "Bucket": pd.qcut(X2, n)})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1
        d3 = pd.DataFrame(d2.min().X, columns=['min_' + X.name])
        d3['max_' + X.name] = d2.max().X
        d3[Y.name] = d2.sum().Y
        d3['total'] = d2.count().Y
        d3[Y.name + '_rate'] = d2.mean().Y
        d4 = (d3.sort_index(by='min_' + X.name)).reset_index(drop=True)
        print("=" * 60)
        print(d4)
        return X[self.columns]


class CommonBinarizer(TransformerMixin):
    def __init__(self, columns, cut_percent_point=None):
        self.columns = columns
        self.cut_percent_point = cut_percent_point

    def transform(self, df):
        data = []
        row_idx = []
        col_idx = []
        for i, v in enumerate(df[self.cols].values):
            for k, vv in enumerate(v):
                col_key = self.cols[k] + '$' + str(vv)
                if col_key in self.feature2id:
                    data.append(1.0)
                    row_idx.append(i)
                    col_idx.append(self.feature2id[col_key])
        return csr_matrix((data, (row_idx, col_idx)))

    def fit(self, df, col_type_info):
        if self.cut_percent_point is None:
            self.cut_percent_point = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
        # self.cut_percent_point = cut_percent_point
        self.col_info = {}
        self.feature_assign_idx = 0
        self.feature2id = {}
        self.id2feature = {}
        df_tmp = pd.DataFrame()
        for k, v in tqdm(col_type_info.items(), desc='Binning and One-Hot Encoding'):
            if v in ('numerical', 'categorical'):
                #                print('col:',k)
                self.col_info[k] = {'type': v}
                info_dict = self.col_info[k]
                # if col_nan_value.has_key(k):
                #    info_dict['nan'] = col_nan_value[k]
                #    df.loc[ df[k].map(lambda x: x in info_dict['nan']), k ] = np.nan
                if v == 'numerical':
                    all_seg = self.get_cut_point(df[k], self.cut_percent_point)
                    info_dict['cut_value'] = all_seg
                    df_tmp[k] = pd.cut(df[k], bins=all_seg, include_lowest=True, right=True).astype(str)
                elif v == 'categorical':
                    df_tmp[k] = df[k].astype(str)

                info_dict['unique'] = df_tmp[k].unique()
                feature_width = len(info_dict['unique'])
                info_dict['idx_start'] = self.feature_assign_idx
                info_dict['idx_end'] = info_dict['idx_start'] + feature_width
                for v in info_dict['unique']:
                    self.feature2id[k + '$' + str(v)] = self.feature_assign_idx
                    self.id2feature[self.feature_assign_idx] = k + '$' + str(v)
                    self.feature_assign_idx += 1
        self.cols = list(self.col_info.keys())
        x = self.transform(df_tmp)
        return x

    def get_cut_point(self, series, cut_percent_point):
        values = series.quantile(cut_percent_point).values
        seg = sorted(np.unique(values))
        all_seg = [-100000000]
        all_seg.extend(list(seg))
        all_seg.append(100000000)
        return all_seg


class MonotonicBinarizer(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y):
        return self

    def transform(self, X, Y):
        X2 = X.fillna(np.median(X))
        r = 0
        while np.abs(r) < 1:
            d1 = pd.DataFrame({"X": X2, "Y": Y, "Bucket": pd.qcut(X2, n)})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1
        d3 = pd.DataFrame(d2.min().X, columns=['min_' + X.name])
        d3['max_' + X.name] = d2.max().X
        d3[Y.name] = d2.sum().Y
        d3['total'] = d2.count().Y
        d3[Y.name + '_rate'] = d2.mean().Y
        d4 = (d3.sort_index(by='min_' + X.name)).reset_index(drop=True)
        print("=" * 60)
        print(d4)
        return X[self.columns]


class LogTransformer(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[self.columns]


class PowerTransformer(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[self.columns]


class LabelTransformer(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.encoders_dict = defaultdict(LabelEncoder)

    def transform(self, X, *_):
        X_1 = pd.DataFrame(X.copy())
        X_1 = X_1.apply(lambda x: self.encoders_dict[x.name].fit_transform(x) if x.name in self.columns else x)
        return X_1

    def fit(self, *_):
        return self


class MultiColumnLabelEncoder(TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def fit(self, X, y=None):
        return self  # not relevant here

    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class OrdinalTransformer(TransformerMixin):
    def __init__(self, ord_col_map):
        self.ord_col_map = ord_col_map

    def transform(self, X, *_):
        X_1 = X.copy()
        for col in X.columns:
            mp = self.ord_col_map[col]
            print("mp", mp)
            X_1[col] = X_1[col].map(lambda x: x.lower() if (type(x) == str) else x)
            X_1[col] = X_1[col].map(lambda x: mp[x] if x in mp else float('nan'))
        return X_1

    def fit(self, *_):
        return self


class DropColumnTransformer(TransformerMixin):
    def __init__(self, columns, missing_threshold=0.9, dev_threshold=0.1):
        self.columns = columns
        self.missing_threshold = missing_threshold
        self.dev_threshold = dev_threshold

    def transform(self, df):
        column_info = pd.DataFrame([(ColumnInfo(df, col)) for col in df.columns.values])
        column_info.columns = ['col_name', 'ColumnType', 'missing_pct']
        summary = df.describe(include='all').transpose()
        summary = summary.reset_index()
        print(summary.columns)
        all = pd.merge(summary, column_info, left_on='index', right_on='col_name')
        misssing_too_much_cols = all.loc[all['missing_pct'] > self.missing_threshold]['col_name'].values.tolist()
        for i in misssing_too_much_cols:
            print('drop column {0} due to missing percentage'.format(i))

        low_variance_cols = all.loc[all['std'] < self.dev_threshold]['col_name'].values.tolist()
        for i in low_variance_cols:
            print('drop column {0} due to low variance'.format(i))

        drop_cols = misssing_too_much_cols + low_variance_cols
        return df.drop(drop_cols, axis=1, inplace=False)

    def fit(self, *_):
        return self


class RowMissingDroperTransformer(TransformerMixin):
    def __init__(self, threshold=95):
        self.threshold = threshold

    def transform(self, df):
        # method 1
        row_missing = df.isnull().sum(axis=1)
        diff = (100 - self.threshold) / 2.0
        minval, maxval = np.percentile(row_missing, [diff, 100 - diff])
        # method 2
        # row_missing = df.isnull().sum(axis=1)
        # q1,median,q2 = np.percentile(row_missing, [0.25,0.5,0.75])
        # #bug when q1=median=q2=0
        # maxval = median+(q2-q1)*1.0/3
        return df.drop(row_missing.index[row_missing > maxval], axis=0, inplace=False)

    def fit(self, *_):
        return self


class LowerTransformer(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def transform(self, df):
        for col in self.columns:
            df[col] = df[col].map(lambda x: x.lower())

        return df

    def fit(self, *_):
        return self


class StripTransformer(TransformerMixin):
    def __init__(self, columns, strip_str=None):
        self.columns = columns
        self.strip_str = strip_str

    def transform(self, df):
        if self.strip_str == None:
            for col in self.columns:
                df[col] = df[col].map(lambda x: x.strip())
        else:
            for col in self.columns:
                df[col] = df[col].map(lambda x: x.strip(self.strip_str))
        return df

    def fit(self, *_):
        return self


class ImportantColMissingDropTransformer(TransformerMixin):
    # columns selected according to feature importance
    def __init__(self, columns=None, threshold=95):
        self.columns = columns
        self.threshold = threshold

    def transform(self, df):
        if self.columns == None:
            row_missing = df.isnull().sum(axis=1)
            diff = (100 - self.threshold) / 2.0
            minval, maxval = np.percentile(row_missing, [diff, 100 - diff])
            return df.drop(row_missing.index[row_missing > maxval], axis=0, inplace=False)
        else:
            row_missing = df[self.columns].isnull().sum(axis=1)
            diff = (100 - self.threshold) / 2.0
            minval, maxval = np.percentile(row_missing, [diff, 100 - diff])
            return df.drop(row_missing.index[row_missing > maxval], axis=0, inplace=False)

    def fit(self, *_):
        return self


class ImputingMissingTransformer(TransformerMixin):
    def __init__(self, transformer, empty_values=[float('nan'), np.NaN, None]):
        self.transformer = transformer
        self.empty_values = empty_values

    def transform(self, X, *_):
        impute = np.vectorize(lambda x: 1 if x in self.empty_values else 0)
        replcae_b4 = np.vectorize(lambda x: np.NaN if x in self.empty_values else x)

        imputed = pd.DataFrame(impute(X))
        X = replcae_b4(X)
        date_frm = pd.DataFrame(self.transformer.fit(X).transform(X))
        return pd.DataFrame(pd.merge(date_frm, imputed, right_index=True, left_index=True))

    def fit(self, *_):
        return self


### The function making up missing values in Continuous or Categorical variable
def ProcessExtremeAndMissingTransformer(TransformerMixin):
    def fit(self, *_):
        return self

    def transform(df, col, type, method):
        '''
        :param df: dataset containing columns with missing value
        :param col: columns with missing value
        :param type: the type of the column, should be Continuous or Categorical
        :return: the made up columns
        '''
        # Take the sample with non-missing value in col
        validDf = df.loc[df[col] == df[col]][[col]]
        if validDf.shape[0] == df.shape[0]:
            return 'There is no missing value in {}'.format(col)

        # copy the original value from col to protect the original dataframe
        missingList = [i for i in df[col]]
        if type == 'Continuous':
            if method not in ['Mean', 'Random', 'Median']:
                return 'Please specify the correct treatment method for missing continuous variable!'
            # get the descriptive statistics of col
            descStats = validDf[col].describe()
            mu = descStats['mean']
            std = descStats['std']
            median = validDf[col].median()
            maxVal = descStats['max']
            # detect the extreme value using 3-sigma method
            if maxVal > mu + 3 * std:
                for i in list(validDf.index):
                    if validDf.loc[i][col] > mu + 3 * std:
                        # decrease the extreme value to normal level
                        validDf.loc[i][col] = mu + 3 * std
                # re-calculate the mean based on cleaned data
                mu = validDf[col].describe()['mean']
            for i in range(df.shape[0]):
                if df.loc[i][col] != df.loc[i][col]:
                    # use the mean or sampled data to replace the missing value
                    if method == 'Mean':
                        missingList[i] = mu
                    elif method == 'Random':
                        missingList[i] = random.sample(validDf[col], 1)[0]
        elif type == 'Categorical':
            if method not in ['Mode', 'Random']:
                return 'Please specify the correct treatment method for missing categorical variable!'
            # calculate the probability of each type of the categorical variable
            freqDict = {}
            recdNum = validDf.shape[0]
            for v in set(validDf[col]):
                vDf = validDf.loc[validDf[col] == v]
                freqDict[v] = vDf.shape[0] * 1.0 / recdNum
            # find the category with highest probability
            modeVal = max(freqDict.items(), key=lambda x: x[1])[0]
            freqTuple = freqDict.items()
            # cumulative sum of each category
            freqList = [0] + [i[1] for i in freqTuple]
            freqCumsum = np.cumsum(freqList)
            for i in range(df.shape[0]):
                if df.loc[i][col] != df.loc[i][col]:
                    if method == 'Mode':
                        missingList[i] = modeVal
                    if method == 'Random':
                        # determine the sampled category using uniform distributed random variable
                        a = random.random(1)
                        position = \
                            [k + 1 for k in range(len(freqCumsum) - 1) if freqCumsum[k] < a <= freqCumsum[k + 1]][0]
                        missingList[i] = freqTuple[position - 1][0]
        print('The missing value in {0} has been made up with the mothod of {1}'.format(col, method))
        return missingList


class ModelTransformer(TransformerMixin):
    def __init__(self, model):
        self.model = model

    def transform(self, X, *_):
        return self.model.fit_transform(X)

    def fit(self, X, y=None):
        self.model.fit(X, y)
        return self


### convert the date variable into the days
def DateGapGenerator(TransformerMixin):
    '''
    :param df: the dataset containing date variable in the format of 2017/1/1
    :param date: the column of date
    :param base: the base date used in calculating day gap
    :return: the days gap
    '''

    def __init__(self, col, base):
        self.col = col
        self.base = base

    def fit(self, X):
        return self

    def transform(self, X):
        base2 = time.strptime(self.base, '%Y/%m/%d')
        base3 = datetime.datetime(base2[0], base2[1], base2[2])
        date1 = [time.strptime(i, '%Y/%m/%d') for i in X[self.col]]
        date2 = [datetime.datetime(i[0], i[1], i[2]) for i in date1]
        daysGap = [(date2[i] - base3).days for i in range(len(date2))]
        return daysGap


class HourOfDayTransformer(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[self.columns]


class DayOfWeekTransformer(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[self.columns]


class DayOfMonthTransformer(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[self.columns]


class DateTimeTransformer(TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X):
        return self

    def transform(self, X):
        from dateutil.parser import parse
        # X[self.column].apply(lambda x: pd.Series({'dayOfWeek':parse(x).weekday(),'hourOfDay':parse(x).hour,'dayOfMonth':parse(x).day})
        X['dayOfWeek'] = X[self.column].apply(lambda x: parse(x).weekday())
        X['hourOfDay'] = X[self.column].apply(lambda x: parse(x).hour)
        X['dayOfMonth'] = X[self.column].apply(lambda x: parse(x).day)
        return X


class FlattenJsonTransformer(TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X, columns):
        if columns is None:
            return X

        for col in columns:
            X_tmp = json_normalize(X[col].apply(lambda x: json.loads(x)).tolist())
            X = pd.concat([X, X_tmp], axis=1)

        X.drop(columns, axis=1, inplace=True)
        return X

    def fit_transform(self, X, columns):
        return self.fit(X).transform(X, columns)


class L1KBestSelector(TransformerMixin):
    def __init__(self):
        self.lsvc = LinearSVC(C=0.01, penalty="l1", dual=False)

    def fit(self, X_scaler, y):
        self.lsvc.fit(X_scaler, y)
        return self

    def transform(self, X_scaler, y):
        lsvc = self.lsvc.fit(X_scaler, y)
        model = SelectFromModel(lsvc, prefit=True)
        X_lsvc = model.transform(X_scaler)
        return X_lsvc

    def fit_transform(self, X_scaler, y):
        lsvc = self.lsvc.fit(X_scaler, y)
        model = SelectFromModel(lsvc, prefit=True)
        X_lsvc = model.transform(X_scaler)
        return X_lsvc


class ReduceVIF(BaseEstimator, TransformerMixin):
    """
    ref:https://www.kaggle.com/ffisegydd/sklearn-multicollinearity-class/code/notebook
    """

    def __init__(self, thresh=5.0, impute=True, impute_strategy='median'):
        # From looking at documentation, values between 5 and 10 are "okay".
        # Above 10 is too high and so should be removed.
        self.thresh = thresh

        # The statsmodel function will fail with NaN values, as such we have to impute them.
        # By default we impute using the median value.
        # This imputation could be taken out and added as part of an sklearn Pipeline.
        if impute:
            self.imputer = Imputer(strategy=impute_strategy)

    def fit(self, X, y=None):
        print('ReduceVIF fit')
        if hasattr(self, 'imputer'):
            self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        print('ReduceVIF transform')
        columns = X.columns.tolist()
        if hasattr(self, 'imputer'):
            X = pd.DataFrame(self.imputer.transform(X), columns=columns)
        return ReduceVIF.calculate_vif(X, self.thresh)

    @staticmethod
    def calculate_vif(X, thresh=5.0):
        # Taken from https://stats.stackexchange.com/a/253620/53565 and modified
        dropped = True
        while dropped:
            variables = X.columns
            dropped = False
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]

            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                print('Dropping' + X.columns[maxloc] + 'with vif=' + max_vif)
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped = True
        return X


