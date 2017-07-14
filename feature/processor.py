from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import scipy.stats.stats as stats



class ColumnExtractor(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[self.columns]

class OutliersFilter(TransformerMixin):
    def __init__(self, column, method="percentile", threshold = 95):
        self.columns = column
        self.method = method
        self.threshold = threshold

    def fit(self, X):
        if self.method == "percentile":
            diff = (100 - self.threshold) / 2.0
            minval, maxval = np.percentile(X[self.column], [diff, 100 - diff])
            self.minval = minval
            self.maxval = maxval
        elif self.method=="mad":
            if len(X[self.column].shape) == 1:
                points = X[self.column][:, None]
            median = np.median(points, axis=0)
            diff = np.sum((points - median) ** 2, axis=-1)
            diff = np.sqrt(diff)
            med_abs_deviation = np.median(diff)

            modified_z_score = 0.6745 * diff / med_abs_deviation

            return modified_z_score > self.threshold
        return self

    def transform(self, X):
        return X.loc[(X[self.column] < self.minval) | (X[self.column] > self.maxval)]

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

class ContinuousFeatureTransformer(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[self.columns]

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
        print "=" * 60
        print d4
        return X[self.columns]

class MonotonicBinarizer(TransformerMixin):
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
        print "=" * 60
        print d4
        return X[self.columns]

class CategoricalFeatureTransformer(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[self.columns]

class LabelTransformer(TransformerMixin):

    def transform(self, X, *_):
        X_1 = pd.DataFrame(X.copy())
        X_1 = X_1.apply(LabelEncoder().fit_transform)
        return X_1

    def fit(self, *_):
        return self

class MultiLabelTransformer(TransformerMixin):

    def transform(self, X, *_):
        X_1 = pd.DataFrame(X.copy())
        X_1 = X_1.apply(LabelEncoder().fit_transform)
        return X_1

    def fit(self, *_):
        return self

class OrdinalTransformer(TransformerMixin):
    def __init__(self, ord_col_map):
        self.ord_col_map = ord_col_map

    def transform(self, X, *_):
        X_1 = X.copy()
        for col in X.columns:
            mp = self.ord_col_map[col]
            print "mp",mp
            X_1[col] = X_1[col].map(lambda x : x.lower() if (type(x) == str) else x)
            X_1[col] = X_1[col].map(lambda x : mp[x] if x in mp else float('nan'))
        return X_1

    def fit(self, *_):
        return self

class LogTransformer(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[self.columns]

class ImputingMissingTransformer(TransformerMixin):
    def __init__(self, transformer, empty_values = [float('nan'), np.NaN, None]):
        self.transformer = transformer
        self.empty_values = empty_values

    def transform(self, X, *_):
        impute = np.vectorize(lambda x :  1 if x in self.empty_values else 0)
        replcae_b4 = np.vectorize(lambda x : np.NaN  if x in self.empty_values else x)

        imputed = pd.DataFrame(impute(X))
        X = replcae_b4(X)
        date_frm = pd.DataFrame(self.transformer.fit(X).transform(X))
        return pd.DataFrame(pd.merge(date_frm, imputed, right_index= True, left_index = True))

    def fit(self, *_):
        return self


class ModelTransformer(TransformerMixin):
    def __init__(self, model) :
        self.model = model

    def transform(self, X, *_):
        return self.model.fit_transform(X)


    def fit(self, X, y =None):
        self.model.fit(X,y)
        return self


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

####################################################################################