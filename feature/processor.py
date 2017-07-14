from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import scipy.stats.stats as stats
from collections import defaultdict
import random
import operator
import numbers
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import datetime
import time


class ColumnExtractor(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[self.columns]

class OutliersFilter(TransformerMixin):
    def __init__(self, column, method="percentile", threshold = 95):
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
        elif self.method=="mad":
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
        elif self.method=="mad":
            return X.loc[self.modified_z_score>self.threshold]
																								

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

    def transform(self, X,Y):
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

class CategoricalFeatureTransformer(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[self.columns]

class LabelTransformer(TransformerMixin):
    def __init__(self,columns):
        self.columns = columns
        self.encoders_dict = defaultdict(LabelEncoder)

    def transform(self, X, *_):
        X_1 = pd.DataFrame(X.copy())
        X_1 = X_1.apply(lambda x: self.encoders_dict[x.name].fit_transform(x) if x.name in self.columns else x)
        return X_1

    def fit(self, *_):
        return self

class MultiColumnLabelEncoder(TransformerMixin):
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
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
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

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

### The function making up missing values in Continuous or Categorical variable
def ProcessExtremeAndMissingTransformer(TransformerMixin):
    def fit(self, *_):
        return self

    def transform(df,col,type,method):
        '''
        :param df: dataset containing columns with missing value
        :param col: columns with missing value
        :param type: the type of the column, should be Continuous or Categorical
        :return: the made up columns
        '''
        #Take the sample with non-missing value in col
        validDf = df.loc[df[col] == df[col]][[col]]
        if validDf.shape[0] == df.shape[0]:
            return 'There is no missing value in {}'.format(col)

        #copy the original value from col to protect the original dataframe
        missingList = [i for i in df[col]]
        if type == 'Continuous':
            if method not in ['Mean','Random']:
                return 'Please specify the correct treatment method for missing continuous variable!'
            #get the descriptive statistics of col
            descStats = validDf[col].describe()
            mu = descStats['mean']
            std = descStats['std']
            maxVal = descStats['max']
            #detect the extreme value using 3-sigma method
            if maxVal > mu+3*std:
                for i in list(validDf.index):
                    if validDf.loc[i][col] > mu+3*std:
                        #decrease the extreme value to normal level
                        validDf.loc[i][col] = mu + 3 * std
                #re-calculate the mean based on cleaned data
                mu = validDf[col].describe()['mean']
            for i in range(df.shape[0]):
                if df.loc[i][col] != df.loc[i][col]:
                    #use the mean or sampled data to replace the missing value
                    if method == 'Mean':
                        missingList[i] = mu
                    elif method == 'Random':
                        missingList[i] = random.sample(validDf[col],1)[0]
        elif type == 'Categorical':
            if method not in ['Mode', 'Random']:
                return 'Please specify the correct treatment method for missing categorical variable!'
            #calculate the probability of each type of the categorical variable
            freqDict = {}
            recdNum = validDf.shape[0]
            for v in set(validDf[col]):
                vDf = validDf.loc[validDf[col] == v]
                freqDict[v] = vDf.shape[0] * 1.0 / recdNum
            #find the category with highest probability
            modeVal = max(freqDict.items(), key=lambda x: x[1])[0]
            freqTuple = freqDict.items()
            # cumulative sum of each category
            freqList = [0]+[i[1] for i in freqTuple]
            freqCumsum = cumsum(freqList)
            for i in range(df.shape[0]):
                if df.loc[i][col] != df.loc[i][col]:
                    if method == 'Mode':
                        missingList[i] = modeVal
                    if method == 'Random':
                        #determine the sampled category using unifor distributed random variable
                        a = random.random(1)
                        position = [k+1 for k in range(len(freqCumsum)-1) if freqCumsum[k]<a<=freqCumsum[k+1]][0]
                        missingList[i] = freqTuple[position-1][0]
        print 'The missing value in {0} has been made up with the mothod of {1}'.format(col, method)
        return missingList

class ModelTransformer(TransformerMixin):
    def __init__(self, model) :
        self.model = model

    def transform(self, X, *_):
        return self.model.fit_transform(X)


    def fit(self, X, y =None):
        self.model.fit(X,y)
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

    def transform(self, X ):
        base2 = time.strptime(self.base,'%Y/%m/%d')
        base3 = datetime.datetime(base2[0],base2[1],base2[2])
        date1 = [time.strptime(i,'%Y/%m/%d') for i in X[self.col]]
        date2 = [datetime.datetime(i[0],i[1],i[2]) for i in date1]
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

####################################################################################