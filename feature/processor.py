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
from sklearn.preprocessing import Imputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
import datetime
import time

def ColumnInfo(df,col):
    col_type=''
    missing_pct = 0.0
    uniq_vals = list(set(df[col]))
    if np.nan in uniq_vals:
        uniq_vals.remove(np.nan)
    if "id" in col:
        col_type = 'id'
    if len(uniq_vals) >= 10 and isinstance(uniq_vals[0], numbers.Real):
        col_type = 'numerical'
    else:
        col_type = 'categorical'

    if(col_type=='numerical' or col_type=='id'):
        missing_vals = df[col].map(lambda x: int(np.isnan(x)))
        missing_pct = sum(missing_vals) * 1.0 / df.shape[0]
    elif(col_type=='categorical'):
        missing_vals = df[col].map(lambda x: int(x != x))
        missing_pct = sum(missing_vals) * 1.0 / df.shape[0]

    return col,col_type,missing_pct

def ColumnSummary(df):
    column_info = pd.DataFrame([(ColumnInfo(df,col)) for col in df.columns.values])
    column_info.columns = ['col_name', 'ColumnType','missing_pct']
    summary = df.describe(include='all').transpose()
    summary = summary.reset_index()
    print(summary.columns)
    all = pd.merge(summary, column_info, left_on='index', right_on='col_name')
    all.drop('col_name',axis=1)
    all.to_csv('colummn_summary.csv')
    return all


class DataProcessor(BaseEstimator):
    def __init__(self, df):
        self.df = df
        self.column_summray = None
        self.transformers = None
        self.feature_matrix = None


    def fit(self, *_):
        self.column_summray = ColumnSummary(self.df)
        return self

    def transform(self, X):
        return X[self.columns]

    def persist(self):
        pass

    def load(self):
        pass


class ColumnExtractor(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, *_):
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

    def iqr_based_outlier(data):
        """
        For outlier detection, upper and lower fences (Q3 + 1.5IQR and Q1 − 1.5IQR) of the differences are often used in statistics, where Q1 is the lower 25% quantile, Q3 is the upper 25% quantile and IQR = Q3 − Q1.
        :param threshold:
        :return:
        """
        Q1, Q3 = np.percentile(data, [0.25,0.75])
        IQR=Q3-Q1
        minval = Q3+1.5*IQR
        maxval = Q1-1.5*IQR
        return (data < minval) | (data > maxval)


class ContinuousFeatureTransformer(TransformerMixin):
    def __init__(self, columns):
        self.fillmethod = self.parameters['fillmethod']

    def fit(self, df):
        self.initialize()
        if self.fillmethod == 'value':
            self.value = self.parameters['value']
        elif self.fillmethod == 'mean':
            self.value = df[self.col].mean()
        elif self.fillmethod == 'median':
            self.value = df[self.col].median()
        self.dimension = 1

    def transform(self, df):
        if self.fillmethod == 'random':
            missing_cnt = df.loc[np.isnan(df[col])][col].size
            not_missing = df.loc[~np.isnan(df[col])][col]
            rnd_value = not_missing.sample(n=missing_cnt)
            df.loc[np.isnan(df[col])][col] = rnd_value
        else:
            df[self.col_name] = df[self.col].fillna(self.value)


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
            print("mp",mp)
            X_1[col] = X_1[col].map(lambda x : x.lower() if (type(x) == str) else x)
            X_1[col] = X_1[col].map(lambda x : mp[x] if x in mp else float('nan'))
        return X_1

    def fit(self, *_):
        return self

class DropColumnTransformer(TransformerMixin):
    def __init__(self, columns, missing_threshold=0.9, dev_threshold = 0.1):
        self.columns = columns
        self.missing_threshold = missing_threshold
        self.dev_threshold = dev_threshold

    def transform(self,df):
        column_info = pd.DataFrame([(ColumnInfo(df, col)) for col in df.columns.values])
        column_info.columns = ['col_name', 'ColumnType', 'missing_pct']
        summary = df.describe(include='all').transpose()
        summary = summary.reset_index()
        print(summary.columns)
        all = pd.merge(summary, column_info, left_on='index', right_on='col_name')
        misssing_too_much_cols = all.loc[all['missing_pct']>self.missing_threshold]['col_name'].values.tolist()
        for i in misssing_too_much_cols:
            print('drop column {0} due to missing percentage'.format(i))

        low_variance_cols =   all.loc[all['std']<self.dev_threshold]['col_name'].values.tolist()
        for i in low_variance_cols:
            print('drop column {0} due to low variance'.format(i))

        drop_cols = misssing_too_much_cols + low_variance_cols
        return df.drop(drop_cols,axis = 1,inplace=False)

    def fit(self, *_):
        return self

class RowMissingDroperTransformer(TransformerMixin):
    def __init__(self,  threshold = 95):
        self.threshold = threshold

    def transform(self,df):
        #method 1
        row_missing = df.isnull().sum(axis=1)
        diff = (100 - self.threshold) / 2.0
        minval, maxval = np.percentile(row_missing, [diff, 100 - diff])
        #method 2
        # row_missing = df.isnull().sum(axis=1)
        # q1,median,q2 = np.percentile(row_missing, [0.25,0.5,0.75])
        # #bug when q1=median=q2=0
        # maxval = median+(q2-q1)*1.0/3
        return df.drop(row_missing.index[row_missing>maxval],axis=0,inplace = False)


    def fit(self, *_):
        return self

class LowerTransformer(TransformerMixin):
    def __init__(self,  columns):
        self.columns = columns

    def transform(self,df):
        for col in self.columns:
            df[col] = df[col].map(lambda x:x.lower())

        return df


    def fit(self, *_):
        return self

class StripTransformer(TransformerMixin):
    def __init__(self,  columns, strip_str = None):
        self.columns = columns
        self.strip_str = strip_str

    def transform(self,df):
        if self.strip_str==None:
            for col in self.columns:
                df[col] = df[col].map(lambda x:x.strip())
        else:
            for col in self.columns:
                df[col] = df[col].map(lambda x:x.strip(self.strip_str))
        return df


    def fit(self, *_):
        return self


class ImportantColMissingDropTransformer(TransformerMixin):
    #columns selected according to feature importance
    def __init__(self,  columns=None,threshold = 95):
        self.columns = columns
        self.threshold = threshold

    def transform(self,df):
        if self.columns==None:
            row_missing = df.isnull().sum(axis=1)
            diff = (100 - self.threshold) / 2.0
            minval, maxval = np.percentile(row_missing, [diff, 100 - diff])
            return df.drop(row_missing.index[row_missing>maxval],axis=0,inplace = False)
        else:
            row_missing = df[self.columns].isnull().sum(axis=1)
            diff = (100 - self.threshold) / 2.0
            minval, maxval = np.percentile(row_missing, [diff, 100 - diff])
            return df.drop(row_missing.index[row_missing>maxval],axis=0,inplace = False)


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
            if method not in ['Mean','Random','Median']:
                return 'Please specify the correct treatment method for missing continuous variable!'
            #get the descriptive statistics of col
            descStats = validDf[col].describe()
            mu = descStats['mean']
            std = descStats['std']
            median = validDf[col].median()
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
            freqCumsum = np.cumsum(freqList)
            for i in range(df.shape[0]):
                if df.loc[i][col] != df.loc[i][col]:
                    if method == 'Mode':
                        missingList[i] = modeVal
                    if method == 'Random':
                        #determine the sampled category using uniform distributed random variable
                        a = random.random(1)
                        position = [k+1 for k in range(len(freqCumsum)-1) if freqCumsum[k]<a<=freqCumsum[k+1]][0]
                        missingList[i] = freqTuple[position-1][0]
        print('The missing value in {0} has been made up with the mothod of {1}'.format(col, method))
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
                print('Dropping'+ X.columns[maxloc] +'with vif='+max_vif)
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped = True
        return X
        
        
        
        





