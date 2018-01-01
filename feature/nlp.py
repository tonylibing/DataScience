# -*- coding: utf-8 -*-
import datetime
import json
import numbers
import random
import time
import operator
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
import scipy.stats.stats as stats
import xgboost as xgb
import lightgbm as lgb
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


class ChnTextProcessor(TransformerMixin):
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