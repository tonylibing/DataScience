from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import PolynomialFeatures

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import processor
reload(processor)
from processor import *



classification_model= reload(classification_model)
path= 'D:/workspace/DataScience/data/ppd/'
#path= '/home/tanglek/workspace/DataScience/data/ppd/'
data1 = pd.read_csv(path+'Training Set/PPD_LogInfo_3_1_Training_Set.csv', header = 0)
data2 = pd.read_csv(path+'Training Set/PPD_Training_Master_GBK_3_1_Training_Set.csv', header = 0,encoding = 'gbk')
data3 = pd.read_csv(path+'Training Set/PPD_Userupdate_Info_3_1_Training_Set.csv', header = 0)

def process_data(df):
    data1_pip = Pipeline([('drop_missing',DropColumnTransformer()),
                         ('drop_row',RowMissingDroperTransformer())])

    data3_pip = Pipeline([('lower_string',LowerTransformer('UserupdateInfo1'))])

    data2_pip = Pipeline([('lower_string',StripTransformer('UserInfo_9'))])

    d1=data1_pip.fit(data1)
    d2 = data2_pip.fit(data2)
    d3=data3_pip.fit(data3)





#features with date, category manipulation without feature importance
def get_basic_features(df, ordinal, categorical,date_manip ,cont):
    date_pip = Pipeline([('extract',column_extractor.column_extractor(date_manip)),
                ('date_manip', date_transformer.date_transformer('%Y-%m-%d')),
                ('d-m-y-q-dow', FeatureUnion([('day',date_transformer.day_of_month_transformer()),
                                            ('month',date_transformer.month_transformer()),
                                            ('dow',date_transformer.day_of_week_transformer()),
                                            ('quarter',date_transformer.month_quarter_transformer()),
                                            ('year',date_transformer.year_transformer())])),
                ('impute',imputing_transformer.imputing_transformer(Imputer(strategy='most_frequent')))])

    continuous = Pipeline([
            ('extract', column_extractor.column_extractor(cont)),
            ('impute',imputing_transformer.imputing_transformer(Imputer(strategy='most_frequent'))),
            ('scale', Normalizer())])

    ordinal_pip = Pipeline([('extract', column_extractor.column_extractor(ordinal)),
                            ('ord', categorical_transformer.ordinal_transformer(ordinal)),
                           ('impute',imputing_transformer.imputing_transformer(Imputer(strategy='most_frequent')))])

    one_hot = Pipeline([('extract', column_extractor.column_extractor(categorical)),
                        ('lab_enc', categorical_transformer.label_transformer()),
                        ('one_hot', ModelTransformer.ModelTransformer(OneHotEncoder(sparse=False)))])


    features = Pipeline([('parallel', FeatureUnion([('date',date_pip),
                                                    ('continuous',continuous),
                                                    ('ordinal_pip',ordinal_pip),
                                                    ('one_hot',one_hot)])),
                        ('cleanup',cleanup_transformer.cleanup_transformer())])

    return features, features.transform(df)

#features with date, category manipulation without feature importance
def get_poly_features(df, ordinal, categorical,date_manip ,cont):
    date_pip = Pipeline([('extract',column_extractor.column_extractor(date_manip)),
                ('date_manip', date_transformer.date_transformer('%Y-%m-%d')),
                ('d-m-y-q-dow', FeatureUnion([('day',date_transformer.day_of_month_transformer()),
                                            ('month',date_transformer.month_transformer()),
                                            ('dow',date_transformer.day_of_week_transformer()),
                                            ('quarter',date_transformer.month_quarter_transformer()),
                                            ('year',date_transformer.year_transformer())])),
                ('impute',imputing_transformer.imputing_transformer(Imputer(strategy='most_frequent')))])

    continuous = Pipeline([
            ('extract', column_extractor.column_extractor(cont)),
            ('impute',imputing_transformer.imputing_transformer(Imputer(strategy='most_frequent'))),
            ('scale', Normalizer())])

    ordinal_pip = Pipeline([('extract', column_extractor.column_extractor(ordinal)),
                            ('ord', categorical_transformer.ordinal_transformer(ordinal)),
                           ('impute',imputing_transformer.imputing_transformer(Imputer(strategy='most_frequent')))])

    one_hot = Pipeline([('extract', column_extractor.column_extractor(categorical)),
                        ('lab_enc', categorical_transformer.label_transformer()),
                        ('one_hot', ModelTransformer.ModelTransformer(OneHotEncoder(sparse=False)))])


    features = Pipeline([('parallel', FeatureUnion([('date',date_pip),
                                                    ('continuous',continuous),
                                                    ('ordinal_pip',ordinal_pip),
                                                    ('one_hot',one_hot)])),
                        ('poly',ModelTransformer.ModelTransformer(PolynomialFeatures(degree=2))),
                        ('cleanup',cleanup_transformer.cleanup_transformer())])

    return features, features.transform(df)
