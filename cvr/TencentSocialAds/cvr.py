from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import processor
reload(processor)
from processor import *


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
ad = pd.read_csv('ad.csv')
app = pd.read_csv('app_categories.csv')
user = pd.read_csv('user.csv')

def get_time_day(t): #click time is DDHHMM
    t = str(t)
    t=int(t[0:2])
    return t

def get_time_hour(t):
    t = str(t)
    t=int(t[2:4])
    if t<6:
        return 0
    elif t<12:
        return 1
    elif t<18:
        return 2
    else:
        return 3

train['clickTime_day'] = train['clickTime'].apply(get_time_day)
train['clickTime_hour']= train['clickTime'].apply(get_time_hour)

test['clickTime_day'] = test['clickTime'].apply(get_time_day)
test['clickTime_hour']= test['clickTime'].apply(get_time_hour)

# method 1
def categories_first_class(cate):
    cate = str(cate)
    if len(cate)==1:
        if int(cate)==0:
            return 0
    else:
        return int(cate[0])

# method 2
def categories_second_class(cate):
    cate = str(cate)
    if len(cate)<3:
        return 0
    else:
        return int(cate[1:])

app["app_categories_first_class"] = app['appCategory'].apply(categories_first_class)
app["app_categories_second_class"] = app['appCategory'].apply(categories_second_class)

### user
user[user['age']!=0].describe() #age 0 means unknown

# deal with age
def age_process(age):
    age = int(age)
    if age==0:
        return 0
    elif age<15:
        return 1
    elif age<21:
        return 2
    elif age<28:
        return 3
    else:
        return 4

# deal with provice
def process_province(hometown):
    hometown = str(hometown)
    province = int(hometown[0:2])
    return province

#城市处理
def process_city(hometown):
    hometown = str(hometown)
    if len(hometown)>1:
        province = int(hometown[2:])
    else:
        province = 0
    return province

user['age_process'] = user['age'].apply(age_process)
user["hometown_province"] = user['hometown'].apply(process_province)
user["hometown_city"] = user['hometown'].apply(process_city)
user["residence_province"] = user['residence'].apply(process_province)
user["residence_city"] = user['residence'].apply(process_city)


# merge data
train_user = pd.merge(train,user,on='userID')
train_user_ad = pd.merge(train_user,ad,on='creativeID')
train_user_ad_app = pd.merge(train_user_ad,app,on='appID')
train_user_ad_app.head()

test_user = pd.merge(test,user,on='userID')
test_user_ad = pd.merge(test_user,ad,on='creativeID')
test_user_ad_app = pd.merge(test_user_ad,app,on='appID')
test_user_ad_app.head()


feature_columns = ['creativeID','userID','positionID',
 'connectionType','telecomsOperator','clickTime_day','clickTime_hour','age', 'gender' ,'education',
 'marriageStatus' ,'haveBaby' , 'residence' ,
 'hometown_province', 'hometown_city','residence_province', 'residence_city',
 'adID', 'camgaignID', 'advertiserID', 'appID' ,'appPlatform' ,
 'app_categories_first_class' ,'app_categories_second_class']

train_x = train_user_ad_app[feature_columns]
test_x = test_user_ad_app[feature_columns]

train_y = pd.DataFrame(train_user_ad_app['label'])

# save as csv
train_x.to_csv('train_x.csv',index=False)
train_y.to_csv('train_y.csv',index=False)
test_x.to_csv('test_x.csv',index=False)


#feature preprocessing
#for linear models:ID level features(one-hot,feature hashing),continuous features(binning),categorical features(ordinal,one-hot)
#for non-linear models:ID level features(one-hot),raw continuous features,categorical features
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
