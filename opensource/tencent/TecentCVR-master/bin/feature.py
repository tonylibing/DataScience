#!/usr/bin/env python
# encoding: utf-8
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

import path
import preprocessing as preproc

def READ(path, iflog=1):
    return preproc.read_csv_file(path, logging = iflog)

#['creativeID' 'adID' 'camgaignID' 'advertiserID' 'appID' 'appPlatform']
ad = READ(path.AD_CSV)

# app
app_categories = READ(path.APP_CATEGORIES_CSV)
app_categories["app_categories_1st_class"] = app_categories['appCategory'].apply(preproc.categories_process_first_class)
app_categories['app_categories_2nd_class'] = app_categories['appCategory'].apply(preproc.categories_process_second_class)


# user
#['userID', 'age', 'gender', 'education', 'marriageStatus','haveBaby', 'hometown', 'residence']
user = READ(path.USER_CSV, 0)
user['age_process'] = user['age'].apply(preproc.age_process)
user['hometown_province'] = user['hometown'].apply(preproc.process_province)
user['hometown_city'] = user['hometown'].apply(preproc.process_city)
user['residence_province'] = user['residence'].apply(preproc.process_province)
user['residence_city'] = user['residence'].apply(preproc.process_city)

"""
user[user.age!=0].describe()
user.age.value_count()
"""

# train_data
#['label' 'clickTime' 'conversionTime' 'creativeID' 'userID' 'positionID' 'connectionType' 'telecomsOperator']
train_data = READ(path.TRAIN_CSV)
train_data['clickTime_day'] = train_data['clickTime'].apply(preproc.get_time_day)
train_data['clickTime_hour'] = train_data['clickTime'].apply(preproc.get_time_hour)

# test_data
test_data = READ(path.TEST_CSV)
test_data['clickTime_day'] = test_data['clickTime'].apply(preproc.get_time_day)
test_data['clickTime_hour'] = test_data['clickTime'].apply(preproc.get_time_hour)

# merge data
## train_user_ad_app
train_user = pd.merge(train_data, user, on='userID')
train_user_ad = pd.merge(train_user, ad, on='creativeID')
train_user_ad_app = pd.merge(train_user_ad, app_categories, on='appID')


# 取出数据和label
x_user_ad_app = train_user_ad_app.loc[:, ['creativeID','userID','positionID',
                                          'connectionType','telecomsOperator','clickTime_day','clickTime_hour','age', 'gender' ,'education',
                                          'marriageStatus' ,'haveBaby' , 'residence' ,'age_process',
                                          'hometown_province', 'hometown_city','residence_province', 'residence_city',
                                          'adID', 'camgaignID', 'advertiserID', 'appID' ,'appPlatform' ,
                                          'app_categories_first_class' ,'app_categories_second_class']
                                      ]

## x
x_user_ad_app = x_user_ad_app.values
x_user_ad_app = np.array(x_user_ad_app, dtype='int32')

## y
y_user_ad_app = train_user_ad_app.loc[:, ['label']].values

feat_labels = np.array(['creativeID','userID','positionID',
                        'connectionType','telecomsOperator','clickTime_day','clickTime_hour','age', 'gender' ,'education',
                        'marriageStatus' ,'haveBaby' , 'residence' ,'age_process',
                        'hometown_province', 'hometown_city','residence_province', 'residence_city',
                        'adID', 'camgaignID', 'advertiserID', 'appID' ,'appPlatform' ,
                        'app_categories_first_class' ,'app_categories_second_class'])
