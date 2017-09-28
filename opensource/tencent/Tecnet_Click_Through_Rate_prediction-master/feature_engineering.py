import pandas as pd
import numpy as np
import scipy as sp
from pandas import DataFrame

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
ad = pd.read_csv('ad.csv')
app = pd.read_csv('app_categories.csv')
user = pd.read_csv('user.csv')

### train test
train.head()
test.head()

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

### ad
ad.head()

## app
app.head()

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

user.head()
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

### merge data

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
 'marriageStatus' ,'haveBaby' , 'residence' ,'age',
 'hometown_province', 'hometown_city','residence_province', 'residence_city',
 'adID', 'camgaignID', 'advertiserID', 'appID' ,'appPlatform' ,
 'app_categories_first_class' ,'app_categories_second_class']

train_x = train_user_ad_app[feature_columns]
test_x = test_user_ad_app[feature_columns]

train_y = DataFrame(train_user_ad_app['label'])

# save as csv
train_x.to_csv('train_x.csv',index=False)
train_y.to_csv('train_y.csv',index=False)
test_x.to_csv('test_x.csv',index=False)





