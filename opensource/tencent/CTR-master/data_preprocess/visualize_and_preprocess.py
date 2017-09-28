#coding=utf-8
import pandas as pd
import sys


from util import read_csv_file,get_time_day,get_time_hour,categories_process_first_class,categories_process_second_class,\
    age_process,hometown_process_city,hometown_process_province



ad = read_csv_file("../data/ad.csv")
app_categories = read_csv_file("../data/app_categories.csv").head(100)
position = read_csv_file("../data/position.csv").head(100)
test = read_csv_file("../data/test.csv").head(100)
train = read_csv_file("../data/train.csv")

# test.to_csv("result.csv")

# sys.exit()

user = read_csv_file("../data/user.csv").head(100)
user_app_actions = read_csv_file("../data/user_app_actions.csv").head(100)
user_installedapps = read_csv_file("../data/user_installedapps.csv").head(100)



'''
 ad.csv preprocess
 ['creativeID' 'adID' 'camgaignID' 'advertiserID' 'appID' 'appPlatform']

 ['creativeID' 'adID' 'camgaignID' 'advertiserID' 'appID' 'appPlatform_1'
 'appPlatform_2']
'''
ad_columns = ad.columns.values
print ad_columns

creativeID_process = pd.get_dummies(ad['creativeID'],prefix='creativeID')
adID_process = pd.get_dummies(ad['adID'],prefix='adID')
camgaignID_process = pd.get_dummies(ad['camgaignID'],prefix='camgaignID')
advertiserID_process = pd.get_dummies(ad['advertiserID'],prefix='advertiserID')
appPlatform_binary = pd.get_dummies(ad['appPlatform'],prefix='appPlatform')
ad_process = pd.concat([ad,creativeID_process,adID_process,camgaignID_process,advertiserID_process,appPlatform_binary],axis=1)
# ad_process.drop(['appPlatform'],axis=1,inplace=True)
print ad_process.columns.values
print appPlatform_binary.head(5)

ad_process_temp =ad_process.head(100)
del ad
del creativeID_process
del adID_process
del camgaignID_process
del advertiserID_process
del appPlatform_binary


'''
    app_categories.csv  preprocess
    ['appID' 'appCategory']

    ['appID' 'app_categories_first_class_0.0' 'app_categories_first_class_1.0'
 'app_categories_first_class_2.0' 'app_categories_first_class_3.0'
 'app_categories_first_class_4.0' 'app_categories_first_class_5.0'
 'app_categories_second_class_0' 'app_categories_second_class_1'
 'app_categories_second_class_2' 'app_categories_second_class_3'
 'app_categories_second_class_4' 'app_categories_second_class_5'
 'app_categories_second_class_6' 'app_categories_second_class_7'
 'app_categories_second_class_8' 'app_categories_second_class_9'
 'app_categories_second_class_10' 'app_categories_second_class_11']


'''


app_categories_columns = app_categories.columns.values
print app_categories_columns


# appID_process = pd.get_dummies(app_categories['appID'],prefix='appID')
app_categories_first_class = app_categories['appCategory'].apply(categories_process_first_class)
app_categories_first_class_dummy = pd.get_dummies(app_categories_first_class,prefix='app_categories_first_class')
app_categories_second_class = app_categories['appCategory'].apply(categories_process_second_class)
app_categories_second_class_dummy = pd.get_dummies(app_categories_second_class,prefix='app_categories_second_class')


app_categories_process = pd.concat([app_categories,app_categories_first_class_dummy,app_categories_second_class_dummy],axis=1)
# app_categories_process.drop(['appCategory'],axis=1,inplace=True)
print app_categories_process.columns.values

app_categories_process_temp = app_categories_process.head(100)
del app_categories_process
del app_categories_first_class_dummy
del app_categories_second_class_dummy
del app_categories

'''
    position.csv process
    ['positionID' 'sitesetID' 'positionType']

    ['positionID' 'siteset_id_0' 'siteset_id_1' 'siteset_id_2'
 'position_type_0' 'position_type_1' 'position_type_2' 'position_type_3'
 'position_type_4' 'position_type_5']
'''



position_columns = position.columns.values
#['positionID'广告曝光的具体位置  'sitesetID'多个广告位的聚合，如QQ空间 'positionType'对于某些站点，人工定义的一套广告位规格分类，如Banner广告位。]
print position_columns

position_process = pd.get_dummies(position['positionID'],prefix='positionID')

siteset_id = pd.get_dummies(position['sitesetID'],prefix='sitesetId')

position_type = pd.get_dummies(position['positionType'],prefix='positionType')

position_process = pd.concat([position,position_process,siteset_id,position_type],axis=1)
# position_process.drop(['sitesetID','positionType'],axis=1,inplace=True)
print position_process.columns

position_process_temp = position_process.head(100)
del position_process
del siteset_id
del position_type
del position



'''
    user.csv process
    ['userID' 'age' 'gender' 'education' 'marriageStatus' 'haveBaby' 'hometown' 'residence']

    ['userID' 'age_0' 'age_1' 'age_2' 'age_3' 'age_4' 'age_5' 'gender_0'
 'gender_1' 'gender_2' 'education_0' 'education_1' 'education_2'

 'marriageStatus_0' 'marriageStatus_1' 'marriageStatus_2'
 'marriageStatus_3' 'haveBaby_0' 'haveBaby_1' 'haveBaby_2' 'haveBaby_3'
 'haveBaby_4' 'haveBaby_5' 'haveBaby_6' 'hometown_province_0'
 'hometown_province_10' 'hometown_province_11' 'hometown_province_12'
 'hometown_province_90' 'hometown_province_91' 'hometown_city_'
 'hometown_city_0' 'hometown_city_01' 'hometown_city_02' 'hometown_city_03'

 'hometown_city_9' 'resident_province_0' 'resident_province_10'
 'resident_province_11' 'resident_province_12' 'resident_province_13'

 'resident_province_91' 'resident_city_' 'resident_city_0'
 'resident_city_00' 'resident_city_01' 'resident_city_02'
 '
 'resident_city_5' 'resident_city_6' 'resident_city_7' 'resident_city_8'
 'resident_city_9']
'''


user_columns = user.columns.values
#
print user_columns

userID_process = pd.get_dummies(user['userID'],prefix='userID')
gender_process = pd.get_dummies(user['gender'],prefix='gender')
education_process = pd.get_dummies(user['education'],prefix='education')
marriageStatus_process = pd.get_dummies(user['marriageStatus'],prefix='marriageStatus')
haveBaby_process = pd.get_dummies(user['haveBaby'],prefix='haveBaby')

from sklearn.preprocessing import MinMaxScaler





user['age_process'] = user['age'].apply(age_process)
age_process = pd.get_dummies(user['age_process'],prefix='age')

hometown_province = user['hometown'].apply(hometown_process_province)
hometown_province_process = pd.get_dummies(hometown_province,prefix='hometown_province')

hometown_city = user['hometown'].apply(hometown_process_city)
hometown_city_process = pd.get_dummies(hometown_city,prefix='hometown_city')

residence_province = user['residence'].apply(hometown_process_province)
residence_province_process = pd.get_dummies(residence_province,prefix='resident_province')

residence_city = user['residence'].apply(hometown_process_city)
residence_city_process = pd.get_dummies(residence_city,prefix='resident_city')

user_process = pd.concat([user,userID_process,age_process,gender_process,education_process,marriageStatus_process,
                  haveBaby_process,hometown_province_process,hometown_city_process,
                  residence_province_process,residence_city_process],axis=1)

# user_process.drop([])

print user_process.columns.values

user_process_temp = user_process.head(100)
del user_process
del age_process
del hometown_city_process
del hometown_province_process
del hometown_city
del hometown_province
del residence_city_process
del residence_city
del residence_province
del residence_province_process
del user



'''
    user_app_actions.csv process
    ['userID' 'installTime' 'appID']
'''



user_app_actions_columns = user_app_actions.columns.values
print user_app_actions_columns

# print user_app_actions.appID.value_counts()

import collections

counter = collections.Counter(user_app_actions['appID'])


sorted_items =  sorted(counter.items(),lambda x,y:cmp(x[1],y[1]),reverse=True)

# print type(sorted_items)
j=0
for k,i in sorted_items:
    if j==100:
        break
    j+=1
    # print k,"  : ",i




'''
    user_installedapps.csv process
'''

user_installedapps_columns = user_installedapps.columns.values

print user_installedapps_columns

user_installedapps_group_by_user = user_installedapps.groupby('userID')
# print user_installedapps_group_by_user.columns.values

'''
    train.csv
    ['label' 'clickTime' 'conversionTime' 'creativeID' 'userID' 'positionID' 'connectionType' 'telecomsOperator']
'''


train_columns = train.columns.values
#['label' 'clickTime' 'conversionTime' 'creativeID' 'userID' 'positionID'
# 'connectionType' 'telecomsOperator']

#creativeID 展示给用户直接看到的广告内容，一条广告下可以有多组素材。
#posicitonID 广告位ID

print train_columns





train['clickTime_day'] = train['clickTime'].apply(get_time_day)
train['clickTime_hour'] = train['clickTime'].apply(get_time_hour)

clickTime_day_process = pd.get_dummies(train['clickTime_day'],prefix='clickTime_day')
clickTime_hour_process = pd.get_dummies(train['clickTime_hour'],prefix='clickTime_hour')

connectType_process = pd.get_dummies(train['connectionType'],prefix='connectionType')
telecomsOperator_process = pd.get_dummies(train['telecomsOperator'],prefix='telecomsOperator')

train_process = pd.concat([train,clickTime_day_process,clickTime_hour_process,connectType_process,telecomsOperator_process],axis=1)

print train_process.columns.values
train_process_temp = train_process.head(100)
del train_process
del clickTime_day_process
del clickTime_hour_process
del connectType_process
del train


#数据join

# train_data_train_user = pd.merge(train_process,user_process,how='inner',left_on='userID',right_on='userID')
# train_data_train_user_position = pd.merge(train_data_train_user,position_process,how='inner',on='positionID')
# train_data_train_user_position_ad = pd.merge(train_data_train_user_position,ad_process,how='inner',on='creativeID')
# print train_data_train_user_position_ad.columns.value



train_data_train_user_temp = pd.merge(train_process_temp,user_process_temp,how='inner',left_on='userID',right_on='userID')
train_data_train_user_position_temp = pd.merge(train_data_train_user_temp,position_process_temp,how='inner',on='positionID')
print train_data_train_user_position_temp.columns.values
print "===================="
print ad_process_temp.columns.values
train_user_position_ad_temp = pd.merge(train_data_train_user_position_temp,ad_process_temp,how='inner',on='creativeID')
print train_user_position_ad_temp.columns.values

window  = 10

'''
    获得时间窗口内的数据
    最后一天计算用户成功转化
    前面几天用户发生点击行为,但未发生转化
   train:  label' 'clickTime' 'conversionTime' 'creativeID' 'userID' 'positionID' 'connectionType' 'telecomsOperator']

'''

period = train_user_position_ad_temp
print period.head(100)

period.to_csv("result.csv")

