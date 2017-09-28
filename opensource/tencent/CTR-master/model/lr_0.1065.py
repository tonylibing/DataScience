# -*- coding: utf-8 -*-
"""
baseline 2: ad.csv (creativeID/adID/camgaignID/advertiserID/appID/appPlatform) + lr
"""
import xgboost as xgb
import scipy as sp
import zipfile
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import util
import numpy as np

# load data
data_root = "/home/zsc/PycharmProjects/Tencent_CTR/data"
dfTrain = pd.read_csv("%s/train.csv"%data_root)

dfTest = pd.read_csv("%s/test.csv"%data_root)

print dfTrain.columns.values
print dfTest.columns.values
print dfTest.shape

# import sys
# sys.exit()


#preprocess

#train
dfTrain['clickTime_day_gap'] = dfTrain['clickTime'].apply(util.get_train_time_day)
dfTrain['clickTime_hour'] = dfTrain['clickTime'].apply(util.get_time_hour)

#test

dfTest['clickTime_day_gap'] = dfTest['clickTime'].apply(util.get_test_time_day)
dfTest['clickTime_hour'] = dfTest['clickTime'].apply(util.get_time_hour)


#ad
ad = util.read_csv_file(data_root+'/ad.csv',logging=True)

#app
app_categories = util.read_csv_file(data_root+'/app_categories.csv',logging=True)
app_categories["app_categories_first_class"] = app_categories['appCategory'].apply(util.categories_process_first_class)
app_categories["app_categories_second_class"] = app_categories['appCategory'].apply(util.categories_process_second_class)



#user

user = util.read_csv_file(data_root+'/user.csv',logging=True)
user['age_process'] = user['age'].apply(util.age_process)

user["hometown_province"] = user['hometown'].apply(util.hometown_process_province)
user["hometown_city"] = user['hometown'].apply(util.hometown_process_city)
user["residence_province"] = user['residence'].apply(util.hometown_process_province)
user["residence_city"] = user['residence'].apply(util.hometown_process_city)

#position
position = util.read_csv_file(data_root+'/position.csv')



# 每个用户安装的应用的数量
user_installedapps  = pd.read_csv(data_root+'/user_installedapps.csv')
app_install_per_user = user_installedapps.groupby('userID').size().reset_index()
app_install_per_user.columns = ['userID','app_install_per_user']
# 每个用户点击的次数
click_per_user = dfTrain.groupby('userID').size().reset_index()
click_per_user.columns = ['userID','click_per_user']
# 每个应用被点击的次数:应该统计一个周期的点击次数
user_click_per_app =dfTrain.groupby('creativeID').size().reset_index()
user_click_per_app.columns=['creativeID','user_click_per_creative']
# 每个应用被安装的次数
user_install_per_app = user_installedapps.groupby('appID').size().reset_index()
user_install_per_app.columns=['appID','user_per_app']
# 是否已经安装

#统计窗口时间一个礼拜,最后一天的预测结果,预测的x应该包含前一个礼拜的数据
#训练集只包含一个礼拜的数据,其他数据应该丢弃


# process data
dfTrain = pd.merge(dfTrain, ad, how='left',on="creativeID")
dfTrain = pd.merge(dfTrain, user,how='left',on='userID')
dfTrain = pd.merge(dfTrain, app_categories, how='left',on='appID')
dfTrain = pd.merge(dfTrain, app_install_per_user,how='left',on='userID')
dfTrain = pd.merge(dfTrain, click_per_user,how='left',on='userID')
dfTrain = pd.merge(dfTrain, user_click_per_app,how='left',on='creativeID')
dfTrain = pd.merge(dfTrain, user_install_per_app,how='left',on='appID')
dfTrain = pd.merge(dfTrain, position,how='left',on='positionID')


dfTest = pd.merge(dfTest, ad, how='left',on="creativeID")
dfTest = pd.merge(dfTest, user,how='left',on='userID')
dfTest = pd.merge(dfTest, app_categories, how='left',on='appID')
print dfTest.shape
dfTest = pd.merge(dfTest, app_install_per_user,how='left',on='userID')
dfTest = pd.merge(dfTest, click_per_user,how='left',on='userID')
print dfTest.shape
dfTest = pd.merge(dfTest, user_click_per_app,how='left',on='creativeID')
dfTest = pd.merge(dfTest, user_install_per_app,how='left',on='appID')
print dfTest.shape
dfTest = pd.merge(dfTest, position,how='left',on='positionID')


dfTrain = dfTrain[dfTrain['clickTime_day_gap']<10]
dfTest = dfTest[dfTest['clickTime_day_gap']<10]
print dfTest.shape
dfTrain.sort_values("userID", inplace=True)

dfTrain = dfTrain.fillna(0)
dfTest = dfTest.fillna(0)

dfTrain.to_csv("dfTrain.csv",index=False)




y_train = dfTrain["label"].values

# import sys
# sys.exit()

# feature engineering/encoding
enc = OneHotEncoder()
# special ,'clickTime_day_gap'
feats = ["creativeID", "adID", "advertiserID",'camgaignID', "appID", "appPlatform",'gender','education','haveBaby','marriageStatus',"app_categories_first_class",'app_categories_second_class',
         'age_process','hometown_province','hometown_city','residence_province','residence_city','sitesetID','positionType','clickTime_hour']
for i,feat in enumerate(feats):
    train_data = np.array(dfTrain[feat].values,dtype='int32')
    print  i,":  ",train_data
    print train_data.any()<0
    x_train = enc.fit_transform(train_data.reshape(-1, 1))

    test_data = np.array(dfTest[feat].values,dtype='int32')
    x_test = enc.transform(test_data.reshape(-1, 1))
    print i
    if i == 0:
        X_train, X_test = x_train, x_test
    else:
        X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))


feats_2 = ['app_install_per_user','click_per_user','user_click_per_creative','user_per_app']
from sklearn.preprocessing import MinMaxScaler
max_min_scaler = MinMaxScaler()
for i,feat in enumerate(feats_2):
    train_data = np.array(dfTrain[feat].values,dtype='int32').reshape(-1, 1)
    print  i, ":  ", train_data
    x_train = max_min_scaler.fit_transform(train_data)

    test_data = np.array(dfTest[feat].values,dtype='int32').reshape(-1, 1)
    x_test = max_min_scaler.transform(test_data)
    X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))



# model training
from sklearn.ensemble import GradientBoostingClassifier

lr = LogisticRegression(tol=1e-4,max_iter=100)
lr.fit(X_train, y_train)
proba_test = lr.predict_proba(X_test)[:,1]


# submission

print dfTest['instanceID'].shape
print proba_test.shape

df = pd.DataFrame({"instanceID": dfTest["instanceID"].values, "proba": proba_test})
df.sort_values("instanceID", inplace=True)
df.to_csv("submission.csv", index=False)
with zipfile.ZipFile("submission.zip", "w") as fout:
    fout.write("submission.csv", compress_type=zipfile.ZIP_DEFLATED)
