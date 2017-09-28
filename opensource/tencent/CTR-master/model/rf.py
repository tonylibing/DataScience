#coding=utf-8

from sklearn.linear_model import  SGDClassifier
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression


import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys


from data_preprocess import util


train_data = util.read_csv_file('../data/train.csv',logging=True)
#['label' 'clickTime' 'conversionTime' 'creativeID' 'userID' 'positionID'
 # 'connectionType' 'telecomsOperator']
ad = util.read_csv_file('../data/ad.csv',logging=True)
#['creativeID' 'adID' 'camgaignID' 'advertiserID' 'appID' 'appPlatform']



rf = RandomForestClassifier(n_estimators=10)
#class_weight : dict, list of dicts, "balanced",
# "balanced_subsample" or None, optional (default=None)
# Weights associated with classes in the form ``{class_label: weight}``.

from data_preprocess import  util




#ad
ad = util.read_csv_file('../data/ad.csv',logging=True)

#app
app_categories = util.read_csv_file('../data/app_categories.csv',logging=True)
app_categories["app_categories_first_class"] = app_categories['appCategory'].apply(util.categories_process_first_class)
app_categories["app_categories_second_class"] = app_categories['appCategory'].apply(util.categories_process_second_class)



#user

user = util.read_csv_file('../data/user.csv',logging=True)
user['age_process'] = user['age'].apply(util.age_process)

user["hometown_province"] = user['hometown'].apply(util.hometown_process_province)
user["hometown_city"] = user['hometown'].apply(util.hometown_process_city)
user["residence_province"] = user['residence'].apply(util.hometown_process_province)
user["residence_city"] = user['residence'].apply(util.hometown_process_city)





#train data
train_data['clickTime_day'] = train_data['clickTime'].apply(util.get_time_day)
train_data['clickTime_hour']= train_data['clickTime'].apply(util.get_time_hour)


#
# train_data['conversionTime_day'] = train_data['conversionTime'].apply(util.get_time_day)
# train_data['conversionTime_hour'] = train_data['conversionTime'].apply(util.get_time_hour)


#test_data
test_data = util.read_csv_file('../data/test.csv')
test_data['clickTime_day'] = test_data['clickTime'].apply(util.get_time_day)
test_data['clickTime_hour']= test_data['clickTime'].apply(util.get_time_hour)

# test_data['conversionTime_day'] = test_data['conversionTime'].apply(util.get_time_day)
# test_data['conversionTime_hour'] = test_data['conversionTime'].apply(util.get_time_hour)




train_user = pd.merge(train_data,user,on='userID')
train_user_ad = pd.merge(train_user,ad,on='creativeID')
train_user_ad_app = pd.merge(train_user_ad,app_categories,on='appID')


# train_user_ad_app.to_csv("train_user.csv")



train_user_ad_app_positive = train_user_ad_app[train_user_ad_app['label']==1]
train_user_ad_app_negetive = train_user_ad_app[train_user_ad_app['label']==0]


x_user_ad_app_positive = train_user_ad_app_positive.loc[:,['creativeID','userID','positionID',
 'connectionType','telecomsOperator','clickTime_day','clickTime_hour','age', 'gender' ,'education',
 'marriageStatus' ,'haveBaby' , 'residence' ,'age_process',
 'hometown_province', 'hometown_city','residence_province', 'residence_city',
 'adID', 'camgaignID', 'advertiserID', 'appID' ,'appPlatform' ,
 'app_categories_first_class' ,'app_categories_second_class']]


x_user_ad_app_positive = x_user_ad_app_positive.values


x_user_ad_app_negetive = train_user_ad_app_negetive.loc[:,['creativeID','userID','positionID',
 'connectionType','telecomsOperator','clickTime_day','clickTime_hour','age', 'gender' ,'education',
 'marriageStatus' ,'haveBaby' , 'residence' ,'age_process',
 'hometown_province', 'hometown_city','residence_province', 'residence_city',
 'adID', 'camgaignID', 'advertiserID', 'appID' ,'appPlatform' ,
 'app_categories_first_class' ,'app_categories_second_class']]

# print x_user_ad_app_negetive.head(10)
#
# sys.exit()

# x_user_ad_app_negetive.to_csv("x_user_ad_app_negetive.csv")
# sys.exit()
x_user_ad_app_negetive = x_user_ad_app_negetive.values


x_user_ad_app_positive = np.array(x_user_ad_app_positive,dtype='int32')
x_user_ad_app_negetive = np.array(x_user_ad_app_negetive,dtype='int32')

print np.max(x_user_ad_app_positive)
y_user_ad_app_positive =train_user_ad_app_positive.loc[:,['label']].values
y_user_ad_app_negetive = train_user_ad_app_negetive.loc[:,['label']].values





#split train data and test data

x_train_user_ad_app_positive = x_user_ad_app_positive[:-500]
x_train_user_ad_app_negetive = x_user_ad_app_negetive[:-500]
x_test_user_ad_app_positive = x_user_ad_app_positive[-500:]
x_test_user_ad_app_negetive = x_user_ad_app_negetive[-500:]

y_train_user_ad_app_positive = y_user_ad_app_positive[:-500]
y_train_user_ad_app_negetive = y_user_ad_app_negetive[:-500]
y_test_user_ad_app_positive = y_user_ad_app_positive[-500:]
y_test_user_ad_app_negetive= y_user_ad_app_negetive[-500:]


num_positive = x_train_user_ad_app_positive.shape[0] #训练集中positive类的数量
num_negetive = x_train_user_ad_app_negetive.shape[0]#训练集中negetive类的数量




print "=========sample unbalance=================="
#93262
# 3749528


#建立40个分类器,每个分类器训练样本中的一部分样本

scale = num_negetive//num_positive

clfs = []
for i in range(scale):
    rf_i = RandomForestClassifier(n_estimators=10)
    clfs.append(rf_i)


#训练
for i in range(scale):
    print i
    clf = clfs[i]
    x_batch_1 = x_train_user_ad_app_negetive[i*num_positive : (i+1)*num_positive]
    x_batch_2 = x_train_user_ad_app_positive[:num_positive]

    y_batch_1 = y_train_user_ad_app_negetive[i*num_positive : (i+1)*num_positive]
    y_batch_2 = y_train_user_ad_app_positive[:num_positive]

    x_batch = np.vstack([x_batch_1,x_batch_2])
    y_batch = np.vstack([y_batch_1,y_batch_2])

    #shuffle
    index = [j for j in range(x_batch.shape[0])]
    np.random.shuffle(index)

    x_batch = x_batch[index]
    y_batch = y_batch[index]

    clfs[i].fit(x_batch,y_batch)

#测试
x_test = np.vstack([x_test_user_ad_app_negetive,x_test_user_ad_app_positive])
y_test = np.vstack([y_test_user_ad_app_negetive,y_test_user_ad_app_positive])

result_predict_prob = []
result_predict=[]
for i in range(scale):
    result_indiv = clfs[i].predict(x_test)
    result_indiv_proba = clfs[i].predict_proba(x_test)[:,1]
    result_predict.append(result_indiv)
    result_predict_prob.append(result_indiv_proba)

def max_count(pred):
    sum = np.sum(pred,axis=1)
    return sum>pred.shape[1]/2


result_predict_prob = np.reshape(result_predict_prob,[-1,scale])
result_predict = np.reshape(result_predict,[-1,scale])

result_predict_prob = np.mean(result_predict_prob,axis=1)
result_predict = max_count(result_predict)


y_test = np.array(y_test).reshape([-1,1])
result_predict_prob = np.array(result_predict_prob).reshape([-1,1])
logloss = util.logloss(y_test,result_predict_prob)

print "logloss:",logloss

from sklearn.metrics import  confusion_matrix
from sklearn.metrics import classification_report
print confusion_matrix(y_test,result_predict)

print classification_report(y_test,result_predict)



##############################################teset###########################

test_data = pd.merge(test_data,user,on='userID')
test_user_ad = pd.merge(test_data,ad,on='creativeID')
test_user_ad_app = pd.merge(test_user_ad,app_categories,on='appID')


# train_user_ad_app.to_csv("train_user.csv")



x_test_clean = test_user_ad_app.loc[:,['creativeID','userID','positionID',
 'connectionType','telecomsOperator','clickTime_day','clickTime_hour','age', 'gender' ,'education',
 'marriageStatus' ,'haveBaby' , 'residence' ,'age_process',
 'hometown_province', 'hometown_city','residence_province', 'residence_city',
 'adID', 'camgaignID', 'advertiserID', 'appID' ,'appPlatform' ,
 'app_categories_first_class' ,'app_categories_second_class']].values


# print x_test_clean.columns.values

x_test_clean = np.array(x_test_clean,dtype='int32')

result_predict_prob = []
result_predict=[]
for i in range(scale):
    result_indiv = clfs[i].predict(x_test_clean)
    result_indiv_proba = clfs[i].predict_proba(x_test_clean)[:,1]
    result_predict.append(result_indiv)
    result_predict_prob.append(result_indiv_proba)


result_predict_prob = np.reshape(result_predict_prob,[-1,scale])
result_predict = np.reshape(result_predict,[-1,scale])

result_predict_prob = np.mean(result_predict_prob,axis=1)
result_predict = max_count(result_predict)


result_predict_prob = np.array(result_predict_prob).reshape([-1,1])


test_data['prob'] = result_predict_prob
test_data = test_data.loc[:,['instanceID','prob']]
test_data.to_csv('predict.csv',index=False)
print "over"




