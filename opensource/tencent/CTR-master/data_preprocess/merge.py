
#coding=utf-8
import scipy as sp
import zipfile
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
import util
import numpy as np
np.random.seed(1024)
from scipy import sparse
import sys

root = "/home/zsc/PycharmProjects/Tencent_CTR/data/"
data_dir = './process_data/'

train_df = pd.read_csv(root+"train.csv")
test_df=  pd.read_csv(root+'test.csv')

train_df['source']='train'
test_df['source']='submit'

data = pd.concat([train_df,test_df],axis=0)

# print data.head()

'''
#训练集和测试集的数据分别从23 - 28(预测28号的label), 24-29(预测29号的label)
提交的数据source为submit
'''
dfTrain = data[data.clickTime.between(230000,282359)]
dfTrain_not_currentday = dfTrain[dfTrain.clickTime<280000]
dfTest = data[data.clickTime.between(240000,292359)]
dfTest_not_currentday = dfTest[dfTest.clickTime<290000]


########################################################
'''
    train.csv
'''
'''
click
'''
###################每个用户的点击次数统计
click_per_user_train = dfTrain.groupby('userID').size().reset_index()
click_per_user_train.columns = ['userID','click_per_user']

click_per_user_test = dfTest.groupby('userID').size().reset_index()
click_per_user_test.columns = ['userID','click_per_user']


################### 每个creative的点击次数统计
click_per_creative_train = dfTrain.groupby('creativeID').size().reset_index()
click_per_creative_train.columns = ['creativeID','click_per_creative']


click_per_creative_test = dfTest.groupby('creativeID').size().reset_index()
click_per_creative_test.columns = ['creativeID','click_per_creative']


'''
conversion
'''
################### 每个用户的转化次数统计
conversion_train_not_currentday = dfTrain_not_currentday[dfTrain_not_currentday.label==1]
conversion_per_user_train = conversion_train_not_currentday.groupby('userID')['label'].sum().reset_index()
conversion_per_user_train.columns = ['userID','conversion_per_user']

conversion_test_not_currentday = dfTest_not_currentday[dfTest_not_currentday.label==1]
conversion_per_user_test = conversion_test_not_currentday.groupby('userID')['label'].sum().reset_index()
conversion_per_user_test.columns = ['userID','conversion_per_user']



################### 每个creative的转化次数统计
conversion_per_creative_train = conversion_train_not_currentday.groupby('creativeID').size().reset_index()
conversion_per_creative_train.columns = ['creativeID','conversion_per_creative']

conversion_per_creative_test = conversion_test_not_currentday.groupby('creativeID').size().reset_index()
conversion_per_creative_test.columns = ['creativeID','conversion_per_creative']

'''
conversion rate
'''
###################每个用户的转化率统计
conversion_by_click_per_user_train = (dfTrain_not_currentday.groupby(['userID'])['label'].sum()/dfTrain_not_currentday.groupby(['userID'])['creativeID'].count()).reset_index()
conversion_by_click_per_user_train.columns=['userID','conversion_by_click_per_user']

conversion_by_click_per_user_test= (dfTest_not_currentday.groupby(['userID'])['label'].sum()/dfTest_not_currentday.groupby(['userID'])['creativeID'].count()).reset_index()
conversion_by_click_per_user_test.columns=['userID','conversion_by_click_per_user']


#################每个creative的转化率
conversion_by_click_per_creative_train = (dfTrain_not_currentday.groupby(['creativeID'])['label'].sum()/dfTrain_not_currentday.groupby(['creativeID'])['creativeID'].count()).reset_index()
conversion_by_click_per_creative_train.columns=['creativeID','conversion_by_click_per_creative']

conversion_by_click_per_creative_test = (dfTest_not_currentday.groupby(['creativeID'])['label'].sum()/dfTest_not_currentday.groupby(['creativeID'])['creativeID'].count()).reset_index()
conversion_by_click_per_creative_test.columns=['creativeID','conversion_by_click_per_creative']




'''
    currentday 点击的creative 之前是否转化
'''
#currentday 点击的creative 之前是否安装过

user_creative_pre_label_train= dfTrain_not_currentday.groupby(['userID','creativeID'])['label'].sum().reset_index()
user_creative_pre_label_train.columns= ['userID','creativeID','pre_label']
# user_creative_pre_label_train.sort_values('pre_label',inplace=True,ascending=False)


user_creative_pre_label_test= dfTest_not_currentday.groupby(['userID','creativeID'])['label'].sum().reset_index()
user_creative_pre_label_test.columns= ['userID','creativeID','pre_label']



###########################################################
'''
user.csv
['userID' 'age' 'gender' 'education' 'marriageStatus' 'haveBaby' 'hometown'
 'residence' 'age_process' 'hometown_province' 'hometown_city'
 'residence_province' 'residence_city']
'''
user = pd.read_csv(root+"user.csv")
user['age_process'] = user['age'].apply(util.age_process)
user['hometown_province'] = user['hometown'].apply(util.hometown_process_province)
user['hometown_city'] = user['hometown'].apply(util.hometown_process_city)
user['residence_province'] = user['residence'].apply(util.hometown_process_province)
user['residence_city'] = user['residence'].apply(util.hometown_process_city)
###########################################################
'''
ad.csv
'''
ad = pd.read_csv(root+'ad.csv')


############################################################
'''
 app_categories_process.csv
 ['appID' 'appCategory' 'app_category_first_class'
 'app_category_second_class']

'''
app_category = pd.read_csv(root+'app_categories.csv')
app_category['app_category_first_class'] = app_category['appCategory'].apply(util.categories_process_first_class)
app_category['app_category_second_class'] = app_category['appCategory'].apply(util.categories_process_second_class)

###########################################################
'''
user_app_action_process.csv

'''
user_app_actions = pd.read_csv(root+'user_app_actions.csv')
#每个用户安装的次数
app_install_per_user = user_app_actions.groupby('userID').size().reset_index()
app_install_per_user.columns = ['userID','app_install_per_user']

print app_install_per_user.head()

#每个app安装的次数
app_install_per_app = user_app_actions.groupby('appID').size().reset_index()
app_install_per_app.columns = ['appID','app_install_per_app']
print app_install_per_app.head()

'''
position
'''
position = pd.read_csv(root+'position.csv')

##########################################################################################################
'''
dfTrain = data[data.clickTime.between(230000,282359)]
dfTrain_not_currentday = dfTrain[dfTrain.clickTime<280000]
dfTest = data[data.clickTime.between(240000,292359)]
dfTest_not_currentday = dfTest[dfTest.clickTime<290000]
'''

train_merge = pd.merge(dfTrain,user,how='left',on='userID')
train_merge = pd.merge(train_merge,ad,how='left',on='creativeID')
train_merge = pd.merge(train_merge,app_category,how='left',on='appID')
train_merge = pd.merge(train_merge,position,how='left',on='positionID')

train_merge = pd.merge(train_merge,app_install_per_user,how='left',on='userID')
train_merge = pd.merge(train_merge,app_install_per_app,how='left',on='appID')

train_merge = pd.merge(train_merge,click_per_user_train,how='left',on='userID')
train_merge = pd.merge(train_merge,click_per_creative_train,how='left',on='creativeID')
train_merge = pd.merge(train_merge,conversion_per_user_train,how='left',on='userID')
train_merge = pd.merge(train_merge,conversion_per_creative_train,how='left',on='creativeID')
train_merge = pd.merge(train_merge,conversion_by_click_per_user_train,how='left',on='userID')
train_merge = pd.merge(train_merge,conversion_by_click_per_creative_train,how='left',on='creativeID')
train_merge = pd.merge(train_merge,user_creative_pre_label_train,how='left',on=['userID','creativeID'])




###########################################################################################################

test_merge = pd.merge(dfTest,user,how='left',on='userID')
test_merge = pd.merge(test_merge,ad,how='left',on='creativeID')
test_merge = pd.merge(test_merge,app_category,how='left',on='appID')
test_merge = pd.merge(test_merge,position,how='left',on='positionID')

test_merge = pd.merge(test_merge,app_install_per_user,how='left',on='userID')
test_merge = pd.merge(test_merge,app_install_per_app,how='left',on='appID')

test_merge = pd.merge(test_merge,click_per_user_test,how='left',on='userID')
test_merge = pd.merge(test_merge,click_per_creative_test,how='left',on='creativeID')
test_merge = pd.merge(test_merge,conversion_per_user_test,how='left',on='userID')
test_merge = pd.merge(test_merge,conversion_per_creative_test,how='left',on='creativeID')
test_merge = pd.merge(test_merge,conversion_by_click_per_user_test,how='left',on='userID')
test_merge = pd.merge(test_merge,conversion_by_click_per_creative_test,how='left',on='creativeID')
test_merge = pd.merge(test_merge,user_creative_pre_label_test,how='left',on=['userID','creativeID'])


'''

'''
train_merge = train_merge.fillna({'app_install_per_user':0,'app_install_per_app':0,'click_per_user':0,'click_per_creative':0,
                                  'conversion_per_user':0,'conversion_per_creative':0,'conversion_by_click_per_user':0,
                                  'conversion_by_click_per_creative':0,'pre_label':0})

test_merge = test_merge.fillna({'app_install_per_user':0,'app_install_per_app':0,'click_per_user':0,'click_per_creative':0,
                                  'conversion_per_user':0,'conversion_per_creative':0,'conversion_by_click_per_user':0,
                                  'conversion_by_click_per_creative':0,'pre_label':0})





############################################################################################################
#label
merge = pd.concat([train_merge,test_merge],axis=0)

dfTrain_currentday = train_merge[train_merge['clickTime'].between(280000,282359)]
dfTest_currentday = test_merge[test_merge['clickTime'].between(290000,292359)]
y_test = dfTest_currentday['label']

dfTrain_negetive = dfTrain_currentday[dfTrain_currentday.label!=1]
dfTrain_positive = dfTrain_currentday[dfTrain_currentday.label==1]

dfTrain_negetive  = dfTrain_negetive.iloc[:100000,:]

dfTrain_balance = pd.concat([dfTrain_negetive,dfTrain_positive],axis=0)

y_train = dfTrain_balance['label']

index = [i for i in range(dfTrain_balance.shape[0])]
np.random.shuffle(index)
# X_train = X_train.toarray()
dfTrain_balance = dfTrain_balance.iloc[index,:]
y_train = y_train[index]

# #
# dfTrain_balance.to_csv("dftrain_balance.csv")
# test_merge.to_csv('dftest.csv')
# sys.exit()
# feature engineering/encoding
enc = OneHotEncoder()
# special ,'clickTime_day_gap'
#

feats = ['creativeID','positionID','telecomsOperator','userID','gender','education','marriageStatus','haveBaby',
         'age_process','hometown_province','hometown_city','residence_province','residence_city','adID',
         'camgaignID','advertiserID','appID','appPlatform','app_category_first_class','app_category_second_class',
         'sitesetID','positionType']
# for i,feat in enumerate(feats):
#     #某些稀疏的feature 需要在全局进行 one hot编码,否则在 test集中会出现feature未出现在训练集中的情况,发生错误
#     global_train_data = np.array(merge[feat].values,dtype='int32')
#     print  i,global_train_data,global_train_data
#     print np.any(global_train_data==np.nan)
#     enc = enc.fit(global_train_data.reshape(-1, 1))
#     #x-train
#     train_data = np.array(dfTrain_balance[feat].values,dtype='int32')
#     x_train = enc.transform(train_data.reshape(-1, 1))
#     #x_test
#     test_data = np.array(dfTest_currentday[feat].values,dtype='int32')
#     x_test = enc.transform(test_data.reshape(-1, 1))
#     print i
#     if i == 0:
#         X_train, X_test = x_train, x_test
#     else:
#         X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))

feats_2 = ['app_install_per_user','app_install_per_app','click_per_user','click_per_creative','conversion_per_user',
           'conversion_per_creative','conversion_by_click_per_user','conversion_by_click_per_creative','pre_label']

from sklearn.preprocessing import MinMaxScaler
max_min_scaler = MinMaxScaler()
for i,feat in enumerate(feats_2):
    train_data = np.array(dfTrain_balance[feat].values,dtype='int32').reshape(-1, 1)
    print  i, ":  ", train_data.shape
    print np.any(train_data==np.nan)
    x_train = max_min_scaler.fit_transform(train_data)

    test_data = np.array(dfTest_currentday[feat].values,dtype='int32').reshape(-1, 1)
    x_test = max_min_scaler.transform(test_data)
    if i == 0:
        X_train, X_test = x_train, x_test
    else:
        X_train, X_test = np.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))




from sklearn.feature_selection import SelectKBest
#选择K个最好的特征，返回选择特征后的数据
from sklearn.feature_selection import chi2
print X_train.shape
feature_selector = SelectKBest(chi2, k='all')# 200 0.76  #300 0.77 #400 0.77


X_train =feature_selector.fit_transform(X_train,y_train)
X_test = feature_selector.transform(X_test)




# model training
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1')


from sklearn.cross_validation import KFold
from sklearn.metrics import auc,roc_curve
import matplotlib.pyplot as plt

kf = KFold(dfTrain_currentday.shape[0], n_folds=10)
print(kf)
ROC_AUCs=[]
LOGLOSS = []
# KFold(n=4, n_folds=2, shuffle=False,random_state=None)
from sklearn.metrics import confusion_matrix


for train_index, test_index in kf:
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train_temp, X_test_temp = X_train[train_index,:], X_train[test_index,:]
    y_train_temp, y_test_temp = y_train[train_index], y_train[test_index]
    lr.fit(X_train_temp,y_train_temp)
    pred_probas = lr.predict_proba(X_test_temp)[:, 1]
    pred = lr.predict(X_test_temp)
    fpr, tpr, _ = roc_curve(y_test_temp, pred_probas)
    roc_auc = auc(fpr, tpr)
    ROC_AUCs.append(roc_auc)
    logloss = util.logloss(y_test_temp,pred_probas)
    LOGLOSS.append(logloss)
    print(" logloss:",logloss," auc: ",roc_auc)
    print confusion_matrix(y_test_temp,pred)

print "K FOLD: AUC:  ===>",np.mean(ROC_AUCs)
print "K FOLD: LOGLOSS:  ===>",np.mean(LOGLOSS)


# submission

print dfTest['instanceID'].shape
print proba_test.shape

df = pd.DataFrame({"instanceID": np.array(dfTest["instanceID"].values,dtype='int32'), "proba": proba_test})
df.sort_values("instanceID", inplace=True)
df.to_csv("submission.csv", index=False)
with zipfile.ZipFile("submission.zip", "w") as fout:
    fout.write("submission.csv", compress_type=zipfile.ZIP_DEFLATED)






dfSubmit = data[data.source=='submit']











