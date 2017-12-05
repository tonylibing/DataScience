import pandas as pd
from pandas import DataFrame
import datetime
import collections
import numpy as np
import numbers
import random
from pandas.tools.plotting import scatter_matrix
from itertools import combinations
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import sys
reload(sys)
sys.setdefaultencoding( "utf-8")
sys.path.append(path+"/Notes/07 申请评分卡中的数据预处理和特征衍生/")
from scorecard_fucntions import *
# -*- coding: utf-8 -*-

###############################################################################################
# Step 0: Reading the raw testing data, which are in the same structure with training datasets#
###############################################################################################

data1b = pd.read_csv(path+'/数据/bank default/PD-Second-Round-Data/first round test data/LogInfo_9w_2.csv', header = 0)
data2b = pd.read_csv(path+'/数据/bank default/PD-Second-Round-Data/first round test data/Kesci_Master_9w_gbk_2.csv', header = 0,encoding = 'gbk')
data3bb = pd.read_csv(path+'/数据/bank default/PD-Second-Round-Data/first round test data/Userupdate_Info_9w_2.csv', header = 0)




#################################################################################
# Step 1: Derivate the features using in the same with as it in training dataset#
#################################################################################

### Extract the applying date of each applicant
data1b['logInfo'] = data1b['LogInfo3'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
data1b['Listinginfo'] = data1b['Listinginfo1'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
data1b['ListingGap'] = data1b[['logInfo','Listinginfo']].apply(lambda x: (x[1]-x[0]).days,axis = 1)

'''
We use 180 as the maximum time window to work out some features in data1b.
The used time windows can be set as 7 days, 30 days, 60 days, 90 days, 120 days, 150 days and 180 days.
We calculate the count of total and count of distinct of each raw field within selected time window.
'''
time_window = [7, 30, 60, 90, 120, 150, 180]
var_list = ['LogInfo1','LogInfo2']
data1bGroupbyIdx = pd.DataFrame({'Idx':data1b['Idx'].drop_duplicates()})

for tw in time_window:
    data1b['TruncatedLogInfo'] = data1b['Listinginfo'].map(lambda x: x + datetime.timedelta(-tw))
    temp = data1b.loc[data1b['logInfo'] >= data1b['TruncatedLogInfo']]
    for var in var_list:
        #count the frequences of LogInfo1 and LogInfo2
        count_stats = temp.groupby(['Idx'])[var].count().to_dict()
        data1bGroupbyIdx[str(var)+'_'+str(tw)+'_count'] = data1bGroupbyIdx['Idx'].map(lambda x: count_stats.get(x,0))

        # count the distinct value of LogInfo1 and LogInfo2
        Idx_UserupdateInfo1 = temp[['Idx', var]].drop_duplicates()
        uniq_stats = Idx_UserupdateInfo1.groupby(['Idx'])[var].count().to_dict()
        data1bGroupbyIdx[str(var) + '_' + str(tw) + '_unique'] = data1bGroupbyIdx['Idx'].map(lambda x: uniq_stats.get(x,0))

        # calculate the average count of each value in LogInfo1 and LogInfo2
        data1bGroupbyIdx[str(var) + '_' + str(tw) + '_avg_count'] = data1bGroupbyIdx[[str(var)+'_'+str(tw)+'_count',str(var) + '_' + str(tw) + '_unique']].\
            apply(lambda x: x[0]*1.0/x[1], axis=1)


data3b['ListingInfo'] = data3b['ListingInfo1'].map(lambda x: datetime.datetime.strptime(x,'%Y/%m/%d'))
data3b['UserupdateInfo'] = data3b['UserupdateInfo2'].map(lambda x: datetime.datetime.strptime(x,'%Y/%m/%d'))
data3b['ListingGap'] = data3b[['UserupdateInfo','ListingInfo']].apply(lambda x: (x[1]-x[0]).days,axis = 1)
data3b['UserupdateInfo1'] = data3b['UserupdateInfo1'].map(ChangeContent)
data3bGroupbyIdx = pd.DataFrame({'Idx':data3b['Idx'].drop_duplicates()})

time_window = [7, 30, 60, 90, 120, 150, 180]
for tw in time_window:
    data3b['TruncatedLogInfo'] = data3b['ListingInfo'].map(lambda x: x + datetime.timedelta(-tw))
    temp = data3b.loc[data3b['UserupdateInfo'] >= data3b['TruncatedLogInfo']]

    #frequency of updating
    freq_stats = temp.groupby(['Idx'])['UserupdateInfo1'].count().to_dict()
    data3bGroupbyIdx['UserupdateInfo_'+str(tw)+'_freq'] = data3bGroupbyIdx['Idx'].map(lambda x: freq_stats.get(x,0))

    # number of updated types
    Idx_UserupdateInfo1 = temp[['Idx','UserupdateInfo1']].drop_duplicates()
    uniq_stats = Idx_UserupdateInfo1.groupby(['Idx'])['UserupdateInfo1'].count().to_dict()
    data3bGroupbyIdx['UserupdateInfo_' + str(tw) + '_unique'] = data3bGroupbyIdx['Idx'].map(lambda x: uniq_stats.get(x, x))

    #average count of each type
    data3bGroupbyIdx['UserupdateInfo_' + str(tw) + '_avg_count'] = data3bGroupbyIdx[['UserupdateInfo_'+str(tw)+'_freq', 'UserupdateInfo_' + str(tw) + '_unique']]. \
        apply(lambda x: x[0] * 1.0 / x[1], axis=1)

    #whether the applicant changed items like IDNUMBER,HASBUYCAR, MARRIAGESTATUSID, PHONE
    Idx_UserupdateInfo1['UserupdateInfo1'] = Idx_UserupdateInfo1['UserupdateInfo1'].map(lambda x: [x])
    Idx_UserupdateInfo1_V2 = Idx_UserupdateInfo1.groupby(['Idx'])['UserupdateInfo1'].sum()
    for item in ['_IDNUMBER','_HASBUYCAR','_MARRIAGESTATUSID','_PHONE']:
        item_dict = Idx_UserupdateInfo1_V2.map(lambda x: int(item in x)).to_dict()
        data3bGroupbyIdx['UserupdateInfo_' + str(tw) + str(item)] = data3bGroupbyIdx['Idx'].map(lambda x: item_dict.get(x, x))

# Combine the above features with raw features in PPD_Training_Master_GBK_3_1_Training_Set
allData = pd.concat([data2b.set_index('Idx'), data3bGroupbyIdx.set_index('Idx'), data1bGroupbyIdx.set_index('Idx')],axis= 1)
allData.to_csv(path+'/数据/bank default/allData_0_Test.csv',encoding = 'gbk')


####################################################
# Step 2: Makeup missing value continuous variables#
####################################################

#make some change to the string type varaiable, espeically converting nan to NAN so as it could be read in the mapping dictionary
testData = pd.read_csv(path+'/数据/bank default/allData_0_Test.csv',header = 0,encoding = 'gbk')
allData[col] = allData[col].map(lambda x: str(x).upper())
changedCols = ['WeblogInfo_20', 'UserInfo_17']
for col in changedCols:
    testData[col] = testData[col].map(lambda x: str(x).upper())

allFeatures = list(testData.columns)
allFeatures.remove('ListingInfo')
allFeatures.remove('target')
allFeatures.remove('Idx')


### read the saved WOE encoding dictionary ###
fread = open(path+'/数据/bank default/var_WOE.pkl','r')
WOE_dict = pickle.load(fread)
fread.close()

### the below features are selected into the scorecard model in Step 5
var_WOE_model = ['UserInfo_15_encoding_WOE', u'ThirdParty_Info_Period6_10_WOE', u'ThirdParty_Info_Period5_2_WOE', 'UserInfo_16_encoding_WOE', 'WeblogInfo_20_encoding_WOE',
            'UserInfo_7_encoding_WOE', u'UserInfo_17_WOE', u'ThirdParty_Info_Period3_10_WOE', u'ThirdParty_Info_Period1_10_WOE', 'WeblogInfo_2_encoding_WOE',
            'UserInfo_1_encoding_WOE']


#some features are catgorical type and we need to encode them
var_encoding = [i.replace('_WOE','').replace('_encoding','') for i in var_WOE_model if i.find('_encoding')>=0]
for col in var_encoding:
    print col
    [col1, encode_dict] = encoded_features[col]
    testData[col1] = testData[col].map(lambda x: encode_dict.get(str(x),-99999))
    col2 = str(col1) + "_WOE"
    cutOffPoints = var_cutoff[col1]
    special_attribute = []
    if - 1 in cutOffPoints:
        special_attribute = [-1]
    binValue = testData[col1].map(lambda x: AssignBin(x, cutOffPoints, special_attribute=special_attribute))
    testData[col2] = binValue.map(lambda x: WOE_dict[col1][x])

#other features can be mapped to WOE directly
var_others = [i.replace('_WOE','').replace('_encoding','') for i in var_WOE_model if i.find('_encoding') < 0]
for col in var_others:
    print col
    col2 = str(col) + "_WOE"
    if col in var_cutoff.keys():
        cutOffPoints = var_cutoff[col]
        special_attribute = []
        if - 1 in cutOffPoints:
            special_attribute = [-1]
        binValue = testData[col].map(lambda x: AssignBin(x, cutOffPoints, special_attribute=special_attribute))
        testData[col2] = binValue.map(lambda x: WOE_dict[col][x])
    else:
        testData[col2] = testData[col].map(lambda x: WOE_dict[col][x])


### make the design matrix
X = testData[var_WOE_model]
X['intercept'] = [1]*X.shape[0]
y = testData['target']


#### load the training model
saveModel =open(path+'/数据/bank default/LR_Model_Normal.pkl','r')
LR = pickle.load(saveModel)
saveModel.close()

y_pred = LR.predict(X)

scorecard_result = pd.DataFrame({'prob':y_pred, 'target':y})
# we check the performance of the model using KS and AR
# both indices should be above 30%
performance = KS_AR(scorecard_result,'prob','target')
print "KS and AR for the scorecard in the test dataset are %.0f%% and %.0f%%"%(performance['AR']*100,performance['KS']*100)
