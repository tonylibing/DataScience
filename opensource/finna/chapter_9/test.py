# -*- coding: utf-8 -*-
import pandas as pd
import datetime
import collections
import numpy as np
import numbers
import random
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm
import sys
import pickle
reload(sys)
sys.setdefaultencoding( "utf-8")
import scorecard_fucntions
reload(scorecard_fucntions)
from scorecard_fucntions import *
from sklearn.linear_model import LogisticRegressionCV



#########################################################################################################
#Step 0: Initiate the data processing work, including reading csv files, checking the consistency of Idx#
#########################################################################################################
#path= 'D:/workspace/DataScience/data/ppd/'
path= '~/workspace/DataScience/data/ppd/'




##################################################################################
# Step 2: Makeup missing value for categorical variables and continuous variables#
##################################################################################
allData = pd.read_csv(path+'bank default/allData_0.csv',header = 0,encoding = 'gbk')
allFeatures = list(allData.columns)
allFeatures.remove('ListingInfo')
allFeatures.remove('target')
allFeatures.remove('Idx')

#check columns and remove them if they are a constant
for col in allFeatures:
    if len(set(allData[col])) == 1:
        del allData[col]
        allFeatures.remove(col)

#devide the whole independent variables into categorical type and numerical type
numerical_var = []
for var in allFeatures:
    uniq_vals = list(set(allData[var]))
    if np.nan in uniq_vals:
        uniq_vals.remove( np.nan)
    if len(uniq_vals) >= 10 and isinstance(uniq_vals[0],numbers.Real):
        numerical_var.append(var)

categorical_var = [i for i in allFeatures if i not in numerical_var]



'''
For each categorical variable, if the missing value occupies more than 50%, we remove it.
Otherwise we will use missing as a special status
'''
missing_pcnt_threshould_1 = 0.5
for col in categorical_var:
    missingRate = MissingCategorial(allData,col)
    print '{0} has missing rate as {1}'.format(col,missingRate)
    if missingRate > missing_pcnt_threshould_1:
        categorical_var.remove(col)
        del allData[col]
    if 0 < missingRate < missing_pcnt_threshould_1:
        allData[col] = allData[col].map(lambda x: str(x).upper())


'''
For continuous variable, if the missing value is more than 30%, we remove it.
Otherwise we use random sampling method to make up the missing
'''
missing_pcnt_threshould_2 = 0.3
for col in numerical_var:
    missingRate = MissingContinuous(allData, col)
    print '{0} has missing rate as {1}'.format(col, missingRate)
    if missingRate > missing_pcnt_threshould_2:
        numerical_var.remove(col)
        del allData[col]
        print 'we delete variable {} because of its high missing rate'.format(col)
    else:
        if missingRate > 0:
            not_missing = allData.loc[allData[col] == allData[col]][col]
            value = not_missing.median()
            allData[col]=allData[col].fillna(value)
            missingRate2 = MissingContinuous(allData, col)
            print 'missing rate after making up is:{}'.format(str(missingRate2))


allData.to_csv(path+'bank default/allData_1b.csv', header=True,encoding='gbk', columns = allData.columns, index=False)


####################################
# Step 3: Group variables into bins#
####################################
#for each categorical variable, if it has distinct values more than 5, we use the ChiMerge to merge it

trainData = pd.read_csv(path+'bank default/allData_1b.csv',header = 0, encoding='gbk')
allFeatures = list(trainData.columns)
allFeatures.remove('ListingInfo')
allFeatures.remove('target')
allFeatures.remove('Idx')
#devide the whole independent variables into categorical type and numerical type
numerical_var = []
for var in allFeatures:
    uniq_vals = list(set(trainData[var]))
    if np.nan in uniq_vals:
        uniq_vals.remove( np.nan)
    if len(uniq_vals) >= 10 and isinstance(uniq_vals[0],numbers.Real):
        numerical_var.append(var)

categorical_var = [i for i in allFeatures if i not in numerical_var]

for col in categorical_var:
    trainData[col] = trainData[col].map(lambda x: str(x).upper())


'''
For cagtegorical variables, follow the below steps
1, if the variable has distinct values more than 5, we calculate the bad rate and encode the variable with the bad rate
2, otherwise:
(2.1) check the maximum bin, and delete the variable if the maximum bin occupies more than 90%
(2.2) check the bad percent for each bin, if any bin has 0 bad samples, then combine it with samllest non-zero bad bin,
        and then check the maximum bin again
'''
deleted_features = []  #delete the categorical features in one of its single bin occupies more than 90%
encoded_features = {}
merged_features = {}
var_IV = {}  #save the IV values for binned features
var_WOE = {}
for col in categorical_var:
    print 'we are processing {}'.format(col)
    if len(set(trainData[col]))>5:
        print '{} is encoded with bad rate'.format(col)
        col0 = str(col)+'_encoding'
        #(1), calculate the bad rate and encode the original value using bad rate
        encoding_result = BadRateEncoding(trainData, col, 'target')
        trainData[col0], br_encoding = encoding_result['encoding'],encoding_result['br_rate']
        #(2), push the bad rate encoded value into numerical varaible list
        numerical_var.append(col0)
        #(3), save the encoding result, including new column name and bad rate
        encoded_features[col] = [col0, br_encoding]
        #(4), delete the original value
        #del trainData[col]
        deleted_features.append(col)
    else:
        maxPcnt = MaximumBinPcnt(trainData, col)
        if maxPcnt > 0.9:
            print '{} is deleted because of large percentage of single bin'.format(col)
            deleted_features.append(col)
            categorical_var.remove(col)
            #del trainData[col]
            continue
        bad_bin = trainData.groupby([col])['target'].sum()
        if min(bad_bin) == 0:
            print '{} has 0 bad sample!'.format(col)
            col1 = str(col) + '_mergeByBadRate'
            #(1), determine how to merge the categories
            mergeBin = MergeBad0(trainData, col, 'target')
            #(2), convert the original data into merged data
            trainData[col1] = trainData[col].map(mergeBin)
            maxPcnt = MaximumBinPcnt(trainData, col1)
            if maxPcnt > 0.9:
                print '{} is deleted because of large percentage of single bin'.format(col)
                deleted_features.append(col)
                categorical_var.remove(col)
                del trainData[col]
                continue
            #(3) if the merged data satisify the requirement, we keep it
            merged_features[col] = [col1, mergeBin]
            WOE_IV = CalcWOE(trainData, col1, 'target')
            var_WOE[col1] = WOE_IV['WOE']
            var_IV[col1] = WOE_IV['IV']
            #del trainData[col]
            deleted_features.append(col)
        else:
            WOE_IV = CalcWOE(trainData, col, 'target')
            var_WOE[col] = WOE_IV['WOE']
            var_IV[col] = WOE_IV['IV']


'''
For continous variables, we do the following work:
1, split the variable by ChiMerge (by default into 5 bins)
2, check the bad rate, if it is not monotone, we decrease the number of bins until the bad rate is monotone
3, delete the variable if maximum bin occupies more than 90%
'''
var_cutoff = {}
for col in numerical_var:
    print "{} is in processing".format(col)
    col1 = str(col) + '_Bin'
    #(1), split the continuous variable and save the cutoff points. Particulary, -1 is a special case and we separate it into a group
    if -1 in set(trainData[col]):
        special_attribute = [-1]
    else:
        special_attribute = []
    cutOffPoints = ChiMerge_MaxInterval(trainData, col, 'target',special_attribute=special_attribute)
    var_cutoff[col] = cutOffPoints
    trainData[col1] = trainData[col].map(lambda x: AssignBin(x, cutOffPoints,special_attribute=special_attribute))

    #(2), check whether the bad rate is monotone
    BRM = BadRateMonotone(trainData, col1, 'target',special_attribute=special_attribute)
    if not BRM:
        for bins in range(4,1,-1):
            cutOffPoints = ChiMerge_MaxInterval(trainData, col, 'target',max_interval = bins,special_attribute=special_attribute)
            trainData[col1] = trainData[col].map(lambda x: AssignBin(x, cutOffPoints,special_attribute=special_attribute))
            BRM = BadRateMonotone(trainData, col1, 'target',special_attribute=special_attribute)
            if BRM:
                break
        var_cutoff[col] = cutOffPoints

    #(3), check whether any single bin occupies more than 90% of the total
    maxPcnt = MaximumBinPcnt(trainData, col1)
    if maxPcnt > 0.9:
        #del trainData[col1]
        deleted_features.append(col)
        numerical_var.remove(col)
        print 'we delete {} because the maximum bin occupies more than 90%'.format(col)
        continue
    WOE_IV = CalcWOE(trainData, col1, 'target')
    var_IV[col] = WOE_IV['IV']
    var_WOE[col] = WOE_IV['WOE']
    #del trainData[col]



trainData.to_csv(path+'bank default/allData_2a.csv', header=True,encoding='gbk', columns = trainData.columns, index=False)