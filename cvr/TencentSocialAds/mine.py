# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:17:17 2017

@author: win7
"""
#数据说明 http://algo.tpai.qq.com/home/information/index.html
import pandas as pd

train = pd.read_csv("../../data/tencent_ad/first/train.csv")
user = pd.read_csv("../../data/tencent_ad/first/user.csv")
data = pd.merge(train, user, how='left', on='userID')
count = data.groupby('gender', as_index=False).apply(lambda x: x['label'].sum())
print(count)  # gender is a good feature: gender = 1 is best

count = data.groupby('education', as_index=False).apply(lambda x: x['label'].sum())
print(count)  # education is a good feature