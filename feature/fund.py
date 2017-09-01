# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

#fund_data=pd.read_excel("stock_fund.xlsx")
#fund_data.to_csv("stock_fund.csv",headers=True,index=False,encoding='utf-8')
fund_data=pd.read_excel("stock_fund.xlsx")

#dev_12m_qt=np.percentile(fund_data['dev_12m'], [0,15,65,90,95,100])
#dev_12m_qt=(dev_12m_qt[1:]+dev_12m_qt[:-1])*0.5
#beta_12m_qt=np.percentile(fund_data['beta_12m'], [0,15,65,90,95,100])
#beta_12m_qt=(beta_12m_qt[1:]+beta_12m_qt[:-1])*0.5
dev_12m_qt=np.nanpercentile(fund_data['dev_12m'], [0,15,65,90,95,100])
dev_12m_qt=(dev_12m_qt[1:]+dev_12m_qt[:-1])*0.5
beta_12m_qt=np.nanpercentile(fund_data['beta_12m'], [0,15,65,90,95,100])
beta_12m_qt=(beta_12m_qt[1:]+beta_12m_qt[:-1])*0.5

fund_pool.columns=['fund_code','fund_name']
fund_data['fund_code']=fund_data['fund_code'].apply(lambda x:x[:-3]).astype('int64')
fund=pd.merge(fund_pool,fund_data,how='left',on='fund_code')

list(set(fund_pool['fund_code'].values) & set(fund_data['fund_code'].values))


f1=pd.read_excel("QDII基金.xlsx")
f2=pd.read_excel("股票型基金.xlsx")
f3=pd.read_excel("混合型基金.xlsx")
f=pd.concat([f1,f2,f3],axis=0)
f.columns=range(0,23)

f['fund_code']=f[20].apply(lambda x:x[:-3]).astype('int64')
pl = list(set(fund_pool['fund_code'].values) & set(f[20].values))


all_fund_data=pd.read_excel("non_currency_fund.xlsx")
dev_12m_qt=np.nanpercentile(fund_data['dev_12m'], [0,15,65,90,95,100])
dev_12m_qt=(dev_12m_qt[1:]+dev_12m_qt[:-1])*0.5
beta_12m_qt=np.nanpercentile(fund_data['beta_12m'], [0,15,65,90,95,100])
beta_12m_qt=(beta_12m_qt[1:]+beta_12m_qt[:-1])*0.5

fund_pool=pd.read_csv("fund_pool.csv",header=None)
fund_pool.columns=['fund_code','fund_name']
zntg_pool9=pd.read_excel("智能投顾基金池月度跟踪表.xlsx").dropna()

from pandas import ExcelWriter
fund_pool_detail=pd.merge(fund_pool,all_fund_data,how='left',left_on='fund_code',right_on=u'证券代码').drop(['fund_code','fund_name'],axis=1)
writer = ExcelWriter("fund_pool.xlsx")
fund_pool_detail.to_excel(writer,"sheet1")
writer.save()

zntg_pool9_detail=pd.merge(zntg_pool9,all_fund_data,how='left',left_on=u'基金代码',right_on=u'证券代码')
writer = ExcelWriter("智能投顾基金池月度跟踪表_detail.xlsx")
fund_pool_detail.to_excel(writer,"sheet1")
writer.save()
