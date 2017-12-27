import pandas as pd
import numpy as np
import json
from dateutil.parser import parse
from pandas.io.json import json_normalize
from sys import platform
from time import gmtime, strftime,localtime
import datetime

dir_prefix = ""

if platform == "linux" or platform == "linux2":
    dir_prefix="~"
elif platform == "win32":
    dir_prefix="E:"

data=pd.read_csv(dir_prefix + "/dataset/user_profile.csv",sep=',')
    
cat_df = json_normalize(data['category_preference'].apply(lambda x:json.loads(x)).tolist())
cat_df.fillna(0,inplace=True)

amt_df = json_normalize(data['amount_preference'].apply(lambda x:json.loads(x)).tolist())
amt_df.fillna(0,inplace=True)

period_df = json_normalize(data['period_preference'].apply(lambda x:json.loads(x)).tolist())
period_df.fillna(0,inplace=True)

data['request_time']=strftime("%Y-%m-%d %H:%M:%S", localtime())
data['dayOfWeek'] = data['request_time'].apply(lambda x: parse(x).weekday())
data['hourOfDay'] = data['request_time'].apply(lambda x: parse(x).hour)
data['dayOfMonth'] = data['request_time'].apply(lambda x: parse(x).day)
data.drop(['request_time'],axis=1,inplace=True)
data['address'] = 1
data['12m-24m']=0.0
data['24m-36m']=0.0
data['36m+']=0.0
pref = pd.concat([data,cat_df,amt_df,period_df],axis=1)
pref.drop(['recent_invest','recent_invest', 'recent_browse', 'category_preference', 'amount_preference', 'period_preference'],axis=1,inplace=True)

bdw_user_profile = pd.read_csv(dir_prefix +"/dataset/bdw_crm_user_profile.csv",sep=',')

up = pd.merge(pref,bdw_user_profile,on='user_id',how='left')

products =pd.read_csv(dir_prefix +"/dataset/online_products.csv",sep=',')
products['invest_period_by_days'].fillna(0,inplace=True)
# products=products[products['interest_rate']]


def df_crossjoin(df1, df2, **kwargs):
    """
    Make a cross join (cartesian product) between two dataframes by using a constant temporary key.
    Also sets a MultiIndex which is the cartesian product of the indices of the input dataframes.
    See: https://github.com/pydata/pandas/issues/5401
    :param df1 dataframe 1
    :param df1 dataframe 2
    :param kwargs keyword arguments that will be passed to pd.merge()
    :return cross join of df1 and df2
    """
    df1['_tmpkey'] = 1
    df2['_tmpkey'] = 1

    res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)
    res.index = pd.MultiIndex.from_product((df1.index, df2.index))

    df1.drop('_tmpkey', axis=1, inplace=True)
    df2.drop('_tmpkey', axis=1, inplace=True)

    return res


df = df_crossjoin(up,products)

df.to_csv(dir_prefix +"/dataset/user_profile_products.csv",index=False,header=True)
df2=df[df['user_id']==876553]
df2.to_csv(dir_prefix +"/dataset/user_profile_products2.csv",index=False,header=True)