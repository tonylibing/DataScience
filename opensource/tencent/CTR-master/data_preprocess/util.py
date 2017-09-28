#coding=utf-8
import  pandas as pd
import numpy as np
import scipy as sp



def read_csv_file(f,logging=False):
    print "============================read========================",f
    data = pd.read_csv(f)
    if logging:
        print data.head(10)
        print f,"  columns...."
        print data.columns.values
        print data.describe()
        print data.info()
    return  data

def categories_process_first_class(cate):
    cate = str(cate)
    if len(cate)==1:
        if int(cate)==0:
            return 0
    else:
        return int(cate[0])

def categories_process_second_class(cate):
    cate = str(cate)
    if len(cate)<3:
        return 0
    else:
        return int(cate[1:])


def age_process(age):
    age = int(age)
    if age==0:
        return 0
    elif age<15:
        return 1
    elif age<25:
        return 2
    elif age<40:
        return 3
    elif age<60:
        return 4
    else:
        return 5




def hometown_process_province(hometown):
    hometown = str(hometown)
    province = int(hometown[0:2])
    return province

def hometown_process_city(hometown):
    hometown = str(hometown)
    if len(hometown)>1:
        province = int(hometown[2:])
    else:
        province = 0
    return province



def get_time_day(t):
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



##################################################################
def logloss(act, pred):
  epsilon = 1e-15
  pred = sp.maximum(epsilon, pred)
  pred = sp.minimum(1-epsilon, pred)
  ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
  ll = ll * -1.0/len(act)
  return ll
