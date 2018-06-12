import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import gc
import math
import numpy as np
from tqdm import tqdm
import jieba
import re

def remove_illegal_char(user_input):
    line = user_input.strip().replace("<br>", "")
    line, _ = re.subn('【', '', line)
    line, _ = re.subn('】', '', line)
    line, _ = re.subn(r'/:[a-z]+', '', line)
    line, _ = re.subn(r'%[0-9A-Z]+', '', line)
    line, _ = re.subn(r' +', ' ', line)
    return line.upper()


# data_dir="/home/tanglek/dataset/atec/nlp"
data_dir="E:/dataset/atec/nlp"
jieba.load_userdict(os.path.join(data_dir,'jieba','user_dict.txt'))
corpus_data=[]
with open(os.path.join(data_dir,"atec_nlp_sim_train.csv"),'r',encoding="utf8") as f:
    for i,line in enumerate(f):
        data_dict={}
        l = remove_illegal_char(line)
        strs = l.strip().split('\t')
        if i % 100000 == 0:
            print(i)
        data_dict['first']=strs[1]
        data_dict['second']=strs[2]
        data_dict['label']=int(strs[3])
        corpus_data.append(data_dict)

corpus=pd.DataFrame(corpus_data)
corpus['first_seg']=corpus['first'].apply(lambda x:jieba.lcut(x.strip()))
corpus['second_seg']=corpus['second'].apply(lambda x:jieba.lcut(x.strip()))

corpus_seg = pd.concat([corpus['first_seg'],corpus['second_seg']],axis=0,ignore_index=True)
corpus_seg2 = corpus_seg.apply(lambda x:' '.join(x))

corpus_seg2.to_csv("E:/dataset/atec/nlp/ant_corpus.txt",index=False,header=False,encoding='utf-8')