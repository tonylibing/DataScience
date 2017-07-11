# -*- coding:utf-8 -*-
import datetime
import pandas as pd
import numpy as np
from scipy import stats
from compiler.ast import flatten
import time

__author__ = 'kevin'


def segment(X, Y, seg=None, n=20):
    # Monotonic binning
    r = 0
    x1 = flatten(X)
    while 1:
        try:
            df1 = pd.DataFrame({"X": x1, "Y": Y, "bucket": pd.qcut(x1, n)})
            df2 = df1.groupby('bucket', as_index=True)
            # spearman correlation:Correlations of -1 or +1 imply an exact monotonic relationship
            r, p = stats.spearmanr(df2['X'].mean(), df2['Y'].mean())
        except:
            pass

        if np.abs(r) < 1:
            n -= 1
        else:
            print 'r, p, n=', r, p, n
            break

    total = df2['Y'].count()
    bad = df2['Y'].sum()
    good = total - bad

    df3 = pd.DataFrame(bad, columns=['bad'])
    df3['bad'] = bad
    df3['good'] = good

    segments = []
    for i in range(1, n + 1):
        segments.append(seg + str(i))
    df3['segment'] = segments

    df4 = df3.reset_index()

    rangement = []
    bucket = df4['bucket']
    for buck in bucket:
        rangement.append(buck[1:len(buck) - 1].replace(', ', '#'))
    df3['range'] = rangement

    bad_ratio = bad / df3['bad'].sum()
    good_ratio = good / df3['good'].sum()
    woe = np.log(bad_ratio / good_ratio)
    iv = woe * (bad_ratio - good_ratio)
    df3['woe'] = woe
    df3['iv'] = iv

    return df3


if __name__ == '__main__':
    """
    optimal binning : Monotonic binning
    先等频分，然后再基于x,y是否单调，来确认最佳分箱个数
    分箱效果不理想，数字较小的特征分不出来
    """
    begin = time.time()
    # import data
    user_feature = "/home/kevin/temp/users3.csv"
    features = np.loadtxt(user_feature, delimiter=',')
    qualifiers = "label,userid," \
                 "f1,f2,f3,f4,f5," \
                 "f6,f7,f8,ip_cities,match_num," \
                 "refund_times,f9,f10".split(',')
    df_src = pd.DataFrame(features, index=np.arange(features.shape[0]), columns=qualifiers)
    df = df_src
    column_len = len(df.columns)
    path = "/home/kevin/temp/%s.csv"
    for i in range(1, column_len-4):
        column_name = 'f%s' % i
        seg = 'A%s' % i
        # segment
        df_seg = segment(df.loc[:, [column_name]].loc[:, column_name], df['label'], seg=seg)
        file_name = path % column_name
        df_seg.to_csv(file_name)
        print df_seg

    end = time.time()
    now = datetime.datetime.now()
    dat = now.strftime("%Y-%m-%d %H:%M:%S")
    print dat, 'total cost', round((end - begin) / 60, 3), 'minutes'