import numpy as np
import pandas as pd

def ks_statistic(Y,Y_hat):
    data = {"Y":Y,"Y_hat":Y_hat}
    df = pd.DataFrame(data)
    bins = np.array([-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    category = pd.cut(df["Y_hat"],bins=bins)
    category = category.sort_values()
    #max_index = len(np.unique(df["Y_hat"]))
    Y = df.ix[category.index,:]['Y']
    Y_hat = df.ix[category.index,:]['Y_hat']
    df2 = pd.concat([Y,Y_hat],axis=1)
    df3 = pd.pivot_table(df2,values = ['Y_hat'],index ='Y_hat',columns='Y',aggfunc=len,fill_value=0)
    df4 = np.cumsum(df3)
    df5 = df4/df4.iloc[:,1].max()
    ks = max(abs(df5.iloc[:,0] - df5.iloc[:,1]))
    return ks/len(bins)