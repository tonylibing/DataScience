from processor import OutliersFilter

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from itertools import combinations


for num in [10, 50, 100, 1000]:
    x = np.random.normal(0, 0.5, num-3)
    x = np.r_[x, -3, -10, 12]

df=pd.DataFrame(x,columns=["test"])

ot = OutliersFilter("test",method="percentile",threshold=95)
ot.fit(df)
o = ot.transform(df)

ot = OutliersFilter("test",method="mad",threshold=3.5)
ot.fit(df)
o = ot.transform(df)