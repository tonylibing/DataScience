#!/usr/bin/env python
# encoding: utf-8

# bagging 修正正负样本比例不均衡
import pandas as pd
import feature as fea

positive_num = fea.train_user_ad_app[1==fea.train_user_ad_app['label']].value.shape[0]
negative_num = fea.train_user_ad_app[0==fea.train_user_ad_app['label']].value.shape[0]

print negative_num / float(positive_num)
#可以看出正负样本数量相差非常大，严重unbalance
#我们用bagging修正后， 处理不均衡样本的b(l)agging 来进行训练和实验

from blagging import BlaggingClassifier

classifier = BlaggingClassifier(n_jobs=-1)
classifier.fit(fea.x_user_ad_app, fea.y_user_ad_app)

classifier.predict_proba(fea.x_test_clean)


#TEST
test_data = pd.merge(test_data, user, on='userID')
test_user_ad = pd.merge(test_data, ad, on='creativeID')
test_user_ad_app = pd.merge(test_user_ad, app_cateories, on="appID")

x_test_clean = test_user_ad_app.loc[:, []]
