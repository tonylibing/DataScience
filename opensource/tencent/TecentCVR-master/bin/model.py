#!/usr/bin/env python
# encoding: utf-8


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt

import feature as fea

# random forest
forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
forest.fit(fea.x_user_ad_app ,fea.y_user_ad_app.reshape(fea.y_user_ad_app.shape[0], ))

# RF 计算重要度
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]


for f in range(fea.x_user_ad_app.shape[1]):
    print("%2d) %-*s %f" % (f+1, 30,
                            fea.feat_labels[indices[f]],
                            importances[indices[f]]
                            ))

plt.title('Feature Importaces')
plt.bar(range(fea.x_user_ad_app.shape[1]), importances[indices], color='lightblue', align='center')
plt.xticks(range(fea.x_user_ad_app.shape[1]), fea.feat_labels[indices], rotation=90)
plt.xlim([-1, fea.x_user_ad_app.shape[1]])
plt.tight_layout()
plt.show()


