#!/usr/bin/env python
# encoding: utf-8

from skleran.model_selection import GridSearchCV
from skleran.ensemble import RandomForestClassifier
import feature as fea

param_grid = {
    'n_estimators':[10, 100, 500, 1000],
    'max_features':[0.5, 0.7, 0.8, 0.9]
}
rf = RandomForestClassifier()

rfc = GridSearchCV(rf, param_grid, scoring='neg_log_loss', cv=3, n_jobs=2)

rfc.fit(fea.x_user_ad_app, fea.y_user_ad_app.reshape(fea.y_user_ad_app.shape[0], ))

print(rfc.best_score_)
print(rfc.best_params_)
