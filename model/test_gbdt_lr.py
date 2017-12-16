from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.datasets  import  make_hastie_10_2
from GBDTLRClassifier import XgboostLRClassifier

X, y = make_hastie_10_2(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999)


lr = LogisticRegression(C=1.0, penalty='l2', random_state=42)
lr.fit(X_train,y_train)
y_pre= lr.predict(X_test)
y_pro= lr.predict_proba(X_test)[:,1]
print("LR Test AUC Score : {0}", metrics.roc_auc_score(y_test, y_pro))
print("LR  Test Accuracy : {0}" , metrics.accuracy_score(y_test, y_pre))


gbdtlr = XgboostLRClassifier()
gbdtlr.fit(X_train,y_train)
y_pre= gbdtlr.predict(X_test)
y_pro= gbdtlr.predict_proba(X_test)[:,1]
print("GBDT+LR Test AUC Score : {0}", metrics.roc_auc_score(y_test, y_pro))
print("GBDT+LR  Test Accuracy : {0}" , metrics.accuracy_score(y_test, y_pre))

gbm = xgb.XGBClassifier(n_estimators=30,learning_rate =0.3,max_depth=3,min_child_weight=1,gamma=0.3,subsample=0.7,colsample_bytree=0.7,objective= 'binary:logistic',nthread=-1,scale_pos_weight=1,reg_alpha=1e-05,reg_lambda=1,seed=27)
gbm.fit(X_train,y_train)
y_pre= gbm.predict(X_test)
# y_pre_leaf = gbm.predict(X_test,pred_leaf=True)
# print(y_pre_leaf.shape)
y_pro= gbm.predict_proba(X_test)[:,1]
print("Xgboost model Test AUC Score : {0}", metrics.roc_auc_score(y_test, y_pro))
print("Xgboost model Test Accuracy : {0}" , metrics.accuracy_score(y_test, y_pre))

lgbm = lgb.LGBMClassifier(boosting_type='gbdt',  max_depth=3, learning_rate=0.3, n_estimators=30, min_child_weight=1,subsample=0.7,  colsample_bytree=0.7, reg_alpha=1e-05, reg_lambda=1)
lgbm.fit(X_train,y_train)
y_pre= lgbm.predict(X_test)
y_pro= lgbm.predict_proba(X_test)[:,1]
print("lightgbm model Test AUC Score : {0}", metrics.roc_auc_score(y_test, y_pro))
print("lightgbm model Test Accuracy : {0}" , metrics.accuracy_score(y_test, y_pre))


# LR Test AUC Score : {0} 0.495476338626
# LR  Test Accuracy : {0} 0.525
# init gbdt model:30
# init lr model
# pred_leaf=T  AUC Score : 0.96882606182305
# pred_leaf=T  Accuracy : 0.8947916666666667
# orginfeatures:(7680, 10),predleaves:(7680, 224)
# orginfeatures:(1920, 10),predleaves:(1920, 224)
# Training set sample number remains the same
# GBDT+LR Training AUC Score : 0.977231436568786
# GBDT+LR  Training Accuracy : 0.9114583333333334
# orginfeatures:(2400, 10),predleaves:(2400, 224)
# orginfeatures:(2400, 10),predleaves:(2400, 224)
# GBDT+LR Test AUC Score : {0} 0.974602495583
# GBDT+LR  Test Accuracy : {0} 0.9075
# Xgboost model Test AUC Score : {0} 0.96766214069
# Xgboost model Test Accuracy : {0} 0.892083333333
# lightgbm model Test AUC Score : {0} 0.971692046578
# lightgbm model Test Accuracy : {0} 0.896666666667
