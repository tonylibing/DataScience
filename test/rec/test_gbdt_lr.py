from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.datasets  import  make_hastie_10_2
from processor import *

X, y = make_hastie_10_2(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999)


lr = LogisticRegression(C=1.0, penalty='l2', tol=1e-4,solver='liblinear',random_state=42)
lr.fit(X_train,y_train)
y_pre= lr.predict(X_test)
y_pro= lr.predict_proba(X_test)[:,1]
print("="*60)
print("LR Test AUC Score : {0}".format(metrics.roc_auc_score(y_test, y_pro)))
print("LR  Test Accuracy : {0}".format(metrics.accuracy_score(y_test, y_pre)))
print("="*60)

gbm = xgb.XGBClassifier(n_estimators=30,learning_rate =0.3,max_depth=3,min_child_weight=1,gamma=0.3,subsample=0.7,colsample_bytree=0.7,objective= 'binary:logistic',nthread=-1,scale_pos_weight=1,reg_alpha=1e-05,reg_lambda=1,seed=27)
gbm.fit(X_train,y_train)
y_pre= gbm.predict(X_test)
# y_pre_leaf = gbm.predict(X_test,pred_leaf=True)
# print(y_pre_leaf.shape)
y_pro= gbm.predict_proba(X_test)[:,1]
print("="*60)
print("Xgboost model Test AUC Score: {0}".format(metrics.roc_auc_score(y_test, y_pro)))
print("Xgboost model Test Accuracy : {0}".format(metrics.accuracy_score(y_test, y_pre)))
print("="*60)

lgbm = lgb.LGBMClassifier(boosting_type='gbdt',  max_depth=3, learning_rate=0.3, n_estimators=30, min_child_weight=1,subsample=0.7,  colsample_bytree=0.7, reg_alpha=1e-05, reg_lambda=1)
lgbm.fit(X_train,y_train)
y_pre= lgbm.predict(X_test)
y_pro= lgbm.predict_proba(X_test)[:,1]
print("="*60)
print("lightgbm model Test AUC Score: {0}".format(metrics.roc_auc_score(y_test, y_pro)))
print("lightgbm model Test Accuracy : {0}".format(metrics.accuracy_score(y_test, y_pre)))
print("="*60)

gbdtlr = XgboostLRClassifier()
gbdtlr.fit(X_train,y_train)
y_pre= gbdtlr.predict(X_test)
y_pro= gbdtlr.predict_proba(X_test)[:,1]
print("="*60)
print("Xgboost+LR Test AUC Score : {0}".format(metrics.roc_auc_score(y_test, y_pro)))
print("Xgboost+LR  Test Accuracy : {0}".format(metrics.accuracy_score(y_test, y_pre)))
print("="*60)

lgbmlr = LightgbmLRClassifier()
lgbmlr.fit(X_train,y_train)
y_pre= lgbmlr.predict(X_test)
y_pro= lgbmlr.predict_proba(X_test)[:,1]
print("="*60)
print("Lightgbm+LR Test AUC Score : {0}".format(metrics.roc_auc_score(y_test, y_pro)))
print("Lightgbm+LR  Test Accuracy : {0}".format(metrics.accuracy_score(y_test, y_pre)))
print("="*60)
#
# gbdtlr = XgboostLRClassifier(combine_feature=False)
# gbdtlr.fit(X_train,y_train)
# y_pre= gbdtlr.predict(X_test)
# y_pro= gbdtlr.predict_proba(X_test)[:,1]
# print("="*60)
# print("Xgboost+LR Test AUC Score : {0}".format(metrics.roc_auc_score(y_test, y_pro)))
# print("Xgboost+LR  Test Accuracy : {0}".format(metrics.accuracy_score(y_test, y_pre)))
# print("="*60)
#
# lgbmlr = LightgbmLRClassifier(combine_feature=False)
# lgbmlr.fit(X_train,y_train)
# y_pre= lgbmlr.predict(X_test)
# y_pro= lgbmlr.predict_proba(X_test)[:,1]
# print("="*60)
# print("Lightgbm+LR Test AUC Score : {0}".format(metrics.roc_auc_score(y_test, y_pro)))
# print("Lightgbm+LR  Test Accuracy : {0}".format(metrics.accuracy_score(y_test, y_pre)))
# print("="*60)
#
