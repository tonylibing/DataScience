import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix
from wordbatch.models import FTRL, FM_FTRL

class XgboostLRClassifier(BaseEstimator):
    def __init__(self,combine_feature = False,n_estimators=30,learning_rate =0.3,max_depth=3,min_child_weight=1,gamma=0.3,subsample=0.7,colsample_bytree=0.7,objective= 'binary:logistic',nthread=-1,scale_pos_weight=1,reg_alpha=1e-05,reg_lambda=1,seed=27,lr_penalty='l2', lr_c=1.0, lr_random_state=42):
        self.combine_feature = combine_feature
        #gbdt model parameters
        self.n_estimators=n_estimators
        self.learning_rate=learning_rate
        self.max_depth=max_depth
        self.min_child_weight=min_child_weight
        self.gamma=gamma
        self.subsample=subsample
        self.colsample_bytree=colsample_bytree
        self.objective=objective
        self.nthread=nthread
        self.scale_pos_weight=scale_pos_weight
        self.reg_alpha=reg_alpha
        self.reg_lambda=reg_lambda
        self.seed=seed
        print("init gbdt model:{0}".format(n_estimators))
        self.gbdt_model = xgb.XGBClassifier(
               learning_rate =self.learning_rate,
               n_estimators=self.n_estimators,
               max_depth=self.max_depth,
               min_child_weight=self.min_child_weight,
               gamma=self.gamma,
               subsample=self.subsample,
               colsample_bytree=self.colsample_bytree,
               objective= self.objective,
               nthread=self.nthread,
               scale_pos_weight=self.scale_pos_weight,
               reg_alpha=self.reg_alpha,
               reg_lambda=self.reg_lambda,
               seed=self.seed)
        #lr model parameters
        self.lr_penalty = lr_penalty
        self.lr_c = lr_c
        self.lr_random_state = lr_random_state
        print("init lr model")
        self.fm_ftrl_model = FM_FTRL(alpha=0.012, beta=0.01, L1=0.00001, L2=0.1, alpha_fm=0.01, L2_fm=0.0, init_fm=0.01, D_fm=20, e_noise=0.0001, iters=3000, inv_link="identity", threads=4)
        #numerical feature binner
        self.one_hot_encoder = OneHotEncoder()
        self.numerical_feature_processor = None

    def gen_gbdt_features(self,pred_leaves,num_leaves=None):
        if num_leaves is None:
            num_leaves = np.amax(pred_leaves)

        # gbdt_feature_matrix = self.one_hot_encoder.fit_transform(pred_leaves)
        # return gbdt_feature_matrix
        gbdt_feature_matrix = np.zeros([len(pred_leaves), len(pred_leaves[0]) * num_leaves], dtype=np.int64)
        for i in range(0, len(pred_leaves)):
            temp = np.arange(len(pred_leaves[0])) * num_leaves - 1 + np.array(pred_leaves[i])
            gbdt_feature_matrix[i][temp] += 1

        print("pred_leaves:{0},gbdt_feature_matrix:{1},num_leaves:{2}".format(pred_leaves.shape,gbdt_feature_matrix.shape,num_leaves))

        return gbdt_feature_matrix

    def gen_gbdt_lr_features(self, origin_features, pred_leaves, num_leaves=None):
        if num_leaves is None:
            num_leaves = np.amax(pred_leaves)

        # gbdt_feature_matrix = self.one_hot_encoder.fit_transform(pred_leaves)
        # print("onehotencoder active_features:".format(self.one_hot_encoder.active_features_))

        gbdt_feature_matrix = np.zeros([len(pred_leaves), len(pred_leaves[0]) * num_leaves], dtype=np.int64)
        for i in range(0, len(pred_leaves)):
            temp = np.arange(len(pred_leaves[0])) * num_leaves - 1 + np.array(pred_leaves[i])
            gbdt_feature_matrix[i][temp] += 1

        print("orgin_features:{0},pred_leaves:{1},gbdt_feature_matrix:{2},num_leaves:{3}".format(origin_features.shape,
                                                                                                 pred_leaves.shape,
                                                                                                 gbdt_feature_matrix.shape,
                                                                                                 num_leaves))
        # print("orgin_features:{0},pred_leaves:{1}".format(type(origin_features),type(gbdt_feature_matrix)))
        if isinstance(origin_features, csr_matrix) and isinstance(gbdt_feature_matrix, csr_matrix):
            gbdt_lr_feature_matrix = sparse.hstack((origin_features, gbdt_feature_matrix), format='csr')
            # gbdt_lr_feature_matrix = np.concatenate((origin_features,gbdt_feature_matrix),axis=1)
        elif isinstance(origin_features, csr_matrix) and isinstance(gbdt_feature_matrix, np.ndarray):
            gbdt_lr_feature_matrix = sparse.hstack((origin_features, csr_matrix(gbdt_feature_matrix)), format='csr')
        elif isinstance(origin_features, np.ndarray) and isinstance(gbdt_feature_matrix, np.ndarray):
            gbdt_lr_feature_matrix = np.concatenate((origin_features, gbdt_feature_matrix), axis=1)
        return gbdt_lr_feature_matrix

    def fit_model_split(self, X_train, y_train, X_test, y_test):
        ##X_train_1用于生成模型  X_train_2用于和新特征组成新训练集合
        X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, test_size=0.6, random_state=999)
        self.gbdt_model.fit(X_train_1, y_train_1)
        y_pre = self.gbdt_model.predict(X_train_2)
        y_pro = self.gbdt_model.predict_proba(X_train_2)[:, 1]
        print("pred_leaf=T AUC Score :{0}".format(metrics.roc_auc_score(y_train_2, y_pro)))
        print("pred_leaf=T  Accuracy : {0}".format(metrics.accuracy_score(y_train_2, y_pre)))
        new_feature = self.gbdt_model.apply(X_train_2)
        X_train_new2 = self.gen_gbdt_lr_features(X_train_2, new_feature) if self.combine_feature else self.gen_gbdt_features(new_feature)
        new_feature_test = self.gbdt_model.apply(X_test)
        X_test_new = self.gen_gbdt_lr_features(X_test, new_feature_test) if self.combine_feature else self.gen_gbdt_features(new_feature_test)
        print("Training set of sample size 0.4 fewer than before")
        return X_train_new2, y_train_2, X_test_new, y_test

    def fit_model(self, X_train, y_train, X_test, y_test):
        self.gbdt_model.fit(X_train, y_train)
        y_pre = self.gbdt_model.predict(X_test)
        y_pro = self.gbdt_model.predict_proba(X_test)[:, 1]
        print("pred_leaf=T  AUC Score: {0}".format(metrics.roc_auc_score(y_test, y_pro)))
        print("pred_leaf=T  Accuracy : {0}".format(metrics.accuracy_score(y_test, y_pre)))
        new_feature = self.gbdt_model.apply(X_train)
        X_train_new = self.gen_gbdt_lr_features(X_train, new_feature) if self.combine_feature else self.gen_gbdt_features(new_feature)
        new_feature_test = self.gbdt_model.apply(X_test)
        X_test_new = self.gen_gbdt_lr_features(X_test, new_feature_test) if self.combine_feature else self.gen_gbdt_features(new_feature_test)
        print("Training set sample number remains the same")
        return X_train_new, y_train, X_test_new, y_test

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999)
        # generate new feature with partial data
        X_train2, y_train2, X_test2, y_test2 = self.fit_model(X_train, y_train, X_test, y_test)
        self.fm_ftrl_model.fit(X_train2, y_train2)
        y_pre = self.fm_ftrl_model.predict(X_test2)
        y_pro = self.fm_ftrl_model.predict_proba(X_test2)[:, 1]
        print("Xgboost+LR Training AUC Score : {0}".format(metrics.roc_auc_score(y_test2, y_pro)))
        print("Xgboost+LR  Training Accuracy : {0}".format(metrics.accuracy_score(y_test2, y_pre)))
        return self

    def transform(self, X):
        new_feature_test = self.gbdt_model.apply(X)
        X_test_new = self.gen_gbdt_lr_features(X, new_feature_test) if self.combine_feature else self.gen_gbdt_features(new_feature_test)
        return X_test_new

    def predict(self, X):
        test1 = self.transform(X)
        return self.fm_ftrl_model.predict(test1)

    def predict_proba(self, X):
        test1 = self.transform(X)
        return self.fm_ftrl_model.predict_proba(test1)

class LightgbmLRClassifier(BaseEstimator):
    def __init__(self, combine_feature = False,n_estimators=30, learning_rate=0.3, max_depth=3, min_child_weight=1, gamma=0.3,
                 subsample=0.7, colsample_bytree=0.7, objective='binary:logistic', nthread=-1, scale_pos_weight=1,
                 reg_alpha=1e-05, reg_lambda=1, seed=27, lr_penalty='l2', lr_c=1.0, lr_random_state=42):
        self.combine_feature = combine_feature
        # gbdt model parameters
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.objective = objective
        self.nthread = nthread
        self.scale_pos_weight = scale_pos_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.seed = seed
        print("init gbdt model:{0}".format(n_estimators))
        # self.gbdt_model = lgb.LGBMClassifier(
        #     learning_rate=self.learning_rate,
        #     n_estimators=self.n_estimators,
        #     max_depth=self.max_depth,
        #     min_child_weight=self.min_child_weight,
        #     gamma=self.gamma,
        #     subsample=self.subsample,
        #     colsample_bytree=self.colsample_bytree,
        #     objective=self.objective,
        #     nthread=self.nthread,
        #     scale_pos_weight=self.scale_pos_weight,
        #     reg_alpha=self.reg_alpha,
        #     reg_lambda=self.reg_lambda,
        #     seed=self.seed)
        self.gbdt_model = lgb.LGBMClassifier(boosting_type='gbdt',  max_depth=3, learning_rate=0.3, n_estimators=30, min_child_weight=1,subsample=0.7,  colsample_bytree=0.7, reg_alpha=1e-05, reg_lambda=1)
        # lr model parameters
        self.lr_penalty = lr_penalty
        self.lr_c = lr_c
        self.lr_random_state = lr_random_state
        print("init lr model")
        self.fm_ftrl_model = FM_FTRL(alpha=0.012, beta=0.01, L1=0.00001, L2=0.1, alpha_fm=0.01, L2_fm=0.0, init_fm=0.01, D_fm=20, e_noise=0.0001, iters=3000, inv_link="identity", threads=4)
        # numerical feature binner
        self.one_hot_encoder = OneHotEncoder()
        self.numerical_feature_processor = None

    def gen_gbdt_features(self, pred_leaves, num_leaves=None):
        if num_leaves is None:
            num_leaves = np.amax(pred_leaves)

        # gbdt_feature_matrix = self.one_hot_encoder.fit_transform(pred_leaves)
        # return gbdt_feature_matrix
        gbdt_feature_matrix = np.zeros([len(pred_leaves), len(pred_leaves[0]) * num_leaves], dtype=np.int64)
        for i in range(0, len(pred_leaves)):
            temp = np.arange(len(pred_leaves[0])) * num_leaves - 1 + np.array(pred_leaves[i])
            gbdt_feature_matrix[i][temp] += 1

        print(
        "pred_leaves:{0},gbdt_feature_matrix:{1},num_leaves:{2}".format(pred_leaves.shape, gbdt_feature_matrix.shape,
                                                                        num_leaves))
        return gbdt_feature_matrix

    def gen_gbdt_lr_features(self, origin_features, pred_leaves, num_leaves=None):
        if num_leaves is None:
            num_leaves = np.amax(pred_leaves)

        # gbdt_feature_matrix = self.one_hot_encoder.fit_transform(pred_leaves)
        # print("onehotencoder active_features:".format(self.one_hot_encoder.active_features_))

        gbdt_feature_matrix = np.zeros([len(pred_leaves), len(pred_leaves[0]) * num_leaves], dtype=np.int64)
        for i in range(0, len(pred_leaves)):
            temp = np.arange(len(pred_leaves[0])) * num_leaves - 1 + np.array(pred_leaves[i])
            gbdt_feature_matrix[i][temp] += 1

        print("orgin_features:{0},pred_leaves:{1},gbdt_feature_matrix:{2},num_leaves:{3}".format(origin_features.shape,
                                                                                                 pred_leaves.shape,
                                                                                                 gbdt_feature_matrix.shape,
                                                                                                 num_leaves))
        # print("orgin_features:{0},pred_leaves:{1}".format(type(origin_features),type(gbdt_feature_matrix)))
        if isinstance(origin_features, csr_matrix) and isinstance(gbdt_feature_matrix, csr_matrix):
            gbdt_lr_feature_matrix = sparse.hstack((origin_features,gbdt_feature_matrix),format='csr')
            # gbdt_lr_feature_matrix = np.concatenate((origin_features,gbdt_feature_matrix),axis=1)
        elif isinstance(origin_features, csr_matrix) and isinstance(gbdt_feature_matrix, np.ndarray):
            gbdt_lr_feature_matrix = sparse.hstack((origin_features,csr_matrix(gbdt_feature_matrix)),format='csr')
        elif isinstance(origin_features, np.ndarray) and isinstance(gbdt_feature_matrix, np.ndarray):
            gbdt_lr_feature_matrix = np.concatenate((origin_features, gbdt_feature_matrix), axis=1)
        return gbdt_lr_feature_matrix

    def fit_model_split(self, X_train, y_train, X_test, y_test):
        ##X_train_1用于生成模型  X_train_2用于和新特征组成新训练集合
        X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, test_size=0.6, random_state=999)
        self.gbdt_model.fit(X_train_1, y_train_1)
        y_pre = self.gbdt_model.predict(X_train_2)
        y_pro = self.gbdt_model.predict_proba(X_train_2)[:, 1]
        print("pred_leaf=T AUC Score :{0}".format(metrics.roc_auc_score(y_train_2, y_pro)))
        print("pred_leaf=T  Accuracy : {0}".format(metrics.accuracy_score(y_train_2, y_pre)))
        new_feature = self.gbdt_model.apply(X_train_2)
        X_train_new2 = self.gen_gbdt_lr_features(X_train_2, new_feature) if self.combine_feature else self.gen_gbdt_features(new_feature)
        new_feature_test = self.gbdt_model.apply(X_test)
        X_test_new = self.gen_gbdt_lr_features(X_test, new_feature_test) if self.combine_feature else self.gen_gbdt_features(new_feature_test)
        print("Training set of sample size 0.4 fewer than before")
        return X_train_new2, y_train_2, X_test_new, y_test

    def fit_model(self, X_train, y_train, X_test, y_test):
        self.gbdt_model.fit(X_train, y_train)
        y_pre = self.gbdt_model.predict(X_test)
        y_pro = self.gbdt_model.predict_proba(X_test)[:, 1]
        print("pred_leaf=T  AUC Score: {0}".format(metrics.roc_auc_score(y_test, y_pro)))
        print("pred_leaf=T  Accuracy : {0}".format(metrics.accuracy_score(y_test, y_pre)))
        new_feature = self.gbdt_model.apply(X_train)
        X_train_new = self.gen_gbdt_lr_features(X_train, new_feature) if self.combine_feature else self.gen_gbdt_features(new_feature)
        new_feature_test = self.gbdt_model.apply(X_test)
        X_test_new = self.gen_gbdt_lr_features(X_test, new_feature_test) if self.combine_feature else self.gen_gbdt_features(new_feature_test)
        print("Training set sample number remains the same")
        return X_train_new, y_train, X_test_new, y_test

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999)
        # generate new feature with partial data
        X_train2, y_train2, X_test2, y_test2 = self.fit_model(X_train, y_train, X_test, y_test)
        self.fm_ftrl_model.fit(X_train2, y_train2)
        y_pre = self.fm_ftrl_model.predict(X_test2)
        y_pro = self.fm_ftrl_model.predict_proba(X_test2)[:, 1]
        print("Lightgbm+LR Training AUC Score : {0}".format(metrics.roc_auc_score(y_test2, y_pro)))
        print("Lightgbm+LR  Training Accuracy : {0}".format(metrics.accuracy_score(y_test2, y_pre)))
        return self

    def transform(self, X):
        new_feature_test = self.gbdt_model.apply(X)
        X_test_new = self.gen_gbdt_lr_features(X, new_feature_test) if self.combine_feature else self.gen_gbdt_features(new_feature_test)
        return X_test_new

    def predict(self, X):
        test1 = self.transform(X)
        return self.fm_ftrl_model.predict(test1)

    def predict_proba(self, X):
        test1 = self.transform(X)
        return self.fm_ftrl_model.predict_proba(test1)
