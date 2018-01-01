import lightgbm as lgb
import numpy as np
import xgboost as xgb
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


class XgboostLRClassifier(BaseEstimator):
    def __init__(self, combine_feature=False, n_estimators=30, learning_rate=0.3, max_depth=4, min_child_weight=1,
                 gamma=0.3, subsample=0.7, colsample_bytree=0.7, objective='binary:logistic', nthread=-1,
                 scale_pos_weight=1, reg_alpha=1e-05, reg_lambda=1, seed=27, lr_penalty='l2', lr_c=1.0,
                 lr_random_state=42):
        self.combine_feature = combine_feature
        # self.scale = scale
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
        self.gbdt_model = xgb.XGBClassifier(
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective=self.objective,
            nthread=self.nthread,
            scale_pos_weight=self.scale_pos_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            seed=self.seed)
        # lr model parameters
        self.lr_penalty = lr_penalty
        self.lr_c = lr_c
        self.lr_random_state = lr_random_state
        print("init lr model")
        # self.lr_model = SGDClassifier(loss='log')
        self.lr_model = LogisticRegression(C=lr_c, penalty=lr_penalty, tol=1e-4, solver='liblinear',random_state=lr_random_state,class_weight='balanced')
        # numerical feature binner
        self.one_hot_encoder = OneHotEncoder()
        self.numerical_feature_processor = None
        self.numerical_cols = []
        self.categorical_cols = []
        # scaler
        self.scaler = StandardScaler()

    def feature_importance(self):
        return self.gbdt_model.feature_importances_

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
            "pred_leaves:{0},gbdt_feature_matrix:{1},num_leaves:{2}".format(pred_leaves.shape,
                                                                            gbdt_feature_matrix.shape,
                                                                            num_leaves))

        return gbdt_feature_matrix

    def gen_gbdt_lr_features(self, origin_features, pred_leaves, num_leaves=None):
        if num_leaves is None:
            num_leaves = np.amax(pred_leaves)
        # gbdt_feature_matrix = self.one_hot_encoder.fit_transform(pred_leaves)
        # print("onehotencoder active_features:".format(self.one_hot_encoder.active_features_))

        # if self.scale:
        #     df_numerical = origin_features[self.numerical_cols]
        #     df_norm = pd.DataFrame(self.scaler.transform(df_numerical), columns=self.numerical_cols)
        #     origin_features = pd.concat([df_norm,origin_features[self.categorical_cols]],axis=1)
        # if self.scale:
        #     if isinstance(origin_features,csr_matrix):
        #         origin_features = self.scaler.transform(origin_features,with_mean=False)
        #     else:
        #         origin_features = self.scaler.transform(origin_features)
        #

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
        X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, test_size=0.2, random_state=999,
                                                                      stratify=y_train)
        self.gbdt_model.fit(X_train_1, y_train_1)
        # print("feature importance:".format(self.gbdt_model.feature_importances_))
        print("feature importance:{0}".format(self.gbdt_model.feature_importances_.shape))
        print(self.gbdt_model.feature_importances_)
        # print("feature importance:".format(self.gbdt_model.booster().get_fscore()))
        y_pre = self.gbdt_model.predict(X_train_2)
        y_pro = self.gbdt_model.predict_proba(X_train_2)[:, 1]
        print("pred_leaf=T AUC Score :{0}".format(metrics.roc_auc_score(y_train_2, y_pro)))
        print("pred_leaf=T  Accuracy : {0}".format(metrics.accuracy_score(y_train_2, y_pre)))
        new_feature = self.gbdt_model.apply(X_train_2)
        X_train_new2 = self.gen_gbdt_lr_features(X_train_2,
                                                 new_feature) if self.combine_feature else self.gen_gbdt_features(
            new_feature)
        new_feature_test = self.gbdt_model.apply(X_test)
        X_test_new = self.gen_gbdt_lr_features(X_test,
                                               new_feature_test) if self.combine_feature else self.gen_gbdt_features(
            new_feature_test)
        print("Training set of sample size 0.4 fewer than before")
        return X_train_new2, y_train_2, X_test_new, y_test

    def fit_model(self, X_train, y_train, X_test, y_test):
        self.gbdt_model.fit(X_train, y_train)
        print("feature importance:{0}".format(self.gbdt_model.feature_importances_.shape))
        print(self.gbdt_model.feature_importances_)
        # print("feature importance:".format(self.gbdt_model.booster().get_fscore()))
        y_pre = self.gbdt_model.predict(X_test)
        y_pro = self.gbdt_model.predict_proba(X_test)[:, 1]
        print("pred_leaf=T  AUC Score: {0}".format(metrics.roc_auc_score(y_test, y_pro)))
        print("pred_leaf=T  Accuracy : {0}".format(metrics.accuracy_score(y_test, y_pre)))
        new_feature = self.gbdt_model.apply(X_train)
        X_train_new = self.gen_gbdt_lr_features(X_train,
                                                new_feature) if self.combine_feature else self.gen_gbdt_features(
            new_feature)
        new_feature_test = self.gbdt_model.apply(X_test)
        X_test_new = self.gen_gbdt_lr_features(X_test,
                                               new_feature_test) if self.combine_feature else self.gen_gbdt_features(
            new_feature_test)
        print("Training set sample number remains the same")
        return X_train_new, y_train, X_test_new, y_test

    def fit(self, X, y):
        # self.column_summary = ColumnSummary(X)
        # self.column_type = self.column_summary.set_index('col_name')['ColumnType'].to_dict()
        #
        # for col in X.columns.values:
        #     if self.column_type[col] == 'numerical':
        #         self.numerical_cols.append(col)
        #     elif self.column_type[col] == 'categorical':
        #         self.categorical_cols.append(col)
        # if self.scale:
        #     pass
        # if isinstance(X,csr_matrix):
        #     self.scaler.fit_transform(X[self.numerical_cols],with_mean=False)
        # else:
        #     self.scaler.fit_transform(X[self.numerical_cols])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=999, stratify=y)
        # generate new feature with partial data
        X_train2, y_train2, X_test2, y_test2 = self.fit_model(X_train, y_train, X_test, y_test)
        self.lr_model.fit(X_train2, y_train2)
        y_pre = self.lr_model.predict(X_test2)
        y_pro = self.lr_model.predict_proba(X_test2)[:, 1]
        print("Xgboost+LR Training AUC Score : {0}".format(metrics.roc_auc_score(y_test2, y_pro)))
        print("Xgboost+LR  Training Accuracy : {0}".format(metrics.accuracy_score(y_test2, y_pre)))
        return self

    def transform(self, X):
        new_feature_test = self.gbdt_model.apply(X)
        # normalize X numerical features
        # X_tmp = pd.DataFrame(self.scaler.fit_transform(X[self.numerical_cols]), columns=self.numerical_cols)
        X_test_new = self.gen_gbdt_lr_features(X,
                                               new_feature_test) if self.combine_feature else self.gen_gbdt_features(
            new_feature_test)
        return X_test_new

    def predict(self, X):
        test1 = self.transform(X)
        return self.lr_model.predict(test1)

    def predict_proba(self, X):
        test1 = self.transform(X)
        return self.lr_model.predict_proba(test1)


class LightgbmLRClassifier(BaseEstimator):
    def __init__(self, combine_feature=False, n_estimators=30, learning_rate=0.3, max_depth=4, min_child_weight=1,
                 gamma=0.3,
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
        self.gbdt_model = lgb.LGBMClassifier(boosting_type='gbdt', max_depth=4, learning_rate=0.3, n_estimators=30,
                                             scale_pos_weight=scale_pos_weight, min_child_weight=1, subsample=0.7,
                                             colsample_bytree=0.7, reg_alpha=1e-05, reg_lambda=1)
        # lr model parameters
        self.lr_penalty = lr_penalty
        self.lr_c = lr_c
        self.lr_random_state = lr_random_state
        print("init lr model")
        # self.lr_model = SGDClassifier(loss='log')
        self.lr_model = LogisticRegression(C=lr_c, penalty=lr_penalty, tol=1e-4, solver='liblinear',random_state=lr_random_state,class_weight='balanced')
        # numerical feature binner
        self.one_hot_encoder = OneHotEncoder()
        self.numerical_feature_processor = None

    def feature_importance(self):
        return self.gbdt_model.feature_importances_

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
            "pred_leaves:{0},gbdt_feature_matrix:{1},num_leaves:{2}".format(pred_leaves.shape,
                                                                            gbdt_feature_matrix.shape,
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
            gbdt_lr_feature_matrix = sparse.hstack((origin_features, gbdt_feature_matrix), format='csr')
            # gbdt_lr_feature_matrix = np.concatenate((origin_features,gbdt_feature_matrix),axis=1)
        elif isinstance(origin_features, csr_matrix) and isinstance(gbdt_feature_matrix, np.ndarray):
            gbdt_lr_feature_matrix = sparse.hstack((origin_features, csr_matrix(gbdt_feature_matrix)), format='csr')
        elif isinstance(origin_features, np.ndarray) and isinstance(gbdt_feature_matrix, np.ndarray):
            gbdt_lr_feature_matrix = np.concatenate((origin_features, gbdt_feature_matrix), axis=1)
        return gbdt_lr_feature_matrix

    def fit_model_split(self, X_train, y_train, X_test, y_test):
        ##X_train_1用于生成模型  X_train_2用于和新特征组成新训练集合
        X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, test_size=0.2, random_state=999,
                                                                      stratify=y_train)
        self.gbdt_model.fit(X_train_1, y_train_1)
        print("feature importance:{0}".format(self.gbdt_model.feature_importances_.shape))
        print(self.gbdt_model.feature_importances_)
        y_pre = self.gbdt_model.predict(X_train_2)
        y_pro = self.gbdt_model.predict_proba(X_train_2)[:, 1]
        print("pred_leaf=T AUC Score :{0}".format(metrics.roc_auc_score(y_train_2, y_pro)))
        print("pred_leaf=T  Accuracy : {0}".format(metrics.accuracy_score(y_train_2, y_pre)))
        new_feature = self.gbdt_model.apply(X_train_2)
        X_train_new2 = self.gen_gbdt_lr_features(X_train_2,
                                                 new_feature) if self.combine_feature else self.gen_gbdt_features(
            new_feature)
        new_feature_test = self.gbdt_model.apply(X_test)
        X_test_new = self.gen_gbdt_lr_features(X_test,
                                               new_feature_test) if self.combine_feature else self.gen_gbdt_features(
            new_feature_test)
        print("Training set of sample size 0.4 fewer than before")
        return X_train_new2, y_train_2, X_test_new, y_test

    def fit_model(self, X_train, y_train, X_test, y_test):
        self.gbdt_model.fit(X_train, y_train)
        print("feature importance:{0}".format(self.gbdt_model.feature_importances_.shape))
        print(self.gbdt_model.feature_importances_)
        y_pre = self.gbdt_model.predict(X_test)
        y_pro = self.gbdt_model.predict_proba(X_test)[:, 1]
        print("pred_leaf=T  AUC Score: {0}".format(metrics.roc_auc_score(y_test, y_pro)))
        print("pred_leaf=T  Accuracy : {0}".format(metrics.accuracy_score(y_test, y_pre)))
        new_feature = self.gbdt_model.apply(X_train)
        X_train_new = self.gen_gbdt_lr_features(X_train,
                                                new_feature) if self.combine_feature else self.gen_gbdt_features(
            new_feature)
        new_feature_test = self.gbdt_model.apply(X_test)
        X_test_new = self.gen_gbdt_lr_features(X_test,
                                               new_feature_test) if self.combine_feature else self.gen_gbdt_features(
            new_feature_test)
        print("Training set sample number remains the same")
        return X_train_new, y_train, X_test_new, y_test

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=999, stratify=y)
        # generate new feature with partial data
        X_train2, y_train2, X_test2, y_test2 = self.fit_model(X_train, y_train, X_test, y_test)
        self.lr_model.fit(X_train2, y_train2)
        y_pre = self.lr_model.predict(X_test2)
        y_pro = self.lr_model.predict_proba(X_test2)[:, 1]
        print("Lightgbm+LR Training AUC Score : {0}".format(metrics.roc_auc_score(y_test2, y_pro)))
        print("Lightgbm+LR  Training Accuracy : {0}".format(metrics.accuracy_score(y_test2, y_pre)))
        return self

    def transform(self, X):
        new_feature_test = self.gbdt_model.apply(X)
        X_test_new = self.gen_gbdt_lr_features(X, new_feature_test) if self.combine_feature else self.gen_gbdt_features(
            new_feature_test)
        return X_test_new

    def predict(self, X):
        test1 = self.transform(X)
        return self.lr_model.predict(test1)

    def predict_proba(self, X):
        test1 = self.transform(X)
        return self.lr_model.predict_proba(test1)
