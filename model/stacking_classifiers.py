import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
import xgboost as xgb
# 
# train = pd.read_csv('../input/train.csv')
# test = pd.read_csv('../input/test.csv')
# 
# y_train = train['y'].values
# y_mean = np.mean(y_train)
# id_test = test['ID']
# 
# num_train = len(train)
# df_all = pd.concat([train, test])
# df_all.drop(['ID', 'y'], axis=1, inplace=True)
# 
# df_all = pd.get_dummies(df_all, drop_first=True)
# 
# train = df_all[:num_train]
# test = df_all[num_train:]
class GBDTLRClassifier(BaseEstimator, TransformerMixin):
    """
    gbdt+lr,sklearn-clf style
    """

    def __init__(self, gbdt_nrounds=50, gbdt_learning_rate=0.01, gbdt_seed=1580, gbdt_max_depth=6,
                 gbdt_min_child_weight=11, gbdt_subsample=0.7, gbdt_colsample_bytree=0.7,
                 lr_penalty='l2', lr_c=1.0, lr_random_state=42,
                 gbdt_on=True, dump_path="gbdt_node.txt", cat_cols=[]):
        """
        #xgboost parameters
        self._gbdt_nrounds = gbdt_nrounds
        self._gbdt_learning_rate = gbdt_learning_rate
        self._gbdt_max_depth = gbdt_max_depth
        self._gbdt_min_child_weight = gbdt_min_child_weight
        self._gbdt_subsample = gbdt_subsample
        self._gbdt_colsample_bytree = gbdt_colsample_bytree
        self._gbdt_seed = gbdt_seed
        #lr parameters
        self._lr_penalty = lr_penalty
        self._lr_c = lr_c
        self._lr_random_state = lr_random_state
        """
        self.__xgb_model = xgb.XGBClassifier(n_estimators=gbdt_nrounds, max_depth=gbdt_max_depth, seed=gbdt_seed,
                                             learning_rate=gbdt_learning_rate, subsample=gbdt_subsample,
                                             colsample_bytree=gbdt_colsample_bytree)
        # learning_rate = gbdt_learning_rate,\
        # min_child_weight=gbdt_min_child_weight, subsample=gbdt_subsample, \
        # colsample_bytree=gbdt_colsample_bytree
        self.__lr_model = LogisticRegression(C=lr_c, penalty=lr_penalty, random_state=lr_random_state)

        # path related,and tag to switch xgboost on/off
        self.__gbdt_on = gbdt_on
        self.__dump_path = dump_path
        self.__cat_cols = cat_cols

        # linear coef
        self.__coef = []

        # values dict for data transform
        self.__feature_dict = {}

        self.__SYS_MIN = -999999
        self.__SYS_MAX = 999999

    def transform(self, data):
        # feature encoder
        da = pd.DataFrame(data, copy=True)
        origin_cols = list(set(list(data.columns)) - set(self.__cat_cols))
        for feature, value_set in self.__feature_dict.items():
            if feature in self.__cat_cols:
                # skip catogory columns
                continue
            else:
                pass
            # create two inf of the value_list
            value_list = sorted(list(value_set))
            # min1 = value_list[0]
            # max1 = value_list[-1]
            min0 = self.__SYS_MIN
            max0 = self.__SYS_MAX
            value_list.insert(0, min0)
            value_list.insert(len(value_list), max0)
            for i, value in enumerate(value_list):
                # no need for the last
                if len(value_list) == i + 1:
                    break
                # rule: right area of the value
                low_bound = value
                high_bound = value_list[i + 1]
                low_bound_name = str(low_bound) if low_bound != self.__SYS_MIN else "MIN"
                high_bound_name = str(high_bound) if high_bound != self.__SYS_MAX  else "MAX"
                col = "%s<=%s<%s" % (low_bound_name, feature, high_bound_name)  # name the col
                da[col] = da[feature].apply(lambda x: 1 if x >= low_bound and x < high_bound else 0)

        # remove original feature
        da = da.drop(origin_cols, axis=1)
        return da

    def fit_transform(self, data, y=None):
        if self.__gbdt_on:
            self.__xgb_model.fit(data, y)
            self.__xgb_model.booster().dump_model(self.__dump_path)

            f = open(self.__dump_path, 'r')
            feature_dict = {}

            for line in f.readlines():
                if '<' in line:  # feature line
                    line = line.split(':')[1].strip()
                    feature_re = re.match('\[(.*)?\]', line)
                    info = feature_re.group(0)  # should be only one group
                    info = re.sub('\[|\]', '', info)
                    feature = info.split('<')[0].strip()
                    value = float(info.split('<')[1].strip())
                    value_set = feature_dict[feature] if feature in feature_dict else set()
                    value_set.add(value)
                    feature_dict[feature] = value_set

            self.__feature_dict = feature_dict

        result_data = self.transform(data)
        return result_data

    def fit(self, data, y=None):
        global IMPT
        da = data  # pd.DataFrame(data, copy=True)
        for i in list(da.columns):
            self.__SYS_MIN = self.__SYS_MIN if self.__SYS_MIN < da[i].min() else da[i].min() - 1
            self.__SYS_MAX = self.__SYS_MAX if self.__SYS_MAX > da[i].max() else da[i].max() + 1

        da = self.fit_transform(da, y)
        self.__lr_model.fit(da, y)
        self.__coef = self.__lr_model.coef_[0]
        result = sorted(zip(map(lambda x: round(x, IMPT), self.__coef), list(da.columns)), reverse=True)
        self.save_coef("gbdt_coef_analysis.txt", result)
        # plot_coef("coef_analysis.png", result)
        return self

    def predict(self, test):
        test1 = self.transform(test)
        return self.__lr_model.predict(test1)

    def predict_proba(self, test):
        test1 = self.transform(test)
        return self.__lr_model.predict_proba(test1)

    # def print and plot
    def save_coef(self, path, result):
        o = open(path, 'w')
        o.write("len of feature: %s\n" % len(result))
        for (value, key) in result:
            if 0 == value:
                pass
            else:
                writeln = "%s: %.3f\n" % (key, value)
                o.write(writeln)
        o.close()

class MajorityVoteClassifier(BaseEstimator, 
                             ClassifierMixin):
    """ A majority vote ensemble classifier
    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]
      Different classifiers for the ensemble
    vote : str, {'classlabel', 'probability'} (default='label')
      If 'classlabel' the prediction is based on the argmax of
        class labels. Else if 'probability', the argmax of
        the sum of probabilities is used to predict the class label
        (recommended for calibrated classifiers).
    weights : array-like, shape = [n_classifiers], optional (default=None)
      If a list of `int` or `float` values are provided, the classifiers
      are weighted by importance; Uses uniform weights if `weights=None`.
    """
    def __init__(self, classifiers, vote='classlabel', weights=None):

        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value
                                  in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        """ Fit classifiers.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of training samples.
        y : array-like, shape = [n_samples]
            Vector of target class labels.
        Returns
        -------
        self : object
        """
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError("vote must be 'probability' or 'classlabel'"
                             "; got (vote=%r)"
                             % self.vote)

        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d classifiers'
                             % (len(self.weights), len(self.classifiers)))

        # Use LabelEncoder to ensure class labels start with 0, which
        # is important for np.argmax call in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        """ Predict class labels for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of training samples.
        Returns
        ----------
        maj_vote : array-like, shape = [n_samples]
            Predicted class labels.
            
        """
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:  # 'classlabel' vote

            #  Collect results from clf.predict calls
            predictions = np.asarray([clf.predict(X)
                                      for clf in self.classifiers_]).T

            maj_vote = np.apply_along_axis(
                                      lambda x:
                                      np.argmax(np.bincount(x,
                                                weights=self.weights)),
                                      axis=1,
                                      arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        """ Predict class probabilities for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        ----------
        avg_proba : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        probas = np.asarray([clf.predict_proba(X)
                             for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        """ Get classifier parameter names for GridSearch"""
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out


"""
meta_classifier:top layer classifier
"""
class StackingCVClassifierAveraged(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, classifiers, meta_classifier, n_folds=5):
        self.classifiers = classifiers
        self.meta_classifier = meta_classifier
        self.n_folds = n_folds

    def fit(self, X, y):
        self.clr_ = [list() for x in self.classifiers]
        self.meta_clr_ = clone(self.meta_classifier)

        kfold = KFold(n_splits=self.n_folds, shuffle=True)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.classifiers)))

        for i, clf in enumerate(self.classifiers):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(clf)
                self.clr_[i].append(instance)

                instance.fit(X[train_idx], y[train_idx])
                y_pred = instance.predict(X[holdout_idx])
                out_of_fold_predictions[holdout_idx, i] = y_pred

        self.meta_clr_.fit(out_of_fold_predictions, y)

        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([r.predict(X) for r in clrs]).mean(axis=1)
            for clrs in self.clr_
        ])
        return self.meta_clr_.predict(meta_features)
        

class StackingCVClassifierRetrained(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, classifiers, meta_classifier, n_folds=5, use_features_in_secondary=False):
        self.classifiers = classifiers
        self.meta_classifier = meta_classifier
        self.n_folds = n_folds
        self.use_features_in_secondary = use_features_in_secondary

    def fit(self, X, y):
        self.clr_ = [clone(x) for x in self.classifiers]
        self.meta_clr_ = clone(self.meta_classifier)

        kfold = KFold(n_splits=self.n_folds, shuffle=True)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.classifiers)))

        # Create out-of-fold predictions for training meta-model
        for i, clr in enumerate(self.clr_):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(clr)
                instance.fit(X[train_idx], y[train_idx])
                out_of_fold_predictions[holdout_idx, i] = instance.predict(X[holdout_idx])

        # Train meta-model
        if self.use_features_in_secondary:
            self.meta_clr_.fit(np.hstack((X, out_of_fold_predictions)), y)
        else:
            self.meta_clr_.fit(out_of_fold_predictions, y)
        
        # Retrain base models on all data
        for clr in self.clr_:
            clr.fit(X, y)

        return self

    def predict(self, X):
        meta_features = np.column_stack([
            clr.predict(X) for clr in self.clr_
        ])

        if self.use_features_in_secondary:
            return self.meta_clr_.predict(np.hstack((X, meta_features)))
        else:
            return self.meta_clr_.predict(meta_features)
        
class AveragingClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def fit(self, X, y):
        self.clr_ = [clone(x) for x in self.classifiers]
        
        # Train base models
        for clr in self.clr_:
            clr.fit(X, y)

        return self

    def predict(self, X):
        predictions = np.column_stack([
            clr.predict(X) for clr in self.clr_
        ])
        return np.mean(predictions, axis=1)

# en = make_pipeline(RobustScaler(), SelectFromModel(Lasso(alpha=0.03)), ElasticNet(alpha=0.001, l1_ratio=0.1))
#     
# rf = RandomForestClassifier(n_estimators=250, n_jobs=4, min_samples_split=25, min_samples_leaf=25, max_depth=3)
#                            
# et = ExtraTreesClassifier(n_estimators=100, n_jobs=4, min_samples_split=25, min_samples_leaf=35, max_features=150)
# 
# xgbm = xgb.sklearn.XGBClassifier(max_depth=4, learning_rate=0.005, subsample=0.9, base_score=y_mean,
#                                 objective='reg:linear', n_estimators=1000)
#                            
# stack_avg = StackingCVClassifierAveraged((en, rf, et), ElasticNet(l1_ratio=0.1, alpha=1.4))
# 
# stack_with_feats = StackingCVClassifierRetrained((en, rf, et), xgbm, use_features_in_secondary=True)
# 
# stack_retrain = StackingCVClassifierRetrained((en, rf, et), ElasticNet(l1_ratio=0.1, alpha=1.4))
# 
# averaged = AveragingClassifier((en, rf, et, xgbm))
# 
# results = cross_val_score(en, train.values, y_train, cv=5, scoring='r2')
# print("ElasticNet score: %.4f (%.4f)" % (results.mean(), results.std()))
# 
# results = cross_val_score(rf, train.values, y_train, cv=5, scoring='r2')
# print("RandomForest score: %.4f (%.4f)" % (results.mean(), results.std()))
# 
# results = cross_val_score(et, train.values, y_train, cv=5, scoring='r2')
# print("ExtraTrees score: %.4f (%.4f)" % (results.mean(), results.std()))
# 
# results = cross_val_score(xgbm, train.values, y_train, cv=5, scoring='r2')
# print("XGBoost score: %.4f (%.4f)" % (results.mean(), results.std()))
# 
# results = cross_val_score(averaged, train.values, y_train, cv=5, scoring='r2')
# print("Averaged base models score: %.4f (%.4f)" % (results.mean(), results.std()))
# 
# results = cross_val_score(stack_with_feats, train.values, y_train, cv=5, scoring='r2')
# print("Stacking (with primary feats) score: %.4f (%.4f)" % (results.mean(), results.std()))
# 
# results = cross_val_score(stack_retrain, train.values, y_train, cv=5, scoring='r2')
# print("Stacking (retrained) score: %.4f (%.4f)" % (results.mean(), results.std()))
#                  
# results = cross_val_score(stack_avg, train.values, y_train, cv=5, scoring='r2')
# print("Stacking (averaged) score: %.4f (%.4f)" % (results.mean(), results.std()))
