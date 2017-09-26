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
