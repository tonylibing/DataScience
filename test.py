import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from mlxtend.sklearn import EnsembleClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

y = pd.read_csv('y.csv').pop('SLIDE')
lr = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')
model = lr.fit(x, y)
score = model.score(X, y)
