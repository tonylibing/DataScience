import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

train_x_raw = pd.read_csv("train_x.csv")
train_y_raw = pd.read_csv("train_y.csv")
test_x_raw = pd.read_csv("test.csv")

# change dataframe to matrix
train_x = np.array(train_x_raw,dtype='int32')
train_y = np.array(train_y_raw,dtype='int32').reshape(len(train_y_raw),)

features = np.array(train_x_raw.columns)

model = RandomForestClassifier(n_estimators=100,
                                random_state=0,
                                n_jobs=-1)
model.fit(train_x, train_y)

importances = model.feature_importances_

importances_df = DataFrame({"feature":features,
                            "importance":importances})
importances_df = importances_df.sort_values(["importance"],ascending=False)

indices = np.argsort(importances)[::-1]
## plot the importance

for f in range(train_x.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            features[indices[f]], 
                            importances[indices[f]]))
plt.title('Feature Importances')
plt.bar(range(train_x.shape[1]), 
        importances[indices],
        color='lightblue', 
        align='center')

plt.xticks(range(train_x.shape[1]), 
           features[indices], rotation=90)
plt.xlim([-1, train_x.shape[1]])
plt.tight_layout()
#plt.savefig('./random_forest.png', dpi=300)
plt.show()


## grid search
param_grid = {
              'n_estimators': [10, 100, 500, 1000],
              'max_features':[0.6, 0.7, 0.8, 0.9]
             }

rf = RandomForestClassifier()
rfc = GridSearchCV(rf, param_grid, scoring = 'neg_log_loss', cv=3, n_jobs=2)
rfc.fit(train_x,train_y)
print(rfc.best_score_)
print(rfc.best_params_)