import metrics
from importlib import reload
reload(metrics)
from metrics import *

y_tests = []
y_scores = []
model_names = []

# data=pd.read_csv("E:/dataset/lr_y_test_pro.csv",sep=',')
# y_pro=data['y_pro']
# y_test=data['y_test']
# y_tests.append(y_test)
# y_scores.append(y_pro)
# model_names.append('lr')


data=pd.read_csv("E:/dataset/xgb_y_test_pro.csv",sep=',')
y_pro=data['y_pro']
y_test=data['y_test']
y_tests.append(y_test)
y_scores.append(y_pro)
model_names.append('xgb')

data=pd.read_csv("E:/dataset/lgb_lr_no_combine_y_test_pro.csv",sep=',')
y_pro=data['y_pro']
y_test=data['y_test']
y_tests.append(y_test)
y_scores.append(y_pro)
model_names.append('lgb_lr')

plot_multi_auc(y_tests,y_scores,model_names)


#
# data=pd.read_csv("E:/dataset/lgb_lr_no_combine_y_test_pro.csv",sep=',')
# y_pro=data['y_pro']
# y_test=data['y_test']
# y_tests.append(y_test)
# y_scores.append(y_pro)
# model_names.append('lgb_lr')
#
# data=pd.read_csv("E:/dataset/lgb_lr_no_combine_y_test_pro.csv",sep=',')
# y_pro=data['y_pro']
# y_test=data['y_test']
# y_tests.append(y_test)
# y_scores.append(y_pro)
# model_names.append('lgb_lr')
#
# data=pd.read_csv("E:/dataset/lgb_lr_no_combine_y_test_pro.csv",sep=',')
# y_pro=data['y_pro']
# y_test=data['y_test']
# y_tests.append(y_test)
# y_scores.append(y_pro)
# model_names.append('lgb_lr')


