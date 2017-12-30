import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import confusion_matrix, recall_score, precision_score,f1_score, roc_auc_score,accuracy_score,average_precision_score,roc_curve,auc,precision_recall_curve

def ks_statistic(Y,Y_hat):
    data = {"Y":Y,"Y_hat":Y_hat}
    df = pd.DataFrame(data)
    bins = np.array([-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    category = pd.cut(df["Y_hat"],bins=bins)
    category = category.sort_values()
    #max_index = len(np.unique(df["Y_hat"]))
    Y = df.ix[category.index,:]['Y']
    Y_hat = df.ix[category.index,:]['Y_hat']
    df2 = pd.concat([Y,Y_hat],axis=1)
    df3 = pd.pivot_table(df2,values = ['Y_hat'],index ='Y_hat',columns='Y',aggfunc=len,fill_value=0)
    df4 = np.cumsum(df3)
    df5 = df4/df4.iloc[:,1].max()
    ks = max(abs(df5.iloc[:,0] - df5.iloc[:,1]))
    return ks/len(bins)

def plot_auc(y_test,y_score):
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)  # compute area under the curve

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    # create the axis of thresholds (scores)
    ax2 = plt.gca().twinx()
    ax2.plot(fpr, thresholds, markeredgecolor='r', linestyle='dashed', color='r')
    ax2.set_ylabel('Threshold', color='r')
    ax2.set_ylim([thresholds[-1], thresholds[0]])
    ax2.set_xlim([fpr[0], fpr[-1]])

    plt.savefig('roc_and_threshold.png')
    plt.close()

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plot_multi_auc(y_tests,y_scores,model_names):
    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','green','yellow'])
    for i,c in zip(range(len(y_tests)),colors):
        fpr, tpr, thresholds = roc_curve(y_tests[i], y_scores[i])
        roc_auc = auc(fpr, tpr)  # compute area under the curve

        plt.plot(fpr, tpr, label="%s auc= %0.4f" % (model_names[i],roc_auc),color=c)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig('roc.png')
    plt.show()
    plt.close()

def plot_multi_auc_thresholds(y_tests,y_scores,model_names):
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','green','yellow'])
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    ax2 = plt.gca().twinx()
    ax2.set_ylabel('Threshold', color='r')
    ax2.set_ylim([0.0,1.0])
    ax2.set_xlim([0.0,1.0])
    for i,c in zip(range(len(y_tests)),colors):
        fpr, tpr, thresholds = roc_curve(y_tests[i], y_scores[i])
        roc_auc = auc(fpr, tpr)  # compute area under the curve

        plt.plot(fpr, tpr, label="%s auc= %0.4f" % (model_names[i],roc_auc),color=c)
        # create the axis of thresholds (scores)
        ax2.plot(fpr, thresholds, markeredgecolor=c, linestyle='dashed', color=c)


    plt.show()
    plt.savefig('roc_and_threshold.png')
    plt.close()

def plot_precision_recall_curve(y_test,y_score):
    average_precision = average_precision_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))
    plt.savefig('precision_recall_curve.png')
    plt.show()
    plt.close()



def plot_prec_recall_vs_tresh(y_test,y_score):
    precisions, recalls, thresholds = precision_recall_curve(y_test,y_score)
    plt.plot(thresholds, precisions[:-1], 'b--', label='precision')
    plt.plot(thresholds, recalls[:-1], 'g--', label = 'recall')
    plt.xlabel('Threshold')
    plt.legend(loc='upper left')
    plt.ylim([0,1])
    plt.savefig('plot_prec_recall_vs_tresh.png')
    plt.show()
    plt.close()

def plot_prec_recall_f1_vs_tresh(y_test,y_score):
    precisions, recalls, thresholds = precision_recall_curve(y_test,y_score)
    plt.plot(thresholds, precisions[:-1], 'b--', label='precision')
    plt.plot(thresholds, recalls[:-1], 'g--', label = 'recall')
    f1 = f1_score(precisions,recalls)
    plt.plot(thresholds, f1, 'g--', label = 'F1')
    plt.xlabel('Threshold')
    plt.legend(loc='upper left')
    plt.ylim([0,1])
    plt.savefig('plot_prec_recall_vs_tresh.png')
    plt.show()
    plt.close()