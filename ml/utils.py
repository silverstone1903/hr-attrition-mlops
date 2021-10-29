from sklearn.metrics import *
import numpy as np

def scorer(y_true, y_pred, is_return = False):
    if is_return:
        return [f1_score(y_true, np.round(y_pred)), accuracy_score(y_true, np.round(y_pred)), recall_score(y_true, np.round(y_pred)), precision_score(y_true, np.round(y_pred))]
    else:
        print("F1: {:.4f}".format(f1_score(y_true, np.round(y_pred))))
        print("Accuracy: {:.4f}".format(accuracy_score(y_true, np.round(y_pred))))
        print("Recall: {:.4f}".format(recall_score(y_true, np.round(y_pred))))
        print("Precision: {:.4f}".format(precision_score(y_true, np.round(y_pred))))
        print("AUC: {:.4f}".format(roc_auc_score(y_true, y_pred)))
        print((confusion_matrix(y_true, np.round(y_pred))))
