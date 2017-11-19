import numpy as np
import time
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
from TS_datasets import getBlood, getAF


def get_data():

    # original labels: 1=anomaly, 0=nominal   
    (x_tr, y_tr, _, _, _,
        _, _, _, _, _,
        x_te, y_te, _, _, _) = getAF() #getBlood()  

    # transpose [T, N, V] --> [N, T, V]
    x_tr = np.transpose(x_tr,axes=[1,0,2])
    x_te = np.transpose(x_te,axes=[1,0,2]) 

    # stack data
    x = np.vstack((x_tr, x_te))
    y_r = np.vstack((y_tr, y_te))
    # change type to handle negative numbers
    y_r = y_r.astype(np.int8)

    # reshape time series
    n = x.shape[0]
    t = x.shape[1]
    v = x.shape[2]
    x_r = np.reshape(x, [n, t * v])
    
    # modify labels for oneclass-SVM
    # 1=nominal, -1=non nominal
    n_zeros = y_r[y_r == 0].size
    n_ones = y_r[y_r == 1].size

    y_r[y_r == 1] = np.ones([n_ones, ]) * -1
    y_r[y_r == 0] = np.ones([n_zeros, ])

    return x_r, y_r

   
def get_problem_instance(x, y, nominal_percentage_training=0.5):

    index_ones = np.where(y == 1)[0]
    index_mones = np.where(y == -1)[0]

    x_nominal = x[index_ones, :]
    y_nominal = y[index_ones, ]
    x_nonnominal = x[index_mones, :]
    y_nonnominal = y[index_mones, ]

    tr_size = round(len(index_ones) * nominal_percentage_training)

    # shuffle nominal data (labels are all 1s)
    np.random.shuffle(x_nominal)

    # split training and test
    x_tr = x_nominal[:tr_size, :]
    y_tr = y_nominal[:tr_size]
    x_te = np.vstack((x_nominal[tr_size:, :], x_nonnominal))
    y_te = np.vstack((y_nominal[tr_size:], y_nonnominal))

    # shuffle test set data
    c = np.c_[x_te.reshape(len(x_te), -1), y_te.reshape(len(y_te), -1)]
    x2_te = c[:, :x_te.size // len(x_te)].reshape(x_te.shape)
    y2_te = c[:, x_te.size // len(x_te):].reshape(y_te.shape)
    np.random.shuffle(c)

    return x_tr, y_tr, x2_te, y2_te


def get_original_data():
    
    # original labels: 1=anomaly, 0=nominal   
    (x_tr, y_tr, _, _, _,
        _, _, _, _, _,
        x_te, y_te, _, _, _) = getAF()  

    # transpose and reshape [T, N, V] --> [N, T, V] --> [N, T*V]
    x_tr = np.transpose(x_tr,axes=[1,0,2])
    x_tr = np.reshape(x_tr, (x_tr.shape[0], x_tr.shape[1]*x_tr.shape[2]))
    x_te = np.transpose(x_te,axes=[1,0,2])
    x_te = np.reshape(x_te, (x_te.shape[0], x_te.shape[1]*x_te.shape[2]))   
    
    y_tr = y_tr.astype(np.int8)
    y_tr[y_tr == 1] = -1
    y_tr[y_tr == 0] = 1
    y_te = y_te.astype(np.int8)
    y_te[y_te == 1] = -1
    y_te[y_te == 0] = 1
    
    return x_tr, y_tr, x_te, y_te


if __name__ == "__main__":

    # not reproducible
    np.random.seed(None)

#    x, y = get_data()
#    x_tr, y_tr, x_te, y_te = get_problem_instance(x, y)
    x_tr, y_tr, x_te, y_te = get_original_data()
   
    # ------ OCSVM -------
    clf = svm.OneClassSVM(nu=0.5, kernel="rbf", gamma=0.7) #If gamma is ‘auto’ then 1/n_features will be used.
    clf.fit(x_tr)

    # prediction
    y_tr_pred = clf.predict(x_tr)
    y_te_pred = clf.predict(x_te)
    y_te_scores = clf.decision_function(x_te)
    print("OCSVM -- AUC: " + str(roc_auc_score(y_te, y_te_scores)))  
    
    print(classification_report(y_te, y_te_pred, target_names=['class -1', 'class 1']))

    # ------ IsolationForest -----
    clf_if = IsolationForest(contamination=0.5, random_state=np.random.RandomState())
    clf_if.fit(x_tr)
   
    y_tr_pred = clf_if.predict(x_tr)
    y_te_pred = clf_if.predict(x_te)
    y_te_scores = clf_if.decision_function(x_te)
    print("IForest -- AUC: " + str(roc_auc_score(y_te, y_te_scores)))  
    
    print(classification_report(y_te, y_te_pred, target_names=['class -1', 'class 1']))