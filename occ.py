import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


def get_data():
    import scipy.io as sio

    x_tr = sio.loadmat('Blood/BLOOD_full.mat')['X']
    x_te = sio.loadmat('Blood/BLOOD_full.mat')['Xte']
    # original labels: 1=anomaly, 0=nominal
    y_tr = sio.loadmat('Blood/BLOOD_full.mat')['Y']
    y_te = sio.loadmat('Blood/BLOOD_full.mat')['Yte']
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


def get_problem_instance(x, y, seed, nominal_percentage_training=0.5):

    index_ones = np.where(y == 1)[0]
    index_mones = np.where(y == -1)[0]

    x_nominal = x[index_ones, :]
    y_nominal = y[index_ones, ]
    x_nonnominal = x[index_mones, :]
    y_nonnominal = y[index_mones, ]

    tr_size = round(len(index_ones) * nominal_percentage_training)

    # shuffle nominal data (so we randomize data in the training set)
    np.random.seed(seed)
    c = np.c_[x_nominal.reshape(len(x_nominal), -1), y_nominal.reshape(len(y_nominal), -1)]
    x2_nominal = c[:, :x_nominal.size // len(x_nominal)].reshape(x_nominal.shape)
    y2_nominal = c[:, x_nominal.size // len(x_nominal):].reshape(y_nominal.shape)
    np.random.shuffle(c)

    # split training and test
    x_tr = x2_nominal[:tr_size, :]
    y_tr = y2_nominal[:tr_size]
    x_te = np.vstack((x2_nominal[tr_size:, :], x_nonnominal))
    y_te = np.vstack((y2_nominal[tr_size:], y_nonnominal))

    # shuffle test set data
    c = np.c_[x_te.reshape(len(x_te), -1), y_te.reshape(len(y_te), -1)]
    x2_te = c[:, :x_te.size // len(x_te)].reshape(x_te.shape)
    y2_te = c[:, x_te.size // len(x_te):].reshape(y_te.shape)
    np.random.shuffle(c)

    return x_tr, y_tr, x2_te, y2_te


seed = np.random.randint(low=1, high=10000)

x, y = get_data()
x_tr, y_tr, x_te, y_te = get_problem_instance(x, y, seed)

# fit the model (with fixed hyper-parameters)
clf = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma=0.01)
clf.fit(x_tr)

# prediction
y_tr_pred = clf.predict(x_tr)
y_te_pred = clf.predict(x_te)

# evaluation
acc_tr = accuracy_score(y_tr, y_tr_pred)
acc_te = accuracy_score(y_te, y_te_pred)


print("Accuracy on training set: " + str(acc_tr))
print("Accuracy on test set: " + str(acc_te))
print("AUC: " + str(roc_auc_score(y_te, y_te_pred)))
print(classification_report(y_te, y_te_pred, target_names=['class -1', 'class 1']))
