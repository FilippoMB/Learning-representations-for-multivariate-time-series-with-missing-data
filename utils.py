import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import numpy as np
from scipy import interpolate


def dim_reduction_plot(data, label, block_flag):
  
  # PCA
  PCA_model = TruncatedSVD(n_components=3).fit(data)
  data_PCA = PCA_model.transform(data)
  plt.scatter(data_PCA[:,0],data_PCA[:,1],c=label,marker='o',linewidths = 0,cmap='Paired')
  plt.title('PCA')
  plt.gca().axes.get_xaxis().set_ticks([])
  plt.gca().axes.get_yaxis().set_ticks([])
  plt.show(block=block_flag)
  
  # tSNE
  tSNE_model = TSNE(verbose=2, perplexity=30,min_grad_norm=1E-12,n_iter=3000)
  data_tsne = tSNE_model.fit_transform(data)
  plt.scatter(data_tsne[:,0],data_tsne[:,1],c=label,marker='o',linewidths = 0,cmap='Paired')
  plt.title('tSNE')
  plt.gca().axes.get_xaxis().set_ticks([])
  plt.gca().axes.get_yaxis().set_ticks([])
  plt.show(block=block_flag)
  return


def ideal_kernel(labels):
    K = np.zeros([labels.shape[0], labels.shape[0]])
    
    for i in range(labels.shape[0]):
        k = labels[i] == labels
        k.astype(int)
        K[:,i] = k[:,0]
    return K        
    

def interp_data(X, X_len, restore=False):
    """data are assumed to be time-major """
    
    [T, N, V] = X.shape
    X_new = np.zeros_like(X)
    
    # restore original lengths
    if restore:
        for n in range(N):
            t = np.linspace(start=0, stop=X_len[n], num=T)
            t_new = np.linspace(start=0, stop=X_len[n], num=X_len[n])
            for v in range(V):
                x_n_v = X[:,n,v]
                f = interpolate.interp1d(t, x_n_v)
                X_new[:X_len[n],n,v] = f(t_new)
            
    # interpolate all data to length T    
    else:
        for n in range(N):
            t = np.linspace(start=0, stop=X_len[n], num=X_len[n])
            t_new = np.linspace(start=0, stop=X_len[n], num=T)
            for v in range(V):
                x_n_v = X[:X_len[n],n,v]
                f = interpolate.interp1d(t, x_n_v)
                X_new[:,n,v] = f(t_new)
                
    return X_new


def classify_with_knn(train_data, train_labels, val_data, val_labels, min_k=3, max_k=4, step_k=1, plot_results=False):
    """
    Perform classification with knn by trying multiple k values.
    This function plots
    :param train_data:
    :param train_labels:
    :param val_data:
    :param val_labels:
    :param min_k:
    :param max_k:
    :param step_k:
    :param plot_results: Boolean indicating whether the function should plot the results
    :param return_results: If True, return a pair of k values and related classification accuracy,
        otherwise, the function returns None
    :return:
    """

    from sklearn.neighbors import KNeighborsClassifier

    k_values = []
    knn_acc = []
    for k in range(min_k, max_k, step_k):
        k_values.append(k)
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(train_data, train_labels)
        accuracy = neigh.score(val_data, val_labels)
        knn_acc.append(accuracy)

    if plot_results:
        import matplotlib.pyplot as plt

        plt.plot(k_values, knn_acc)
        plt.xlabel("K")
        plt.ylabel("Accuracy")
        plt.show()

    return k_values, knn_acc

def mse_and_corr(targets, preds, targets_len):
    """
    targets and preds must have shape [time_steps, samples, variables]
    targets_len must have shape [samples,]
    """
    mse_list = []
    corr_list = []
    for i in range(targets.shape[1]):
        len_i = targets_len[i]
        test_data_i = targets[:len_i,i,:]
        pred_i = preds[:len_i,i,:]
        mse_list.append(np.mean((test_data_i-pred_i)**2))
        corr_list.append(np.corrcoef(test_data_i.flatten(), pred_i.flatten())[0,1])
    tot_mse = np.mean(mse_list)
    tot_corr = np.mean(corr_list)
    
    return tot_mse, tot_corr

def corr2_coeff(A,B):
    """
    Row-wise mean of input arrays & subtract from input arrays themeselves
    :param A:
    :param B:
    :return:
    """

    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T)/np.sqrt(np.dot(ssA[:, None], ssB[None]))
