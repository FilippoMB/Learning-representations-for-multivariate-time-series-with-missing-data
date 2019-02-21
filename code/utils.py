from sklearn.decomposition import TruncatedSVD
import numpy as np
from scipy import interpolate
import seaborn as sns
import matplotlib.pyplot as plt
from palettable.colorbrewer import qualitative


def dim_reduction_plot(data, label):
      
    # PCA
    PCA_model = TruncatedSVD(n_components=3).fit(data)
    data_PCA = PCA_model.transform(data)
    
    plot_clusters(data_PCA,label, list(range(np.max(label)))) #['infected','not infected']
    
#    plt.figure(figsize=(3,3))
##    plt.scatter(data_PCA[:,0],data_PCA[:,1],s=80,c=label,marker='.',linewidths=0.15,cmap=bmap, alpha=0.5,edgecolor='black')
#    plt.scatter(data_PCA[label==1,0],data_PCA[label==1,1],s=40,c=colors[1],marker='^',linewidths=0.15,cmap=bmap, alpha=0.5,edgecolor='black',label='not infected')
#    plt.scatter(data_PCA[label==0,0],data_PCA[label==0,1],s=100,c=colors[0],marker='.',linewidths=0.15,cmap=bmap, alpha=0.5,edgecolor='black',label='infected')   
##    plt.title('PCA')
#    plt.gca().axes.get_xaxis().set_ticks([])
#    plt.gca().axes.get_yaxis().set_ticks([])
#    for spine in  plt.gca().axes.spines.values():
#        spine.set_edgecolor('#363636')
##    plt.legend(loc='best', numpoints=1)
#    plt.legend(loc='best', shadow=True, fancybox=True, numpoints=1)
#    plt.savefig('../logs/PCA.pdf',format='pdf')
#    plt.show()
  
#    # tSNE
#    from sklearn.manifold import TSNE
#    tSNE_model = TSNE(verbose=2, perplexity=30,min_grad_norm=1E-12,n_iter=3000)
#    data_tsne = tSNE_model.fit_transform(data)
#    plt.scatter(data_tsne[:,0],data_tsne[:,1],c=label,marker='.',linewidths = 0,cmap='Paired')
#    plt.title('tSNE')
#    plt.gca().axes.get_xaxis().set_ticks([])
#    plt.gca().axes.get_yaxis().set_ticks([])
#    plt.show()
    return

def plot_clusters(data,labels,legend_names):
        
    clust = np.unique(labels)
    n_clust = clust.shape[0]
    
    colors = qualitative.Paired_12.mpl_colors
    print('colors: ',len(colors))
       
    g = sns.JointGrid(x=data[:,0], y=data[:,1], size=4)
    
    for i in range(n_clust):
        
        idx_i = labels==clust[i]
        g.x = data[idx_i,0]
        g.y = data[idx_i,1]

        g.plot_joint(plt.scatter, marker='o', s=40, linewidths=0.15, color=colors[i], alpha=0.5,edgecolor='black',label=legend_names[i])
        g = g.plot_marginals(sns.kdeplot, shade=True, color=colors[i])
        
    g.ax_joint.set_xticks([])
    g.ax_joint.set_yticks([])
    g.ax_joint.spines['bottom'].set_color('gray')
    g.ax_joint.spines['left'].set_color('gray')
    g.ax_marg_x.spines['bottom'].set_color('gray')
    g.ax_marg_y.spines['left'].set_color('gray')
    g.ax_joint.legend(prop={'size':8},loc='upper right', shadow=True, fancybox=True, scatterpoints=1)
    
    plt.savefig('../logs/PCA.pdf',format='pdf')
    return

def ideal_kernel(labels):
    K = np.zeros([labels.shape[0], labels.shape[0]])
    
    for i in range(labels.shape[0]):
        k = labels[i] == labels
        k.astype(int)
        K[:,i] = k[:,0]
    return K        
    

def interp_data(X, X_len, restore=False, interp_kind='linear'):
    """data are assumed to be time-major
    interp_kind: can be 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
    """
    
    [T, N, V] = X.shape
    X_new = np.zeros_like(X)
    
    # restore original lengths
    if restore:
        for n in range(N):
            t = np.linspace(start=0, stop=X_len[n], num=T)
            t_new = np.linspace(start=0, stop=X_len[n], num=X_len[n])
            for v in range(V):
                x_n_v = X[:,n,v]
                f = interpolate.interp1d(t, x_n_v, kind=interp_kind)
                X_new[:X_len[n],n,v] = f(t_new)
            
    # interpolate all data to length T    
    else:
        for n in range(N):
            t = np.linspace(start=0, stop=X_len[n], num=X_len[n])
            t_new = np.linspace(start=0, stop=X_len[n], num=T)
            for v in range(V):
                x_n_v = X[:X_len[n],n,v]
                f = interpolate.interp1d(t, x_n_v, kind=interp_kind)
                X_new[:,n,v] = f(t_new)
                
    return X_new


def classify_with_knn(train_data, train_labels, test_data, test_labels, k=3, metric='minkowski'):
    """
    Perform classification with knn.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import f1_score
    
    num_classes = len(np.unique(test_labels))

    neigh = KNeighborsClassifier(n_neighbors=k, metric=metric)
    neigh.fit(train_data, train_labels)
    accuracy = neigh.score(test_data, test_labels)
    pred_labels = neigh.predict(test_data)
    if num_classes > 2:
        F1 = f1_score(test_labels, pred_labels, average='weighted')
    else:
        F1 = f1_score(test_labels, pred_labels, average='binary')

    return accuracy, F1

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

def anomaly_detect(targets, preds, targets_len, target_labels, plot_on=False):
    from sklearn.metrics import roc_auc_score
    
    mse_list = []
    for i in range(targets.shape[1]):
        len_i = targets_len[i]
        test_data_i = targets[:len_i,i,:]
        pred_i = preds[:len_i,i,:]
        err_i = np.mean((test_data_i-pred_i)**2)
        mse_list.append(err_i)

    auc = roc_auc_score(target_labels, mse_list, average=None)
    
    if plot_on:
        
        import brewer2mpl
        import matplotlib.pyplot as plt
        
        n_class0 = np.sum(target_labels==0)
        n_class1 = np.sum(target_labels==1)
                
        bmap = brewer2mpl.get_map('Set1', 'qualitative', 3)
        colors = bmap.mpl_colors
        plt.figure(figsize=(5.5,3)) 
        #plt.grid() 
        plt.xlim(0,n_class0+n_class1-1)
        plt.ylabel('Reconstruction MSE')
        plt.yscale('log')
        plt.title('Outlier detection')        
        plt.bar(np.arange(n_class0), mse_list[:n_class0], color=colors[0], edgecolor='none',width=1.0,label='nominal')
        plt.bar(np.arange(n_class0,len(mse_list)), mse_list[n_class0:], color=colors[1], edgecolor='none',width=1.0,label='outlier') 
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.legend(loc='best', fontsize=10)       
        for spine in  plt.gca().axes.spines.values():
            spine.set_edgecolor('#565656')
        plt.savefig('../logs/Anomaly_detect.pdf',format='pdf')
        plt.show()
        
    return auc
    

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

def reverse_input(data, data_len):
    
    data_reversed = np.zeros_like(data)
    for i in range(data_len.shape[0]):
        len_i = data_len[i]
        data_i = data[:len_i,i,:]
        data_reversed[:len_i,i,:] = data_i[::-1,:]
    
    return data_reversed