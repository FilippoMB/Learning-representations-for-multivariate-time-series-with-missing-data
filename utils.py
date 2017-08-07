import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import numpy as np

def dim_reduction_plot(data, label):
  
 
  # PCA
  PCA_model = TruncatedSVD(n_components=3).fit(data)
  data_PCA = PCA_model.transform(data)
  plt.scatter(data_PCA[:,0],data_PCA[:,1],c=label,marker='*',linewidths = 0)
  plt.title('PCA')
  plt.show(block=False)
  
  # tSNE
  tSNE_model = TSNE(verbose=2, perplexity=30,min_grad_norm=1E-12,n_iter=3000)
  data_tsne = tSNE_model.fit_transform(data)
  plt.scatter(data_tsne[:,0],data_tsne[:,1],c=label,marker='*',linewidths = 0)
  plt.title('tSNE')
  plt.show(block=False)
  return

def ideal_kernel(labels):
    K = np.zeros([labels.shape[0], labels.shape[0]])
    
    for i in range(labels.shape[0]):
        k = labels[i] == labels
        k.astype(int)
        K[:,i] = k
    return K        
    