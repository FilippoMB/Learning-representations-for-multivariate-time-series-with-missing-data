import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import numpy as np

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
    