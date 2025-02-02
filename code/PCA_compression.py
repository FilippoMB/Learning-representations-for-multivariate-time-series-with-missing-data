from TS_datasets import *
import numpy as np
from sklearn.decomposition import PCA
import argparse, sys
from utils import interp_data, classify_with_knn, mse_and_corr, dim_reduction_plot, anomaly_detect
import matplotlib.pyplot as plt

dim_red = 1
plot_on = 0
interp_on = 0
anomaly_detect_on = 0

# parse input data
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_id", default='TSm', help="ID of the dataset (SYNTH, ECG, JAP, etc..)", type=str)
parser.add_argument("--num_comp", default=10, help="number of PCA components", type=int)
args = parser.parse_args()
print(args)

# ================== DATASET ===================
if args.dataset_id == 'SYNTH':
    (train_data, train_labels, train_len, _, _,
        _, _, _, _, _,
        test_data_orig, test_labels, test_len, _, _) = getSynthData(name='Lorentz', tr_data_samples=100, 
                                                 vs_data_samples=100, 
                                                 ts_data_samples=500)
        
elif args.dataset_id == 'ECG':
    (train_data, train_labels, train_len, _, _,
        _, _, _, _, _,
        test_data_orig, test_labels, test_len, _, _) = getECGData()

elif args.dataset_id == 'ECG2':
    (train_data, train_labels, train_len, _, _,
        _, _, _, _, _,
        test_data_orig, test_labels, test_len, _, _) = getECGDataFull()
        
elif args.dataset_id == 'LIB':
    (train_data, train_labels, train_len, _, _,
        _, _, _, _, _,
        test_data_orig, test_labels, test_len, _, _) = getLibras()

elif args.dataset_id == 'ARAB':
    (train_data, train_labels, train_len, _, _,
        _, _, _, _, _,
        test_data_orig, test_labels, test_len, _, _) = getArab()
          
elif args.dataset_id == 'JAP':        
    (train_data, train_labels, train_len, _, _,
        _, _, _, _, _,
        test_data_orig, test_labels, test_len, _, _) = getJapDataFull() 
    
elif args.dataset_id == 'TSm':        
    (train_data, train_labels, train_len, _, _,
        _, _, _, _, _,
        test_data_orig, test_labels, test_len, _, _) = getDataMiss(ds_name='Arabic', #'JapaneseVowels' 
                                                                      kernel='TCK', 
                                                                      inp='zero', 
                                                                      miss=0.8)
    
elif args.dataset_id == 'CHAR':        
    (train_data, train_labels, train_len, _, _,
        _, _, _, _, _,
        test_data_orig, test_labels, test_len, _, _) = getCharDataFull() 

elif args.dataset_id == 'WAF':        
    (train_data, train_labels, train_len, _, _,
        _, _, _, _, _,
        test_data_orig, test_labels, test_len, _, _) = getWafer()     
    
elif args.dataset_id == 'SIN':        
    (train_data, train_labels, train_len, _, _,
        _, _, _, _, _,
        test_data_orig, test_labels, test_len, _, _) = getSins()    

elif args.dataset_id == 'MSO':        
    (train_data, train_labels, train_len, _, _,
        _, _, _, _, _,
        test_data_orig, test_labels, test_len, _, _) = getMSO() 
    
elif args.dataset_id == 'ODE':        
    (train_data, train_labels, train_len, _, _,
        _, _, _, _, _,
        test_data_orig, test_labels, test_len, _, _) = getODE() 

elif args.dataset_id == 'ODE2':        
    (train_data, train_labels, train_len, _, _,
        _, _, _, _, _,
        test_data_orig, test_labels, test_len, _, _) = getODE_mc()
    
elif args.dataset_id == 'AUS':        
    (train_data, train_labels, train_len, _, _,
        _, _, _, _, _,
        test_data_orig, test_labels, test_len, _, _) = getAuslan()   

elif args.dataset_id == 'BLOOD':        
    (train_data, train_labels, train_len, _, _,
        _, _, _, _, _,
        test_data_orig, test_labels, test_len, _, _) = getBlood()  
    
elif args.dataset_id == 'AF':        
    (train_data, train_labels, train_len, _, _,
        _, _, _, _, _,
        test_data_orig, test_labels, test_len, _, _) = getAF(inp='zero')   
    
else:
    sys.exit('Invalid dataset_id')

# interpolation
if np.min(train_len) < np.max(train_len) and interp_on:
    print('-- Data Interpolation --')
    train_data = interp_data(train_data, train_len)
    test_data = interp_data(test_data_orig, test_len)
else:
    test_data = test_data_orig

# transpose and reshape [T, N, V] --> [N, T, V] --> [N, T*V]
train_data = np.transpose(train_data,axes=[1,0,2])
train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1]*train_data.shape[2]))
test_data = np.transpose(test_data,axes=[1,0,2])
test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1]*test_data.shape[2]))

  
print('\n**** Processing {}: Tr{}, Ts{} ****\n'.format(args.dataset_id, train_data.shape, test_data.shape))

# ==============================================

# PCA compression
pca = PCA(n_components=args.num_comp, svd_solver='full')
tr_proj = pca.fit_transform(train_data)
ts_proj = pca.transform(test_data)
ts_pred = pca.inverse_transform(ts_proj)

# reverse transformations
ts_pred = np.reshape(ts_pred, (test_data_orig.shape[1], test_data_orig.shape[0], test_data_orig.shape[2]))
ts_pred = np.transpose(ts_pred,axes=[1,0,2])
test_data = test_data_orig

if np.min(train_len) < np.max(train_len) and interp_on:
    print('-- Reverse Interpolation --')
    ts_pred = interp_data(ts_pred, test_len, restore=True)

# MSE and corr
test_mse, test_corr = mse_and_corr(test_data, ts_pred, test_len)
print('Test MSE=%.3f, Corr=%.3f'%(test_mse, test_corr))

# kNN classification on the codes
acc, f1 = classify_with_knn(tr_proj, train_labels[:, 0], ts_proj, test_labels[:, 0], k=3)
print('kNN -- acc: %.3f, F1: %.3f'%(acc, f1))

# anomaly detection
if anomaly_detect_on:
    auc = anomaly_detect(test_data, ts_pred, test_len, test_labels, plot_on)
    print('Anomaly detect -- AUC: %.3f'%(auc))

# plot reconstruction
if plot_on:
    plot_idx1 = np.random.randint(low=0,high=test_data.shape[0])
    plot_idx1=4
    ytarget = test_data[:,plot_idx1,0]
    ypred = ts_pred[:,plot_idx1,0]
    plt.plot(ytarget, label='target')
    plt.plot(ypred, label='ts_pred')
    plt.legend(loc='upper right')
    plt.show(block=True)  
    np.savetxt('../logs/PCA_pred',ypred)
    
    plt.scatter(ts_proj[:,0],ts_proj[:,1],s=20,c=test_labels,marker='o',linewidths = 0,cmap='Paired')
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.show()
    
# dim reduction plots
if dim_red:
    dim_reduction_plot(ts_proj, test_labels[:, 0])