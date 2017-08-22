from TS_datasets import getSynthData, getECGData, getJapDataFull, getCharDataFull
import numpy as np
from sklearn.decomposition import PCA
import argparse, sys
from utils import interp_data
from utils import classify_with_knn

# parse input data
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_id", default='CHAR', help="ID of the dataset (SYNTH, ECG, JAP)", type=str)
parser.add_argument("--num_comp", default=6, help="number of PCA components", type=int)
args = parser.parse_args()

# ================== DATASET ===================
if args.dataset_id == 'SYNTH':
    (train_data, train_labels, _, _, _,
        _, _, _, _, _,
        test_data, test_labels, _, _, _) = getSynthData(name='Lorentz', tr_data_samples=2000, 
                                                 vs_data_samples=1, 
                                                 ts_data_samples=2000)
    
    # transpose --> [N, T, V]
    train_data = train_data[:,:,0].T
    test_data = test_data[:,:,0].T
    
elif args.dataset_id == 'ECG':
    (train_data, train_labels, _, _, _,
        _, _, _, _, _,
        test_data, test_labels, _, _, _) = getECGData(tr_ratio = 0)
    
    # transpose --> [N, T, V]
    train_data = train_data[:,:,0].T
    test_data = test_data[:,:,0].T
       
elif args.dataset_id == 'JAP':        
    (train_data, train_labels, train_len, _, _,
        _, _, _, _, _,
        test_data_orig, test_labels, test_len, _, _) = getJapDataFull() 
    
elif args.dataset_id == 'CHAR':        
    (train_data, train_labels, train_len, _, _,
        _, _, _, _, _,
        test_data_orig, test_labels, test_len, _, _) = getCharDataFull() 
    
else:
   
    sys.exit('Invalid dataset_id')

if args.dataset_id == 'JAP' or args.dataset_id == 'CHAR':
    
    # interpolate
    train_data = interp_data(train_data, train_len)
    test_data = interp_data(test_data_orig, test_len)
    
    # transpose and reshape --> [N, T, V] --> [N, T*V]
    train_data = np.transpose(train_data,axes=[1,0,2])
    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1]*train_data.shape[2]))
    test_data = np.transpose(test_data,axes=[1,0,2])
    test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1]*test_data.shape[2]))

print('\n**** Processing {}: Tr{}, Ts{} ****\n'.format(args.dataset_id, train_data.shape, test_data.shape))

# ==============================================

# PCA compression
pca = PCA(n_components=args.num_comp)
tr_proj = pca.fit_transform(train_data)
ts_proj = pca.transform(test_data)
pred = pca.inverse_transform(ts_proj)

# reverse transformations
if args.dataset_id == 'JAP' or args.dataset_id == 'CHAR': 
    pred = np.reshape(pred, (test_data_orig.shape[1], test_data_orig.shape[0], test_data_orig.shape[2]))
    pred = np.transpose(pred,axes=[1,0,2])
    pred = interp_data(pred, test_len, restore=True)
    test_data = test_data_orig

# loss
ts_loss = np.mean((test_data[np.nonzero(test_data)]-pred[np.nonzero(test_data)])**2)
print('Test MSE: {}'.format(ts_loss))

# kNN classification on the codes
classify_with_knn(tr_proj, train_labels[:, 0], ts_proj, test_labels[:, 0])


## plot reconstruction
#plot_idx1 = np.random.randint(low=0,high=test_data.shape[0])
#target = test_data[plot_idx1,:]
#pred = pred[plot_idx1,:-1]
#plt.plot(target, label='target')
#plt.plot(pred, label='pred')
#plt.legend(loc='upper right')
#plt.show(block=False)  
