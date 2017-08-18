from TS_datasets import getSynthData, getECGData, getJapData
import numpy as np
from sklearn.decomposition import PCA
import argparse, sys
import matplotlib.pyplot as plt

# parse input data
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_id", default='JAP', help="ID of the dataset (SYNTH, ECG, JAP)", type=str)
parser.add_argument("--num_comp", default=10, help="number of PCA components", type=int)
args = parser.parse_args()

# ================= DATASET =================
if args.dataset_id == 'SYNTH':
    (train_data, train_labels, _, _, _,
        _, _, _, _, _,
        test_data, _, _, _, _) = getSynthData(name='Lorentz', tr_data_samples=2000, 
                                                 vs_data_samples=1, 
                                                 ts_data_samples=2000)
    
    # transpose
    train_data = train_data[:,:,0].T
    test_data = test_data[:,:,0].T
    
elif args.dataset_id == 'ECG':
    (train_data, train_labels, _, _, _,
        _, _, _, _, _,
        test_data, _, _, _, _) = getECGData(tr_ratio = 0.4)
    
    # transpose
    train_data = train_data[:,:,0].T
    test_data = test_data[:,:,0].T
       
elif args.dataset_id == 'JAP':        
    (train_data, train_labels, _, _, _,
        _, _, _, _, _,
        test_data, _, _, _, _) = getJapData()
    
    # transpose and reshape
    train_data = np.transpose(train_data,axes=[1,0,2])
    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1]*train_data.shape[2]))
    test_data = np.transpose(test_data,axes=[1,0,2])
    test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1]*test_data.shape[2]))
else:
    sys.exit('Invalid dataset_id')

print('\n**** Processing {}: Tr{}, Ts{} ****\n'.format(args.dataset_id, train_data.shape, test_data.shape))

# PCA compression
pca = PCA(n_components=args.num_comp)
pca.fit(train_data)
proj = pca.transform(test_data)
pred = pca.inverse_transform(proj)

# loss
ts_loss = np.mean((test_data-pred)**2)
print('Test MSE: %.3f' % (ts_loss))

plot_idx1 = np.random.randint(low=0,high=test_data.shape[0])
target = test_data[plot_idx1,:]
pred = pred[plot_idx1,:-1]
plt.plot(target, label='target')
plt.plot(pred, label='pred')
plt.legend(loc='upper right')
plt.show(block=False)  
