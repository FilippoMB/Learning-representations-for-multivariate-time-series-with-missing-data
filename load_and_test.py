import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from TS_datasets import getSynthData, getECGData, getJapData
from utils import dim_reduction_plot
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import argparse, sys

block_flag = True

# parse input data
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_id", default='JAP', help="ID of the dataset (SYNTH, ECG, JAP)", type=str)
parser.add_argument("--graph_name", default="20170814-000240", help="name of the file to be loaded", type=str)
parser.add_argument("--reverse_input", dest='reverse_input', action='store_true', help="fed input reversed for training")
parser.add_argument("--dim_red", dest='dim_red', action='store_true', help="compute PCA and tSNE")
parser.add_argument("--plot_on", dest='plot_on', action='store_true', help="make plots")
parser.set_defaults(reverse_input=True)
parser.set_defaults(dim_red=False)
parser.set_defaults(plot_on=True)
args = parser.parse_args()

# ================= LOAD DATA ===================

if args.dataset_id == 'SYNTH':
    (train_data, train_labels, _, _, _,
        _, _, _, _, _,
        test_data, test_labels, _, test_targets, K_ts) = getSynthData(name='Lorentz', 
                                                                      tr_data_samples=2000, 
                                                                      vs_data_samples=2000, 
                                                                      ts_data_samples=2000)
elif args.dataset_id == 'ECG':
    (train_data, train_labels, _, _, _,
        _, _, _, _, _,
        test_data, test_labels, _, test_targets, K_ts) = getECGData(tr_ratio = 0)
       
elif args.dataset_id == 'JAP':        
    (train_data, train_labels, _, _, _,
        _, _, _, _, _,
        test_data, test_labels, _, test_targets, K_ts) = getJapData(kernel='TCK',inp=None,zscore=True)
    
else:
    sys.exit('Invalid dataset_id')

print('\n**** Processing {}: Tr{}, Ts{} ****\n'.format(args.dataset_id, train_data.shape, test_data.shape))

# sort test data (for visualize the learned K)
sort_idx = np.argsort(test_labels,axis=0)[:,0]
test_labels = test_labels[sort_idx,:]
test_data = test_data[:,sort_idx,:]
test_targets = test_targets[:,sort_idx,:]

if args.reverse_input:
    train_data = train_data[::-1,:,:]
    test_data = test_data[::-1,:,:]

# target kernel matrix
if args.plot_on:
    plt.matshow(K_ts,cmap='binary_r')
    plt.title('prior K')
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.show(block=block_flag)

# ================== RESTORE AND EVAL MODEL ==================
sess = tf.Session()
    
# restore graph
new_saver = tf.train.import_meta_graph("/tmp/tkae_models/m_"+args.graph_name+".ckpt.meta", clear_devices=True)
new_saver.restore(sess, "/tmp/tkae_models/m_"+args.graph_name+".ckpt")  

encoder_inputs = tf.get_collection("encoder_inputs")[0]
encoder_inputs_length = tf.get_collection("encoder_inputs_length")[0]
decoder_outputs = tf.get_collection("decoder_outputs")[0]
inf_outputs = tf.get_collection("inf_outputs")[0]
inf_loss = tf.get_collection("inf_loss")[0]
context_vector = tf.get_collection("context_vector")[0]
code_K = tf.get_collection("code_K")[0]

# get context vectors from training data
fdtr = {encoder_inputs: train_data,
        encoder_inputs_length: [train_data.shape[0] for _ in range(train_data.shape[1])],
        }
tr_context = sess.run(context_vector, fdtr)

# get context vectors and predictions from test data
fdts = {encoder_inputs: test_data,
        encoder_inputs_length: [test_data.shape[0] for _ in range(test_data.shape[1])],
        decoder_outputs: test_targets}
ts_pred, ts_loss, ts_context, ts_code_K = sess.run([inf_outputs, inf_loss, context_vector, code_K], fdts)
sess.close()

# =============== DATA ANALYSIS ===============
print('Test MSE: %.3f' % (ts_loss))

if args.plot_on:
    
    # plot kernel code
    plt.matshow(ts_code_K,cmap='binary_r')
    plt.title('code K')
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.show(block=block_flag)
    
    plt_idx1 = np.random.randint(low=0, high=test_data.shape[1]-1, size=3)
    plt_idx2 = np.random.randint(low=0, high=test_data.shape[2]-1, size=3)

    # plot ts1
    target = test_targets[:,plt_idx1[0],plt_idx2[0]]
    pred = ts_pred[:-1,plt_idx1[0],plt_idx2[0]]
    plt.plot(target, label='target')
    plt.plot(pred, label='predicted')
    plt.legend(loc='upper right')
    plt.show(block=block_flag)
    print('Corr: %.3f' % ( pearsonr(target,pred)[0]) )
    
    # plot ts2
    target = test_targets[:,plt_idx1[1],plt_idx2[1]]
    pred = ts_pred[:-1,plt_idx1[1],plt_idx2[1]]
    plt.plot(target, label='target')
    plt.plot(pred, label='predicted')
    plt.legend(loc='upper right')
    plt.show(block=block_flag)
    print('Corr: %.3f' % ( pearsonr(target,pred)[0]) )
    
    # plot ts3
    target = test_targets[:,plt_idx1[2],plt_idx2[2]]
    pred = ts_pred[:-1,plt_idx1[2],plt_idx2[2]]
    plt.plot(target, label='target')
    plt.plot(pred, label='predicted')
    plt.legend(loc='upper right')
    plt.show(block=block_flag)
    print('Corr: %.3f' % ( pearsonr(target,pred)[0]) )

# dim reduction plots
if args.dim_red:
    dim_reduction_plot(ts_context, test_labels, block_flag)

# kNN classification on the codes
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(tr_context, train_labels[:,0])
accuracy = neigh.score(ts_context, test_labels[:,0])
print('kNN accuarcy: {}'.format(accuracy))