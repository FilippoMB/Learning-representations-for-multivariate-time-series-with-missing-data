import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from TS_datasets import getSynthData, getECGData
from utils import dim_reduction_plot, ideal_kernel
import numpy as np

# load data
_, _, _, _, test_data, test_labels = getECGData(tr_ratio = 0.5)
test_targets = test_data
sort_idx = np.argsort(test_labels,axis=0)[:,0]
test_labels = test_labels[sort_idx,:]
test_data = test_data[:,sort_idx,:]

# ideal kernel matrix
K_ts = ideal_kernel(test_labels)
plt.matshow(K_ts)
plt.show(block=False)

graph_name = "20170808-005344"

sess = tf.Session()
    
# restore graph
graph = tf.get_default_graph()
new_saver = tf.train.import_meta_graph("/tmp/tkae_models/m_"+graph_name+".ckpt.meta")
new_saver.restore(sess, "/tmp/tkae_models/m_"+graph_name+".ckpt")  

encoder_inputs = tf.get_collection("encoder_inputs")[0]
encoder_inputs_length = tf.get_collection("encoder_inputs_length")[0]
decoder_outputs = tf.get_collection("decoder_outputs")[0]
inf_outputs = tf.get_collection("inf_outputs")[0]
inf_loss = tf.get_collection("inf_loss")[0]
context_vector = tf.get_collection("context_vector")[0]
code_K = tf.get_collection("code_K")[0]

# evaluate graph on test set
fdts = {encoder_inputs: test_data,
    encoder_inputs_length: [test_data.shape[0] for _ in range(test_data.shape[1])],
    decoder_outputs: test_targets}
ts_pred, ts_loss, ts_context, ts_code_K = sess.run([inf_outputs, inf_loss, context_vector, code_K], fdts)
print('Test MSE: %.3f' % (ts_loss))

plt.matshow(ts_code_K)
plt.show(block=False)


# plot1
target = test_targets[:,0,0]
pred = ts_pred[:-1,0,0]
plt.plot(target, label='target')
plt.plot(pred, label='predicted')
plt.legend(loc='upper right')
plt.show(block=False)
print('Corr: %.3f' % ( pearsonr(target,pred)[0]) )

# plot 2
target = test_targets[:,1,0]
pred = ts_pred[:-1,1,0]
plt.plot(target, label='target')
plt.plot(pred, label='predicted')
plt.legend(loc='upper right')
plt.show(block=False)
print('Corr: %.3f' % ( pearsonr(target,pred)[0]) )

# plot 3
target = test_targets[:,2,0]
pred = ts_pred[:-1,2,0]
plt.plot(target, label='target')
plt.plot(pred, label='predicted')
plt.legend(loc='upper right')
plt.show(block=False)
print('Corr: %.3f' % ( pearsonr(target,pred)[0]) )

# dim reduction plots
dim_reduction_plot(ts_context, test_labels)

sess.close()