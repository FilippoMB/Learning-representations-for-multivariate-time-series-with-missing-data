from gen_seq2seq_TF import s2sModel
import numpy as np
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)
import time
import tensorflow as tf
from sklearn import preprocessing
from TS_datasets import getLorentz, getLM, getSinusoids

# Hyperparams
PAD = 0
EOS = 0
batch_size = 50
num_epochs = 1000
tr_data_samples = 2000
ts_data_samples = 1000

config = dict(cell_type = 'RNN',
              num_layers = 2,
              hidden_units = 10,
              input_dim = 1,
              bidirect = 0,
              max_gradient_norm = 5,
              learning_rate = 0.001,)

# ================= DATASET =================
#training data
TS_gen = getLorentz()
data = np.asarray([next(TS_gen) for _ in range(tr_data_samples)])
data = preprocessing.scale(data,axis=1) # standardize the data
data = np.expand_dims(data,-1)
data = np.transpose(data,axes=[1,0,2]) # time_major=True
targets = data

# test data
test_data = np.asarray([next(TS_gen) for _ in range(ts_data_samples)])
test_data = preprocessing.scale(test_data,axis=1) # standardize the data
test_data = np.expand_dims(test_data,-1)
test_data = np.transpose(test_data,axes=[1,0,2]) # time_major=True
test_targets = test_data

# ================= GRAPH =================
tf.reset_default_graph()
sess = tf.Session()
G = s2sModel(config)
sess.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter('C:/Windows/Temp/tensorboard', graph=sess.graph)

# ================= TRAINING =================
time_tr_start = time.time()
max_batches = int(data.shape[1]/batch_size)
loss_track = []

try:
    for ep in range(num_epochs):
        
        # shuffle data each epoch
        idx = np.random.permutation(data.shape[1])
        data = data[:,idx,:] 
        targets = targets[:,idx,:] 
        
        for batch in range(max_batches):
            
            fd = {G.encoder_inputs: data[:,(batch)*batch_size:(batch+1)*batch_size,:],
                  G.encoder_inputs_length: [data.shape[0] for _ in range(batch_size)],
                  G.decoder_outputs: targets[:,(batch)*batch_size:(batch+1)*batch_size,:],}
            
            res = sess.run([G.update_step, G.train_loss], fd)     
            loss_track.append(res[1])
    
        if ep % 100 == 0:
            print('epoch {}'.format(ep))
            print('  minibatch loss: {}'.format(sess.run(G.train_loss, fd)))
            tr_pred, summary_ = sess.run([G.inf_outputs, G.merged_summary], fd)
#            tr_pred, summary_ = sess.run([G.tr_outputs, G.merged_summary], fd)
            train_writer.add_summary(summary_, ep)
            
            inp = fd[G.encoder_inputs][:,0,0]
            pred = tr_pred[:-1,0,0]
            plt.plot(inp, label='input')
            plt.plot(pred, label='predicted')
            plt.legend(loc='upper right')
            plt.show(block=False)

except KeyboardInterrupt:
    print('training interrupted')


plt.plot(loss_track)
plt.show(block=False)
time_tr_end = time.time()
print('loss {:.4f} after {} examples (batch_size={}). Tot time: {}'.format(loss_track[-1], len(loss_track)*batch_size, batch_size, time_tr_end-time_tr_start))

# ================= TEST =================
print('********** TEST **********')
fdts = {G.encoder_inputs: test_data,
        G.encoder_inputs_length: [test_data.shape[0] for _ in range(ts_data_samples)]}
ts_pred = sess.run(G.inf_outputs, fdts)
#ts_pred = sess.run(G.tr_outputs, fdts)
print('Test MSE: %.3f' % (np.mean(np.power(test_data-ts_pred[:-1,:,:],2))))

# plot 1
inp = fdts[G.encoder_inputs][:,0,0]
pred = ts_pred[:-1,0,0]
plt.plot(inp, label='input')
plt.plot(pred, label='predicted')
plt.legend(loc='upper right')
plt.show(block=False)
print('Corr: %.3f' % ( pearsonr(inp,pred)[0]) )

# plot 2
inp = fdts[G.encoder_inputs][:,1,0]
pred = ts_pred[:-1,1,0]
plt.plot(inp, label='input')
plt.plot(pred, label='predicted')
plt.legend(loc='upper right')
plt.show(block=False)
print('Corr: %.3f' % ( pearsonr(inp,pred)[0]) )

# plot 3
inp = fdts[G.encoder_inputs][:,2,0]
pred = ts_pred[:-1,2,0]
plt.plot(inp, label='input')
plt.plot(pred, label='predicted')
plt.legend(loc='upper right')
plt.show(block=False)
print('Corr: %.3f' % ( pearsonr(inp,pred)[0]) )

train_writer.close()
sess.close()