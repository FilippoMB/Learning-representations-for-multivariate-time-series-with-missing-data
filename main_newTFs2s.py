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
batch_size = 50
num_epochs = 10000
tr_data_samples = 2000
vs_data_samples = 1000
ts_data_samples = 1000

alternate_train = 0 # train both on inferenced and external inputs

config = dict(cell_type = 'RNN',
              num_layers = 2,
              hidden_units = 5,
              input_dim = 1,
              bidirect = 1,
              max_gradient_norm = 5, # TODO: check variance (Bengio)
              learning_rate = 0.001,
              EOS = 0,
              last_layer_state_only = 0)
print(config)

# ================= DATASET =================
#training data
TS_gen = getLorentz()
data = np.asarray([next(TS_gen) for _ in range(tr_data_samples)])
data = preprocessing.scale(data,axis=1) # standardize the data
data = np.expand_dims(data,-1)
data = np.transpose(data,axes=[1,0,2]) # time_major=True
targets = data

# validation
valid_data = np.asarray([next(TS_gen) for _ in range(vs_data_samples)])
valid_data = preprocessing.scale(valid_data,axis=1) # standardize the data
valid_data = np.expand_dims(valid_data,-1)
valid_data = np.transpose(valid_data,axes=[1,0,2]) # time_major=True
valid_targets = valid_data

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

# ================= DEBUG =================

#encoder_state = sess.run(G.encoder_state, {G.encoder_inputs: data, G.encoder_inputs_length: [data.shape[0] for _ in range(tr_data_samples)]} )  

# ================= TRAINING =================
time_tr_start = time.time()
max_batches = int(data.shape[1]/batch_size)
tr_loss_track = []
inf_loss_track = []

try:
    for ep in range(num_epochs):
        
        # shuffle data
#        if ep % 1:
        idx = np.random.permutation(data.shape[1])
        data = data[:,idx,:] 
        targets = targets[:,idx,:] 
        
#        if ep % 10 == 0 and alternate_train: # train on external inputs
        if False:
            for batch in range(max_batches):
                
                fdtr = {G.encoder_inputs: data[:,(batch)*batch_size:(batch+1)*batch_size,:],
                        G.encoder_inputs_length: [data.shape[0] for _ in range(batch_size)],
                        G.decoder_outputs: targets[:,(batch)*batch_size:(batch+1)*batch_size,:]}
                
                _, tr_loss = sess.run([G.tr_update_step, G.tr_loss], fdtr)     
                tr_loss_track.append(tr_loss)                          
                
        else: # train on inferenced inputs
            for batch in range(max_batches):
                
                fdtr = {G.encoder_inputs: data[:,(batch)*batch_size:(batch+1)*batch_size,:],
                        G.encoder_inputs_length: [data.shape[0] for _ in range(batch_size)],
                        G.decoder_outputs: targets[:,(batch)*batch_size:(batch+1)*batch_size,:]}
                
                _, inf_loss = sess.run([G.inf_update_step, G.inf_loss], fdtr)     
                inf_loss_track.append(inf_loss)
            
        # TODO: save model yielding best results on validation
        if ep % 100 == 0:   
            print('Ep: {}'.format(ep))
#            fdvs = {G.encoder_inputs: valid_data,
#                    G.encoder_inputs_length: [valid_data.shape[0] for _ in range(vs_data_samples)],
#                    G.decoder_outputs: valid_targets}
#            vs_pred, vs_loss, summary_ = sess.run([G.inf_outputs, G.inf_loss, G.merged_summary], fdvs)
#            train_writer.add_summary(summary_, ep)
#            print('epoch {} -- Valid MSE {}'.format(ep, vs_loss))            
#            # plot
#            inp = fdvs[G.encoder_inputs][:,0,0]
#            pred = vs_pred[:-1,0,0]
#            plt.plot(inp, label='input')
#            plt.plot(pred, label='predicted')
#            plt.legend(loc='upper right')
#            plt.show(block=False)
            
            fdtr = {G.encoder_inputs: data,
                    G.encoder_inputs_length: [data.shape[0] for _ in range(tr_data_samples)],
                    G.decoder_outputs: data}
            inf_pred, inf_loss = sess.run([G.inf_outputs, G.inf_loss], fdtr)
            print('Train MSE (INF) {}'.format(inf_loss))
            # plot
            inp = fdtr[G.encoder_inputs][:,0,0]
            pred = inf_pred[:-1,0,0]
            plt.plot(inp, label='input')
            plt.plot(pred, label='predicted')
            plt.legend(loc='upper right')
            plt.show(block=False)  
            
            fdtr = {G.encoder_inputs: data,
                    G.encoder_inputs_length: [data.shape[0] for _ in range(tr_data_samples)],
                    G.decoder_outputs: data}
            tr_pred, tr_loss = sess.run([G.tr_outputs, G.tr_loss], fdtr)
            print('Train MSE (EXT) {}'.format(tr_loss))           
            # plot
            inp = fdtr[G.encoder_inputs][:,0,0]
            pred = tr_pred[:-1,0,0]
            plt.plot(inp, label='input')
            plt.plot(pred, label='predicted')
            plt.legend(loc='upper right')
            plt.show(block=False)            

except KeyboardInterrupt:
    print('training interrupted')


plt.plot(tr_loss_track, label='tr_loss_track')
plt.show(block=False)
plt.plot(inf_loss_track, label='inf_loss_track')
plt.show(block=False)
time_tr_end = time.time()
print('inf_loss {:.4f} after {} examples (batch_size={}). Tot time: {}'.format(inf_loss_track[-1], len(inf_loss_track)*batch_size, batch_size, time_tr_end-time_tr_start))

# ================= TEST =================
print('********** TEST **********')
fdts = {G.encoder_inputs: test_data,
        G.encoder_inputs_length: [test_data.shape[0] for _ in range(ts_data_samples)],
        G.decoder_outputs: test_targets}
ts_pred, ts_loss = sess.run([G.inf_outputs, G.inf_loss], fdts)
print('Test MSE: %.3f' % (ts_loss))

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