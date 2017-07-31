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
num_epochs = 1000
tr_data_samples = 2000
vs_data_samples = 1000
ts_data_samples = 1000

alternate_train = 0 # train both on inferenced and external inputs

config = dict(cell_type = 'LSTM',
              num_layers = 1,
              hidden_units = 5,
              input_dim = 1,
              bidirect = 1,
              max_gradient_norm = 15, # TODO: check variance (Bengio)
              learning_rate = 0.001,
              EOS = 0,
              last_layer_state_only = 0,
              reverse_input = 0)
print(config)

# ================= DATASET =================
#training data
TS_gen = getLorentz()
training_data = np.asarray([next(TS_gen) for _ in range(tr_data_samples)])
training_data = preprocessing.scale(training_data,axis=1) # standardize the data
training_data = np.expand_dims(training_data,-1)
training_data = np.transpose(training_data,axes=[1,0,2]) # time_major=True
training_targets = training_data

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

# revert inputs
if config['reverse_input']:
    training_data = training_data[::-1,:,:]
    valid_data = valid_data[::-1,:,:]
    test_data = test_data[::-1,:,:]

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
max_batches = training_data.shape[1]//batch_size
teach_loss_track = []
inf_loss_track = []

try:
    for ep in range(num_epochs):
        
        # shuffle training data
        idx = np.random.permutation(training_data.shape[1])
        training_data = training_data[:,idx,:] 
        training_targets = training_targets[:,idx,:] 
        
#        if ep % 10 == 0 and alternate_train: 
        if False: # train on teacher
            for batch in range(max_batches):
                
                fdtr = {G.encoder_inputs: training_data[:,(batch)*batch_size:(batch+1)*batch_size,:],
                        G.encoder_inputs_length: [training_data.shape[0] for _ in range(batch_size)],
                        G.decoder_outputs: training_targets[:,(batch)*batch_size:(batch+1)*batch_size,:]}
                
                _, inf_loss, teach_loss = sess.run([G.teach_update_step, G.inf_loss, G.teach_loss], fdtr) 
                inf_loss_track.append(inf_loss)
                teach_loss_track.append(teach_loss)                          
                
        elif False: # train on inference
            for batch in range(max_batches):
                
                fdtr = {G.encoder_inputs: training_data[:,(batch)*batch_size:(batch+1)*batch_size,:],
                        G.encoder_inputs_length: [training_data.shape[0] for _ in range(batch_size)],
                        G.decoder_outputs: training_targets[:,(batch)*batch_size:(batch+1)*batch_size,:]}
                
                _, inf_loss, teach_loss = sess.run([G.inf_update_step, G.inf_loss, G.teach_loss], fdtr)     
                inf_loss_track.append(inf_loss)
                teach_loss_track.append(teach_loss) 
                
        elif True: # train on inference + teacher
            for batch in range(max_batches):
                
                fdtr = {G.encoder_inputs: training_data[:,(batch)*batch_size:(batch+1)*batch_size,:],
                        G.encoder_inputs_length: [training_data.shape[0] for _ in range(batch_size)],
                        G.decoder_outputs: training_targets[:,(batch)*batch_size:(batch+1)*batch_size,:]}
                
                _, inf_loss, teach_loss = sess.run([G.teach_inf_update_step, G.inf_loss, G.teach_loss], fdtr)     
                inf_loss_track.append(inf_loss)    
                teach_loss_track.append(teach_loss) 
            
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
            
            fdtr = {G.encoder_inputs: training_data,
                    G.encoder_inputs_length: [training_data.shape[0] for _ in range(tr_data_samples)],
                    G.decoder_outputs: training_targets}
            inf_pred, inf_loss = sess.run([G.inf_outputs, G.inf_loss], fdtr)
            print('Train MSE (INF) {}'.format(inf_loss))
            # plot
            inp = fdtr[G.encoder_inputs][:,0,0]
            pred = inf_pred[:-1,0,0]
            plt.plot(inp, label='input')
            plt.plot(pred, label='predicted')
            plt.legend(loc='upper right')
            plt.show(block=False)  
            
            fdtr = {G.encoder_inputs: training_data,
                    G.encoder_inputs_length: [training_data.shape[0] for _ in range(tr_data_samples)],
                    G.decoder_outputs: training_targets}
            teach_pred, teach_loss = sess.run([G.teach_outputs, G.teach_loss], fdtr)
            print('Train MSE (EXT) {}'.format(teach_loss))           
            # plot
            inp = fdtr[G.encoder_inputs][:,0,0]
            pred = teach_pred[:-1,0,0]
            plt.plot(inp, label='input')
            plt.plot(pred, label='predicted')
            plt.legend(loc='upper right')
            plt.show(block=False)            

except KeyboardInterrupt:
    print('training interrupted')


plt.plot(teach_loss_track, label='teach_loss_track')
plt.legend(loc='upper right')
plt.show(block=False)
plt.plot(inf_loss_track, label='inf_loss_track')
plt.legend(loc='upper right')
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