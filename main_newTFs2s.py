from gen_seq2seq_TF import s2sModel
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)
import time, sys
import tensorflow as tf
from TS_datasets import getSynthData, getECGData
from scipy.stats.stats import pearsonr
from utils import dim_reduction_plot

# Hyperparams
batch_size = 200
num_epochs = 5000

alternate_train = 0 # train both on inferenced and external inputs

config = dict(cell_type = 'LSTM',
              num_layers = 1,
              hidden_units = 5,
              input_dim = 1,
              bidirect = 1,
              max_gradient_norm = 5, # TODO: check variance (Bengio)
              learning_rate = 0.001,
              EOS = 0,
              last_layer_state_only = 0,
              reverse_input = 0,
              training_mode = 'inf')
print(config)

# ================= DATASET =================
#training_data, training_targets, valid_data, valid_targets, test_data, test_targets = getSynthData(
#        name='Lorentz', tr_data_samples=2000, vs_data_samples=2000, ts_data_samples=2000)

training_data, training_labels, valid_data, valid_labels, test_data, test_labels = getECGData()
training_targets, valid_targets, test_targets = training_data, valid_data, test_data
del training_labels, valid_labels

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

train_writer = tf.summary.FileWriter('/tmp/tensorboard', graph=sess.graph)
saver = tf.train.Saver()
# ================= DEBUG =================

#context = sess.run(G.context_vector, {G.encoder_inputs: training_data, G.encoder_inputs_length: [training_data.shape[0] for _ in range(training_data.shape[1])]} )  

# ================= TRAINING =================

# initialize training stuff
time_tr_start = time.time()
max_batches = training_data.shape[1]//batch_size
teach_loss_track = []
inf_loss_track = []
min_vs_loss = np.infty
model_name = "/tmp/tkae_model_"+str(np.random.rand())+".ckpt"

try:
    for ep in range(num_epochs):
        
        # shuffle training data
        idx = np.random.permutation(training_data.shape[1])
        training_data = training_data[:,idx,:] 
        training_targets = training_targets[:,idx,:] 
        
        for batch in range(max_batches):
            
            fdtr = {G.encoder_inputs: training_data[:,(batch)*batch_size:(batch+1)*batch_size,:],
                    G.encoder_inputs_length: [training_data.shape[0] for _ in range(batch_size)],
                    G.decoder_outputs: training_targets[:,(batch)*batch_size:(batch+1)*batch_size,:]}  
            
            if config['training_mode'] == 'teach': # train on teacher
                _, inf_loss, teach_loss = sess.run([G.teach_update_step, G.inf_loss, G.teach_loss], fdtr) 
            
            elif config['training_mode'] == 'inf': # train on inference 
                _, inf_loss, teach_loss = sess.run([G.inf_update_step, G.inf_loss, G.teach_loss], fdtr)     

            elif config['training_mode'] == 'sched': # scheduled training
                _, inf_loss, teach_loss = sess.run([G.sched_update_step, G.inf_loss, G.teach_loss], fdtr)     

            else:
                sys.exit('Invalid training mode')
            
            inf_loss_track.append(inf_loss)
            teach_loss_track.append(teach_loss) 
            
        # check how the training is going on the validations set    
        if ep % 100 == 0:   
            
            print('Ep: {}'.format(ep))
            
            # DEBUG
            fdtr = {G.encoder_inputs: training_data,
                    G.encoder_inputs_length: [training_data.shape[0] for _ in range(training_data.shape[1])],
                    G.decoder_outputs: training_targets}
            inf_loss, teach_loss = sess.run([G.inf_loss, G.teach_loss], fdtr)
            print('TR: inf_loss: %.3f, teach_loss: %.3f, min_inf_loss: %.3f'%(inf_loss, teach_loss, np.min(inf_loss_track)))
#            # plot
#            inp = fdtr[G.encoder_inputs][:,0,0]
#            inf_pred = inf_pred[:-1,0,0]
#            teach_pred = teach_pred[:-1,0,0]
#            plt.plot(inp, label='input')
#            plt.plot(inf_pred, label='inf_pred')
#            plt.plot(teach_pred, label='teach_pred')
#            plt.legend(loc='upper right')
#            plt.show(block=False)              
            
            fdvs = {G.encoder_inputs: valid_data,
                    G.encoder_inputs_length: [valid_data.shape[0] for _ in range(valid_data.shape[1])],
                    G.decoder_outputs: valid_targets}
            inf_outvs, inf_lossvs, teach_outvs, teach_lossvs = sess.run(
                    [G.inf_outputs, G.inf_loss, G.teach_outputs, G.teach_loss], fdvs)
#            train_writer.add_summary(summary_, ep)
            print('VS: inf_loss: %.3f, teach_loss: %.3f'%(inf_lossvs, teach_lossvs))
            
            # plot a random ts from the validation set
            plot_idx = np.random.randint(low=0,high=valid_data.shape[1]-1)
            inp = valid_data[:,plot_idx,0]
            inf_pred = inf_outvs[:-1,plot_idx,0]
            teach_pred = teach_outvs[:-1,plot_idx,0]
            plt.plot(inp, label='inp')
            plt.plot(inf_pred, label='inf')
            plt.plot(teach_pred, label='teach')
            plt.legend(loc='upper right')
            plt.show(block=False)  
            
            # Save model yielding best results on training
            if inf_lossvs < min_vs_loss:
                save_path = saver.save(sess, model_name)
                                        

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

tf.reset_default_graph() # just for debug
saver.restore(sess, model_name)

fdts = {G.encoder_inputs: test_data,
        G.encoder_inputs_length: [test_data.shape[0] for _ in range(test_data.shape[1])],
        G.decoder_outputs: test_targets}
ts_pred, ts_loss, ts_context = sess.run([G.inf_outputs, G.inf_loss, G.context_vector], fdts)
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

# dim reduction plots
dim_reduction_plot(ts_context, test_labels)

train_writer.close()
sess.close()