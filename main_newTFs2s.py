from gen_seq2seq_TF import s2sModel
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)
import time, sys
import tensorflow as tf
from TS_datasets import getSynthData, getECGData
from scipy.stats.stats import pearsonr
from utils import dim_reduction_plot
import argparse

# Hyperparams
batch_size = 250
num_epochs = 5000
plot_on = 0 

# parse input data
parser = argparse.ArgumentParser()
parser.add_argument("--cell_type", default='LSTM', help="type of cell for encoder/decoder", type=str)
parser.add_argument("--num_layers", default=2, help="number of stacked layers in ecoder/decoder", type=int)
parser.add_argument("--hidden_units", default=5, help="number of hidden units in the encoder/decoder. If encoder is bidirectional, decoders units are doubled", type=int)
parser.add_argument("--input_dim", default=1, help="number of variables in the time series", type=int)
parser.add_argument("--bidirect", default=False, help="use an encoder which is bidirectional", type=bool)
parser.add_argument("--max_gradient_norm", default=1.0, help="max gradient norm for gradient clipping", type=float)
parser.add_argument("--learning_rate", default=0.001, help="Adam initial learning rate", type=float)
parser.add_argument("--EOS", default=0, help="special symbol for start/end of time series", type=float)
parser.add_argument("--last_layer_state_only", default=False, help="init decoder with last state of only last layer", type=bool)
parser.add_argument("--reverse_input", default=True, help="fed input reversed for training", type=bool)
parser.add_argument("--training_mode", default='sched', help="training mode of the decoder", type=str)
args = parser.parse_args()

config = dict(cell_type = args.cell_type,
              num_layers = args.num_layers,
              hidden_units = args.hidden_units,
              input_dim = args.input_dim,
              bidirect = args.bidirect,
              max_gradient_norm = args.max_gradient_norm, 
              learning_rate = args.learning_rate,
              EOS = args.EOS,
              last_layer_state_only = args.last_layer_state_only,
              reverse_input = args.reverse_input,
              training_mode = args.training_mode)
print(config)

# ================= DATASET =================
#training_data, training_targets, valid_data, valid_targets, test_data, test_targets = getSynthData(
#        name='Lorentz', tr_data_samples=2000, vs_data_samples=2000, ts_data_samples=2000)

training_data, training_labels, valid_data, valid_labels, test_data, test_labels = getECGData()
training_targets, valid_targets, test_targets = training_data, valid_data, test_data
del training_labels, valid_labels

# revert time
if config['reverse_input']:
    training_data = training_data[::-1,:,:]
    valid_data = valid_data[::-1,:,:]
    test_data = test_data[::-1,:,:]

# ================= GRAPH =================
tf.reset_default_graph()
sess = tf.Session()
G = s2sModel(config)
sess.run(tf.global_variables_initializer())

# ================= DEBUG =================

#mean_grad = sess.run(G.mean_grads, {G.encoder_inputs: training_data, G.encoder_inputs_length: [training_data.shape[0] for _ in range(training_data.shape[1])], G.decoder_outputs: training_targets} )  

# ================= TRAINING =================

# initialize training stuff
time_tr_start = time.time()
max_batches = training_data.shape[1]//batch_size
teach_loss_track = []
inf_loss_track = []
min_vs_loss = np.infty
model_name = "/tmp/tkae_model_"+str(time.strftime("%Y%m%d-%H%M%S"))+".ckpt"
train_writer = tf.summary.FileWriter('/tmp/tensorboard', graph=sess.graph)
saver = tf.train.Saver()

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
            
            elif config['training_mode'] == 'tinf': # teach+inf
                _, inf_loss, teach_loss = sess.run([G.teach_inf_update_step, G.inf_loss, G.teach_loss], fdtr)   

            else:
                sys.exit('Invalid training mode')
            
            inf_loss_track.append(inf_loss)
            teach_loss_track.append(teach_loss)
            
        # check how the training is going on the validations set    
        if ep % 100 == 0:   
            
            print('Ep: {}'.format(ep))
            
            # DEBUG
#            fdtr = {G.encoder_inputs: training_data,
#                    G.encoder_inputs_length: [training_data.shape[0] for _ in range(training_data.shape[1])],
#                    G.decoder_outputs: training_targets}
#            inf_loss, teach_loss = sess.run([G.inf_loss, G.teach_loss], fdtr)
#            print('TR: inf_loss: %.3f, teach_loss: %.3f, min_inf_loss: %.3f'%(inf_loss, teach_loss, np.min(inf_loss_track)))        
            
            fdvs = {G.encoder_inputs: valid_data,
                    G.encoder_inputs_length: [valid_data.shape[0] for _ in range(valid_data.shape[1])],
                    G.decoder_outputs: valid_targets}
            inf_outvs, inf_lossvs, teach_outvs, teach_lossvs, summary = sess.run([G.inf_outputs, G.inf_loss, G.teach_outputs, G.teach_loss, G.merged_summary], fdvs)
            train_writer.add_summary(summary, ep)
            print('VS: inf_loss=%.3f, teach_loss=%.3f -- TR: min_loss=.%3f'%(inf_lossvs, teach_lossvs, np.min(inf_loss_track)))
            
            # plot a random ts from the validation set
            if plot_on:
                plot_idx = np.random.randint(low=0,high=valid_targets.shape[1]-1)
                target = valid_targets[:,plot_idx,0]
                inf_pred = inf_outvs[:-1,plot_idx,0]
                teach_pred = teach_outvs[:-1,plot_idx,0]
                plt.plot(target, label='target')
                plt.plot(inf_pred, label='inf')
                plt.plot(teach_pred, label='teach')
                plt.legend(loc='upper right')
                plt.show(block=False)  
            
            # Save model yielding best results on validation
            if inf_lossvs < min_vs_loss:
                save_path = saver.save(sess, model_name)
                                        

except KeyboardInterrupt:
    print('training interrupted')

if plot_on:
    plt.plot(teach_loss_track, label='teach_loss_track')
    plt.legend(loc='upper right')
    plt.show(block=False)
    plt.plot(inf_loss_track, label='inf_loss_track')
    plt.legend(loc='upper right')
    plt.show(block=False)
time_tr_end = time.time()
print('Tot training time: {}'.format( (time_tr_end-time_tr_start)//60))

# ================= TEST =================
print('********** TEST **********')

sess.run(tf.global_variables_initializer()) # be sure correct weights are loaded
saver.restore(sess, model_name)

fdts = {G.encoder_inputs: test_data,
        G.encoder_inputs_length: [test_data.shape[0] for _ in range(test_data.shape[1])],
        G.decoder_outputs: test_targets}
ts_pred, ts_loss, ts_context = sess.run([G.inf_outputs, G.inf_loss, G.context_vector], fdts)
print('Test MSE: %.3f' % (ts_loss))

# plot 1
if plot_on:
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

train_writer.close()
sess.close()


with open('results','a') as f:
    f.write('cell: '+args.cell_type+
            ', n_layers: '+str(args.num_layers)+
            ', h_units: '+str(args.hidden_units)+
            ', bidir: '+str(args.bidirect)+
            ', max_grad: '+str(args.max_gradient_norm)+ 
            ', lr: '+str(args.learning_rate)+
            ', last_state: '+str(args.last_layer_state_only)+
            ', reverse_inp: '+str(args.reverse_input)+
            ', tr_mode: '+args.training_mode+ 
            ', time: '+str((time_tr_end-time_tr_start)//60)+
            ', MSE: '+str(ts_loss)+'\n')