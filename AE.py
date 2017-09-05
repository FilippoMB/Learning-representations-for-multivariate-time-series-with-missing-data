import tensorflow as tf
import argparse, sys
from TS_datasets import *
import time
import numpy as np
import matplotlib.pyplot as plt
from utils import classify_with_knn, interp_data, mse_and_corr, dim_reduction_plot
import math

dim_red = 0
plot_on = 0
interp_on = 0
tied_weights = 0

# parse input data
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_id", default='ODE2', help="ID of the dataset (SYNTH, ECG, JAP, etc..)", type=str)
parser.add_argument("--code_size", default=10, help="size of the code", type=int)
parser.add_argument("--w_reg", default=0.002, help="weight of the regularization in the loss function", type=float)
parser.add_argument("--num_epochs", default=5000, help="number of epochs in training", type=int)
parser.add_argument("--batch_size", default=25, help="number of samples in each batch", type=int)
parser.add_argument("--max_gradient_norm", default=1.0, help="max gradient norm for gradient clipping", type=float)
parser.add_argument("--learning_rate", default=0.001, help="Adam initial learning rate", type=float)
parser.add_argument("--hidden_size", default=30, help="size of the code", type=int)
args = parser.parse_args()
print(args)

# ================= DATASET =================
if args.dataset_id == 'SYNTH':
    (train_data, train_labels, train_len, _, _,
        valid_data, _, valid_len, _, _,
        test_data_orig, test_labels, test_len, _, _) = getSynthData(name='Lorentz', tr_data_samples=2000, 
                                                                     vs_data_samples=2000, 
                                                                     ts_data_samples=2000)
    
elif args.dataset_id == 'ECG':
    (train_data, train_labels, train_len, _, _,
        valid_data, _, valid_len, _, _,
        test_data_orig, test_labels, test_len, _, _) = getECGData()
       
elif args.dataset_id == 'JAP':        
    (train_data, train_labels, train_len, _, _,
        valid_data, _, valid_len, _, _,
        test_data_orig, test_labels, test_len, _, _) = getJapDataFull()

elif args.dataset_id == 'CHAR':        
    (train_data, train_labels, train_len, _, _,
        valid_data, _, valid_len, _, _,
        test_data_orig, test_labels, test_len, _, _) = getCharDataFull()

elif args.dataset_id == 'LIB':        
    (train_data, train_labels, train_len, _, _,
        valid_data, _, valid_len, _, _,
        test_data_orig, test_labels, test_len, _, _) = getLibras()    
    
elif args.dataset_id == 'WAF':        
    (train_data, train_labels, train_len, _, _,
        valid_data, _, valid_len, _, _,
        test_data_orig, test_labels, test_len, _, _) = getWafer()      
    
elif args.dataset_id == 'SIN':        
    (train_data, train_labels, train_len, _, _,
        valid_data, _, valid_len, _, _,
        test_data_orig, test_labels, test_len, _, _) = getSins()
    
elif args.dataset_id == 'MSO':        
    (train_data, train_labels, train_len, _, _,
        valid_data, _, valid_len, _, _,
        test_data_orig, test_labels, test_len, _, _) = getMSO()  
    
elif args.dataset_id == 'ODE':        
    (train_data, train_labels, train_len, _, _,
        valid_data, _, valid_len, _, _,
        test_data_orig, test_labels, test_len, _, _) = getODE() 
    
elif args.dataset_id == 'ODE2':        
    (train_data, train_labels, train_len, _, _,
        valid_data, _, valid_len, _, _,
        test_data_orig, test_labels, test_len, _, _) = getODE_mc() 
    
else:
    sys.exit('Invalid dataset_id')   
       
# interpolation
if np.min(train_len) < np.max(train_len) and interp_on:
    print('-- Data Interpolation --')
    train_data = interp_data(train_data, train_len)
    valid_data = interp_data(valid_data, valid_len)
    test_data = interp_data(test_data_orig, test_len)
else:
    test_data = test_data_orig

# transpose and reshape [T, N, V] --> [N, T, V] --> [N, T*V]
train_data = np.transpose(train_data,axes=[1,0,2])
train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1]*train_data.shape[2]))
valid_data = np.transpose(valid_data,axes=[1,0,2])
valid_data = np.reshape(valid_data, (valid_data.shape[0], valid_data.shape[1]*valid_data.shape[2]))
test_data = np.transpose(test_data,axes=[1,0,2])
test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1]*test_data.shape[2]))    

print('\n**** Processing {}: Tr{}, Vs{}, Ts{} ****\n'.format(args.dataset_id, train_data.shape, valid_data.shape, test_data.shape))

input_length = train_data.shape[1] # same for all inputs
# ================= GRAPH =================

# init session
tf.reset_default_graph() # needed when working with iPython
sess = tf.Session()

# placeholders
encoder_inputs = tf.placeholder(shape=(None,input_length), dtype=tf.float32, name='encoder_inputs')

## encoder
#with tf.variable_scope("Encoder"):
#    hidden_1 = tf.contrib.layers.fully_connected(encoder_inputs,
#                                           num_outputs=args.hidden_size,
#                                           activation_fn=tf.nn.tanh,
#                                           )
#    
#    code = tf.contrib.layers.fully_connected(hidden_1,
#                                           num_outputs=args.code_size,
#                                           activation_fn=tf.nn.tanh,
#                                           )
#      
#    
## decoder
#with tf.variable_scope("Decoder"):
#    hidden_2 = tf.contrib.layers.fully_connected(code,
#                                           num_outputs=args.hidden_size,
#                                           activation_fn=tf.nn.tanh,
#                                           )
#    
#    dec_out = tf.contrib.layers.fully_connected(hidden_2,
#                                           num_outputs=input_length,
#                                           activation_fn=None,
#                                           )

# encoder
We1 = tf.Variable(tf.random_uniform((input_length, args.hidden_size), -1.0 / math.sqrt(input_length), 1.0 / math.sqrt(input_length)))
We2 = tf.Variable(tf.random_uniform((args.hidden_size, args.code_size), -1.0 / math.sqrt(args.hidden_size), 1.0 / math.sqrt(args.hidden_size)))

be1 = tf.Variable(tf.zeros([args.hidden_size]))
be2 = tf.Variable(tf.zeros([args.code_size]))

hidden_1 = tf.nn.tanh(tf.matmul(encoder_inputs, We1) + be1)
code = tf.nn.tanh(tf.matmul(hidden_1, We2) + be2)

# decoder
if tied_weights:
    Wd1 = tf.transpose(We2)
    Wd2 = tf.transpose(We1)
else:
    Wd1 = tf.Variable(tf.random_uniform((args.code_size, args.hidden_size), -1.0 / math.sqrt(args.code_size), 1.0 / math.sqrt(args.code_size)))
    Wd2 = tf.Variable(tf.random_uniform((args.hidden_size, input_length), -1.0 / math.sqrt(args.hidden_size), 1.0 / math.sqrt(args.hidden_size)))
    
bd1 = tf.Variable(tf.zeros([args.hidden_size]))  
bd2 = tf.Variable(tf.zeros([input_length])) 

hidden_2 = tf.nn.tanh(tf.matmul(code, Wd1) + bd1)
dec_out = tf.matmul(hidden_2, Wd2) + bd2

# ----- LOSS --------

# reconstruction loss    
parameters = tf.trainable_variables()
optimizer = tf.train.AdamOptimizer(args.learning_rate)
reconstruct_loss = tf.losses.mean_squared_error(labels=dec_out, predictions=encoder_inputs)

# L2 loss
reg_loss = 0
for tf_var in tf.trainable_variables():
    reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
        
tot_loss = reconstruct_loss + args.w_reg*reg_loss

# Calculate and clip gradients
gradients = tf.gradients(tot_loss, parameters)
clipped_gradients, _ = tf.clip_by_global_norm(gradients, args.max_gradient_norm)
update_step = optimizer.apply_gradients(zip(clipped_gradients, parameters))

sess.run(tf.global_variables_initializer())

# trainable parameters count
total_parameters = 0
for variable in tf.trainable_variables():
    shape = variable.get_shape()
    variable_parametes = 1
    for dim in shape:
        variable_parametes *= dim.value
    total_parameters += variable_parametes
print('Total parameters: {}'.format(total_parameters))

# ============= TENSORBOARD =============             
mean_grads = tf.reduce_mean([tf.reduce_mean(grad) for grad in gradients])
tf.summary.scalar('mean_grads', mean_grads)
tf.summary.scalar('reconstruct_loss', reconstruct_loss)
tvars = tf.trainable_variables()
for tvar in tvars:
    tf.summary.histogram(tvar.name.replace(':','_'), tvar)
merged_summary = tf.summary.merge_all()

# ================= TRAINING =================

# initialize training stuff
batch_size = args.batch_size
time_tr_start = time.time()
max_batches = train_data.shape[0]//batch_size
loss_track = []
min_vs_loss = np.infty
model_name = "/tmp/tkae_models/m_"+str(time.strftime("%Y%m%d-%H%M%S"))+".ckpt"
#train_writer = tf.summary.FileWriter('/tmp/tensorboard', graph=sess.graph)
saver = tf.train.Saver()

try:
    for ep in range(args.num_epochs):
        
        # shuffle training data
        idx = np.random.permutation(train_data.shape[0])
        train_data_s = train_data[idx,:] 
        
        for batch in range(max_batches):
            
            fdtr = {encoder_inputs: train_data_s[(batch)*batch_size:(batch+1)*batch_size,:]}           
            _,train_loss = sess.run([update_step, reconstruct_loss], fdtr)    
            loss_track.append(train_loss)
            
        # check training progress on the validations set    
        if ep % 100 == 0:            
            print('Ep: {}'.format(ep))
            
            fdvs = {encoder_inputs: valid_data}
            outvs, lossvs = sess.run([dec_out, reconstruct_loss], fdvs) #summary, merged_summary
            #train_writer.add_summary(summary, ep)
            print('VS loss=%.3f -- TR min_loss=.%3f'%(lossvs, np.min(loss_track)))     
            
            # Save model yielding best results on validation
            if lossvs < min_vs_loss:
                min_vs_loss = lossvs
                tf.add_to_collection("encoder_inputs",encoder_inputs)
                tf.add_to_collection("dec_out",dec_out)
                tf.add_to_collection("reconstruct_loss",reconstruct_loss)
                save_path = saver.save(sess, model_name)        
                                           
            # plot a random ts from the validation set
            if plot_on:
                plot_idx1 = np.random.randint(low=0,high=valid_data.shape[0])
                target = valid_data[plot_idx1,:]
                pred = outvs[plot_idx1,:-1]
                plt.plot(target, label='target')
                plt.plot(pred, label='pred')
                plt.legend(loc='upper right')
                plt.show(block=False)  
                                                    
except KeyboardInterrupt:
    print('training interrupted')

if plot_on:
    plt.plot(loss_track, label='loss_track')
    plt.legend(loc='upper right')
    plt.show(block=False)
    
time_tr_end = time.time()
print('Tot training time: {}'.format((time_tr_end-time_tr_start)//60) )

# ================= TEST =================
print('************ TEST ************ \n>>restoring from:'+model_name+'<<')

tf.reset_default_graph() # be sure that correct weights are loaded
saver.restore(sess, model_name)

tr_code = sess.run(code, {encoder_inputs: train_data})
pred, pred_loss, ts_code = sess.run([dec_out, reconstruct_loss, code], {encoder_inputs: test_data})
print('Test loss: %.3f'%(np.mean((pred-test_data)**2)))

if plot_on:
    plot_idx1 = np.random.randint(low=0,high=test_data.shape[0])
    target = test_data[plot_idx1,:]
    ts_out = pred[plot_idx1,:]
    plt.plot(target, label='target')
    plt.plot(ts_out, label='pred')
    plt.legend(loc='upper right')
    plt.show(block=False)  
    
    plt.scatter(ts_code[:,0],ts_code[:,1],c=test_labels,marker='.',linewidths = 0,cmap='Paired')
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.show()

# reverse transformations
pred = np.reshape(pred, (test_data_orig.shape[1], test_data_orig.shape[0], test_data_orig.shape[2]))
pred = np.transpose(pred,axes=[1,0,2])
test_data = test_data_orig

if np.min(train_len) < np.max(train_len) and interp_on:
    print('-- Reverse Interpolation --')
    pred = interp_data(pred, test_len, restore=True)

# MSE and corr
test_mse, test_corr = mse_and_corr(test_data, pred, test_len)
print('Test MSE: %.3f\nTest Pearson correlation: %.3f'%(test_mse, test_corr))

# kNN classification on the codes
acc = classify_with_knn(tr_code, train_labels[:, 0], ts_code, test_labels[:, 0])
print('kNN acc: {}'.format(acc))

# dim reduction plots
if dim_red:
    dim_reduction_plot(ts_code, test_labels, 1)

#train_writer.close()
sess.close()
