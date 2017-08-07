import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from TS_datasets import getSynthData, getECGData
from utils import dim_reduction_plot

# load data
training_data, training_labels, valid_data, valid_labels, test_data, test_labels = getECGData()
training_targets, valid_targets, test_targets = training_data, valid_data, test_data

graph_name = "20170807-102417"

with tf.Session() as sess:
    
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

    # evaluate graph on test set
    fdts = {encoder_inputs: test_data,
        encoder_inputs_length: [test_data.shape[0] for _ in range(test_data.shape[1])],
        decoder_outputs: test_targets}
    ts_pred, ts_loss, ts_context = sess.run([inf_outputs, inf_loss, context_vector], fdts)
    print('Test MSE: %.3f' % (ts_loss))
    
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