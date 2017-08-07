import tensorflow as tf
from TS_datasets import getSynthData, getECGData

training_data, training_labels, valid_data, valid_labels, test_data, test_labels = getECGData()
training_targets, valid_targets, test_targets = training_data, valid_data, test_data

#graph_name = "/tmp/metagraph/graph_G1"
graph_name = "/tmp/tkae_model_20170807-004216.ckpt"

with tf.Session() as sess:
    graph = tf.get_default_graph()
    new_saver = tf.train.import_meta_graph(graph_name+".meta")
    new_saver.restore(sess, graph_name)
    
    encoder_inputs = tf.get_collection("encoder_inputs")[0]
    encoder_inputs_length = tf.get_collection("encoder_inputs_length")[0]
    decoder_outputs = tf.get_collection("decoder_outputs")[0]
    inf_outputs = tf.get_collection("inf_outputs")[0]
    inf_loss = tf.get_collection("inf_loss")[0]
    context_vector = tf.get_collection("context_vector")[0]

#    encoder_inputs = graph.get_operation_by_name('encoder_inputs').outputs[0]
#    encoder_inputs_length = graph.get_operation_by_name('encoder_inputs_length').outputs[0]
#    decoder_outputs =  graph.get_operation_by_name('decoder_outputs').outputs[0]
#    inf_outputs =  graph.get_operation_by_name('inf_outputs').outputs[0]
#    inf_loss =  graph.get_operation_by_name('inf_loss').outputs[0]
#    context_vector =  graph.get_operation_by_name('context_vector').outputs[0]
        
    fdts = {encoder_inputs: test_data,
        encoder_inputs_length: [test_data.shape[0] for _ in range(test_data.shape[1])],
        decoder_outputs: test_targets}
    ts_pred, ts_loss, ts_context = sess.run([inf_outputs, inf_loss, context_vector], fdts)
    print('Test MSE: %.3f' % (ts_loss))