import tensorflow as tf
import numpy as np


def get_next_batch(data, batch_size, buckets):
    indices = np.random.choice(len(data), size=batch_size)
    pairs = np.array(data)[indices]

    q_pads = np.zeros([batch_size, buckets[0][0]])
    a_pads = np.zeros([batch_size, buckets[0][1]])

    for i, (q, a) in enumerate(pairs):
        q_pads[i][:q.shape[0]] = q
        a_pads[i][:a.shape[0]] = a

    return q_pads, a_pads


########################### Merge all sentences ######################################
def train_task(sess, model, nb_epochs, training_data, testing_data, lambdas, batch_size, buckets,
               restore_weights=False):
    for l in range(len(lambdas)):
        if restore_weights:
            sess.run(model.restore_sticky_weights)

    for nb_iter in range(nb_epochs):
        q_s, a_s = get_next_batch(training_data, batch_size, buckets)
        out = model.forward_with_feed_dict(sess, q_s, a_s, is_training=True)
    """
    summary_gen_loss = tf.Summary(value=[
                tf.Summary.Value(tag="gen_loss", simple_value=extras[0]),
])
 summary_writer.add_summary(summary_dis_loss, global_step=current_iter)
    """


########################### Merge all sentences ######################################
def parse_data_for_ewc_experiment(qa_pairs, bucket_lengths, max_sequence_length):
    all_sentences = []
    for i, (enc_length, dec_length) in enumerate(bucket_lengths):
        if enc_length < max_sequence_length and dec_length < max_sequence_length:
            for pair_sentence in qa_pairs[i]:
                all_sentences.append(pair_sentence)
    return all_sentences
