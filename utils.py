import tensorflow as tf
import numpy as np
from tqdm import trange


########################### Get a new batch ######################################
def get_next_batch(data, batch_size, buckets):
    indices = np.random.choice(len(data), size=batch_size)
    pairs = np.array(data)[indices]

    q_pads = np.zeros([batch_size, buckets[0][0]])
    a_pads = np.zeros([batch_size, buckets[0][1]])

    for i, (q, a) in enumerate(pairs):
        q_pads[i][:q.shape[0]] = q
        a_pads[i][:a.shape[0]] = a

    return q_pads, a_pads


########################### Train a model for a task ######################################
def train_task(sess, model,
               nb_epochs,
               training_data,
               testing_datas,
               lambdas,
               batch_size,
               buckets,
               summary_writer,
               exp_name,
               restore_weights=False):
    freq_test = 100
    for l in lambdas:
        if restore_weights:
            sess.run(model.restore_sticky_weights)

        for nb_iter in trange(nb_epochs):
            # Retrieve a training batch
            q_s, a_s = get_next_batch(training_data, batch_size, buckets)

            # Forward and backward pass
            out = model.forward_with_feed_dict(sess, q_s, a_s, is_training=True, ewc_loss_coeff=l)

            # Save the training loss
            train_loss = tf.Summary(value=[tf.Summary.Value(tag=exp_name + "_train", simple_value=out["losses"])])
            summary_writer.add_summary(train_loss, global_step=nb_iter)

            # Compute test loss
            if nb_iter % freq_test == 0:
                # Iterate over all testing sets
                for i, testing_data in enumerate(testing_datas):
                    q_s, a_s = get_next_batch(testing_data, batch_size, buckets)
                    out = model.forward_with_feed_dict(sess, q_s, a_s, is_training=False, ewc_loss_coeff=l)

                    test_loss = tf.Summary(
                        value=[tf.Summary.Value(tag=exp_name + "_test_task_" + str(i), simple_value=out["losses"])])
                    summary_writer.add_summary(test_loss, global_step=nb_iter)

                # Plot histogram for gradients
                summary_gradients = model.merged_summary_op.eval()
                summary_writer.add_summary(summary_gradients, nb_iter)


########################### Merge all sentences ######################################
def parse_data_for_ewc_experiment(qa_pairs, bucket_lengths, max_sequence_length):
    all_sentences = []
    for i, (enc_length, dec_length) in enumerate(bucket_lengths):
        if enc_length <= max_sequence_length and dec_length <= max_sequence_length:
            for pair_sentence in qa_pairs[i]:
                all_sentences.append(pair_sentence)
    return all_sentences
