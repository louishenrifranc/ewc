import tensorflow as tf
from tqdm import trange
import model
import pickle
import os
import utils
import numpy as np

# Global variables
flags = tf.app.flags
flags.DEFINE_integer("nb_epochs", 10000, "Epoch to train [100 000]")
flags.DEFINE_float("learning_rate", 0.0003, "Learning rate of for adam [0.0001")
flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
flags.DEFINE_integer("batch_size", 16, "The size of the batch [64]")

flags.DEFINE_integer("hidden_size", 256, "Hidden size of RNN cell [256]")
flags.DEFINE_integer("embedding_size", 128, "Symbol embedding size")
flags.DEFINE_integer("max_sequence_length", 100, "Maximum sequence length")

flags.DEFINE_integer("vocab_size", 55, "Size of the vocabulary")

cfg = flags.FLAGS

if __name__ == '__main__':
    with open(os.path.join('Data', 'MovieQA', 'idx_to_chars.pkl'), 'rb') as f:
        idx_to_char = pickle.load(f)

    buckets = [(100, 100)]

    # To make things simpler, make a single bucket of size (100, 100)
    model = model.Seq2Seq(buckets, cfg)
    model.build()

    # Interactive session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter('logs/',
                                           graph=sess.graph,
                                           flush_secs=20)

    ########################### First experiment ######################################
    # In this experiment we will train a model to predict answer from question, given
    # that both question and answer are in English
    with open(os.path.join('Data', 'MovieQA', 'QA_Pairs_Chars_Buckets.pkl'), 'rb') as f:
        data_exp1 = pickle.load(f)
        qa_pairs = data_exp1['qa_pairs']
        bucket_lengths = data_exp1['bucket_lengths']
        sentences_exp1 = utils.parse_data_for_ewc_experiment(qa_pairs, bucket_lengths, cfg.max_sequence_length)
        del data_exp1, bucket_lengths

    utils.train_task(sess, model, cfg.nb_epochs, sentences_exp1, [sentences_exp1], cfg.batch_size, buckets, lams=[0])

    ########################### Second experiment ######################################
    # In this experiment we will train the same model on a new dataset where question and
    # answers are in French
    with open(os.path.join('Data', 'Messenger', 'QA_Pairs_Chars_Buckets_FJ.pkl'), 'rb') as f:
        data_exp2 = pickle.load(f)
        qa_pairs = data_exp1['qa_pairs']
        bucket_lengths = data_exp1['bucket_lengths']
        sentences_exp2 = utils.parse_data_for_ewc_experiment(qa_pairs, bucket_lengths, cfg.max_sequence_length)
        del data_exp2, bucket_lengths
