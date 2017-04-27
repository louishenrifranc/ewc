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

flags.DEFINE_integer("hidden_size", 128, "Hidden size of RNN cell [256]")
flags.DEFINE_integer("embedding_size", 64, "Symbol embedding size")
flags.DEFINE_integer("max_sequence_length", 50, "Maximum sequence length")
flags.DEFINE_integer("num_layers", 2, "Num of layers [3]")
flags.DEFINE_integer("vocab_size", 55, "Size of the vocabulary")
flags.DEFINE_float("validation_percent", 0.15, "Percentage for the testing set")

# TODO: check if max_gradient_norm is needed

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

        # Split training and testing set
        indexes = np.arange(len(sentences_exp1))
        np.random.shuffle(indexes)

        indexes = np.split(indexes, [int((1 - cfg.validation_percent) * len(indexes))])

        train_exp1 = sentences_exp1[indexes[0]]
        test_exp1 = sentences_exp1[indexes[1]]
        # __ CLEAN DICTIONARY __
        del data_exp1, bucket_lengths, indexes, sentences_exp1

    utils.train_task(sess=sess,
                     model=model,
                     nb_epochs=cfg.nb_epochs,
                     training_data=train_exp1,
                     testing_datas=[test_exp1],
                     lambdas=[0],
                     batch_size=cfg.batch_size,
                     buckets=buckets,
                     summary_writer=summary_writer,
                     exp_name="experience1")

    sess.run([model.update_fisher, model.update_sticky_weights])

    # __ CLEAN DICTIONARY __
    del train_exp1

    ########################### Second experiment ######################################
    # In this experiment we will train the same model on a new dataset where question and
    # answers are in French
    with open(os.path.join('Data', 'Messenger', 'QA_Pairs_Chars_Buckets_FJ.pkl'), 'rb') as f:
        data_exp2 = pickle.load(f)
        qa_pairs = data_exp2['qa_pairs']
        bucket_lengths = data_exp2['bucket_lengths']
        sentences_exp2 = utils.parse_data_for_ewc_experiment(qa_pairs, bucket_lengths, cfg.max_sequence_length)

        # Split training and testing set
        indexes = np.arange(len(sentences_exp2))
        np.random.shuffle(indexes)

        indexes = np.split(indexes, [int((1 - cfg.validation_percent) * len(indexes))])

        train_exp2 = sentences_exp2[indexes[0]]
        test_exp2 = sentences_exp2[indexes[1]]

        # __ CLEAN DICTIONARY __
        del data_exp2, bucket_lengths, indexes, sentences_exp2

    utils.train_task(sess=sess,
                     model=model,
                     nb_epochs=cfg.nb_epochs,
                     training_data=train_exp2,
                     testing_datas=[test_exp1, test_exp2],
                     lambdas=[0, 15],
                     batch_size=cfg.batch_size,
                     buckets=buckets,
                     summary_writer=summary_writer,
                     exp_name="experience2",
                     restore_weights=True)
