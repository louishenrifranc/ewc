import tensorflow as tf
import model
import pickle
import os
import utils
from termcolor import cprint
import random

debug = False
# Global variables
flags = tf.app.flags
flags.DEFINE_integer("nb_epochs", 50000, "Epoch to train [100 000]")
flags.DEFINE_float("learning_rate", 0.0003, "Learning rate of for adam [0.0001")
flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
flags.DEFINE_integer("batch_size", 16, "The size of the batch [64]")

flags.DEFINE_integer("hidden_size", 256, "Hidden size of RNN cell [256]")
flags.DEFINE_integer("embedding_size", 128, "Symbol embedding size")
flags.DEFINE_integer("max_sequence_length", 100, "Maximum sequence length")
flags.DEFINE_integer("num_layers", 3, "Num of layers [3]")
flags.DEFINE_integer("vocab_size", 55, "Size of the vocabulary")
flags.DEFINE_float("validation_percent", 0.15, "Percentage for the testing set")

cfg = flags.FLAGS

# Debug mode.
if debug:
    cfg.hidden_size = 32
    cfg.num_layers = 1
    cfg.nb_epochs = 200

if __name__ == '__main__':
    if tf.gfile.Exists("logs/"):
        try:
            os.remove("logs/")
        except:
            cprint("[!] Can't remove logs old folder")

    # Dictionnary of chars. Convert idx to char
    with open(os.path.join('Data', 'MovieQA', 'idx_to_chars.pkl'), 'rb') as f:
        idx_to_char = pickle.load(f)

    # In the experiment, we will use a single bucket of length (100, 100)
    buckets = [(200, 200)]

    # Create a Seq2seq model. This model holds all the operation for forwarding, backwarding signal
    # in the neural network, plus computing the Fisher matrix and saving weights for every tasks
    model = model.Seq2Seq(buckets, cfg)
    model.build()

    # Interactive session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # Create an object to save curves and histograms of variables 
    summary_writer = tf.summary.FileWriter('logs/',
                                           graph=sess.graph,
                                           flush_secs=20)

    # Create an object to save a model
    saver = tf.train.Saver()

    ########################### First experiment ######################################
    # In this experiment we will train a model to predict answer from question, given
    # that both question and answer are in English. Training is done with a standalone
    # stocastic gradient descent.
    with open(os.path.join('Data', 'MovieQA', 'QA_Pairs_Chars_Buckets.pkl'), 'rb') as f:
        cprint("[*] Loading dataset for experiment 1", color="yellow")

        # Load parsed data
        data_exp1 = pickle.load(f)
        qa_pairs = data_exp1['qa_pairs']
        bucket_lengths = data_exp1['bucket_lengths']

        # Explicit parse for the experimentation. All buckets of size under $buckets are joined in a same list
        sentences_exp1 = utils.parse_data_for_ewc_experiment(qa_pairs, bucket_lengths, cfg.max_sequence_length)

        # Split training and testing set
        random.shuffle(sentences_exp1)
        train_exp1 = sentences_exp1[:len(sentences_exp1) // 2]
        test_exp1 = sentences_exp1[len(sentences_exp1) // 2:]

        # __ CLEAN DICTIONARY __
        del data_exp1, bucket_lengths, sentences_exp1
        cprint("[*] Loaded", color="green")

    cprint("[*] Starting Experiment 1", color="yellow")
    utils.train_task(sess=sess,
                     model=model,
                     nb_epochs=cfg.nb_epochs,
                     training_data=train_exp1,
                     testing_datas=[test_exp1],
                     lambdas=[0],
                     batch_size=cfg.batch_size,
                     buckets=buckets,
                     summary_writer=summary_writer,
                     saver=saver,
                     exp_name="experience1")

    cprint("[*] Experiment 1 over", color="green")

    cprint("[*] Compute Fisher matrix and saved all weights", color="yellow")
    sess.run([model.update_fisher, model.update_sticky_weights])

    # __ CLEAN DICTIONARY __
    del train_exp1

    ########################### Second experiment ######################################
    # In this experiment we will train the same model on a new dataset where question and
    # answers are in French. We will train in two different ways. 
    # The first way to train it is in plain stocastic gradient descent, while the second
    # experiment is with the ewc quadratic constraint. 
    # During testing, we tried two datasets, the testing set from the first experiment
    # and the testing set of the new experiment. Experiments should show an increase in the 
    # testing loss of the first dataset.  
    with open(os.path.join('Data', 'Messenger', 'QA_Pairs_Chars_Buckets.pkl'), 'rb') as f:
        cprint("[*] Loading dataset for experiment 2", color="yellow")

        # Loaded parse data
        data_exp2 = pickle.load(f)
        qa_pairs = data_exp2['qa_pairs']
        bucket_lengths = data_exp2['bucket_lengths']

        # Explicit parse for the experimentation. All buckets of size under $buckets are joined in a same list
        sentences_exp2 = utils.parse_data_for_ewc_experiment(qa_pairs, bucket_lengths, cfg.max_sequence_length)

        # Split training and testing set
        random.shuffle(sentences_exp2)
        train_exp2 = sentences_exp2[:len(sentences_exp2) // 2]
        test_exp2 = sentences_exp2[len(sentences_exp2) // 2:]

        # __ CLEAN DICTIONARY __
        del data_exp2, bucket_lengths, sentences_exp2

    cprint("[*] Starting Experiment 2", color="yellow")
    utils.train_task(sess=sess,
                     model=model,
                     nb_epochs=cfg.nb_epochs,
                     training_data=train_exp2,
                     testing_datas=[test_exp1, test_exp2],
                     lambdas=[0, 25],
                     batch_size=cfg.batch_size,
                     buckets=buckets,
                     summary_writer=summary_writer,
                     exp_name="experience2",
                     saver=saver,
                     restore_weights=True)
    cprint("[*] Experiment 2 over", color="green")

    cprint("[!] END [!]", color="red")
