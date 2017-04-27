import numpy as np
import tensorflow as tf
from termcolor import cprint
import tensorflow.contrib.legacy_seq2seq as seq2seq


class Seq2Seq(object):
    def __init__(self,
                 buckets,
                 cfg):
        """
        Seq2Seq model
        :param buckets: List of pairs
            Each pair correspond to (max_size_in_bucket_for_encoder_sentence, max_size_in_bucket_for_decoder_sentence)
        """
        # For the purpose of the experiment, there should not be more than 1 buckets
        assert len(buckets) == 1

        self.cfg = cfg
        self.max_gradient_norm = cfg.max_gradient_norm
        self.global_step = tf.Variable(0, trainable=False)
        self.is_training = tf.placeholder(tf.bool)

        # EWC coefficients
        self.ewc_loss_coef = tf.placeholder_with_default(0., [])
        self.beta = tf.placeholder_with_default(0.95, [])
        self.batch_size = cfg.batch_size

        self.buckets = buckets

        self.encoder_inputs = []
        self.decoder_inputs = []
        self.targets = []
        self.target_weights = []

        self.vocab_size_encoder = self.vocab_size_decoder = cfg.vocab_size

        for i in range(self.buckets[0][0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}".format(i)))
        for i in range(self.buckets[0][1]):
            self.targets.append(tf.placeholder(tf.int32, shape=[None],
                                               name="decoder{0}".format(i)))

        # decoder inputs : 'GO' + [ y_1, y_2, ... y_t-1 ]
        self.decoder_inputs = [tf.zeros_like(self.targets[0], dtype=tf.int64, name='GO')] + self.targets[:-1]

        # Binary mask useful for padded sequences.
        self.target_weights = [tf.ones_like(label, dtype=tf.float32) for label in self.targets]

        self.gradient_norms = []

        self.output_projection = None
        self.softmax_loss_function = None

    def build(self):
        """
        Build the model
        :return:
        """
        cprint("[*] Building model (G)", color="yellow")
        cell = tf.contrib.rnn.GRUCell(self.cfg.hidden_size)
        if self.cfg.num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell] * self.cfg.num_layers)

        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return seq2seq.embedding_rnn_seq2seq(
                encoder_inputs,
                decoder_inputs,
                cell,
                num_encoder_symbols=self.vocab_size_encoder,
                num_decoder_symbols=self.vocab_size_decoder,
                output_projection=self.output_projection,
                embedding_size=self.cfg.embedding_size,
                feed_previous=do_decode)

        with tf.variable_scope("seq2seq") as _:
            model_infos = seq2seq.model_with_buckets(
                self.encoder_inputs,
                self.decoder_inputs,
                self.targets,
                self.target_weights,
                self.buckets,
                lambda x, y: seq2seq_f(x, y, tf.logical_not(self.is_training)),
                softmax_loss_function=self.softmax_loss_function)

            self.outputs = model_infos[0][0]
            self.losses = model_infos[1][0]

        # Optimization :
        train_vars = tf.trainable_variables()

        # TODO: try Adam optimizer
        opt = tf.train.GradientDescentOptimizer(self.cfg.learning_rate)
        grads = tf.gradients(self.losses, train_vars)

        grad_variances, fisher, sticky_weights = [], [], []
        update_grad_variances, update_fisher, replace_fisher, update_sticky_weights, restore_sticky_weights = \
            [], [], [], [], []
        ewc_losses = []
        for i, (g, v) in enumerate(zip(grads, train_vars)):
            print(g, v)
            with tf.variable_scope("grad_variance"):
                grad_variances.append(
                    tf.get_variable(
                        "gv_{}".format(v.name.replace(":", "_")),
                        v.get_shape().as_list(),
                        dtype=tf.float32,
                        trainable=False,
                        initializer=tf.zeros_initializer()))
                fisher.append(
                    tf.get_variable(
                        "fisher_{}".format(v.name.replace(":", "_")),
                        v.get_shape().as_list(),
                        dtype=tf.float32,
                        trainable=False,
                        initializer=tf.zeros_initializer()))
            with tf.variable_scope("sticky_weights"):
                sticky_weights.append(
                    tf.get_variable(
                        "sticky_{}".format(v.name.replace(":", "_")),
                        v.get_shape().as_list(),
                        dtype=tf.float32,
                        trainable=False,
                        initializer=tf.zeros_initializer()))
            update_grad_variances.append(
                tf.assign(grad_variances[i], self.beta * grad_variances[i] + (
                    1 - self.beta) * g * g * self.batch_size))
            update_fisher.append(tf.assign(fisher[i], fisher[i] + grad_variances[i]))
            replace_fisher.append(tf.assign(fisher[i], grad_variances[i]))
            update_sticky_weights.append(tf.assign(sticky_weights[i], v))
            restore_sticky_weights.append(tf.assign(v, sticky_weights[i]))
            ewc_losses.append(
                tf.reduce_sum(tf.square(v - sticky_weights[i]) * fisher[i]))

        ewc_loss = self.losses + self.ewc_loss_coef * .5 * tf.add_n(ewc_losses)
        grads_ewc = tf.gradients(ewc_loss, train_vars)
        self.sticky_weights = sticky_weights
        self.grad_variances = grad_variances

        with tf.control_dependencies(update_grad_variances):
            self.update_grad_variances = tf.no_op('update_grad_variances')

        with tf.control_dependencies(update_grad_variances):
            self.updates = tf.cond(
                tf.equal(self.ewc_loss_coef, tf.constant(0.)),
                lambda: opt.apply_gradients(zip(grads, train_vars), global_step=self.global_step),
                lambda: opt.apply_gradients(zip(grads_ewc, train_vars), global_step=self.global_step))

        with tf.control_dependencies(update_fisher):
            self.update_fisher = tf.no_op('update_fisher')
        with tf.control_dependencies(replace_fisher):
            self.replace_fisher = tf.no_op('replace_fisher')
        with tf.control_dependencies(update_sticky_weights):
            self.update_sticky_weights = tf.no_op('update_sticky_weights')
        with tf.control_dependencies(restore_sticky_weights):
            self.restore_sticky_weights = tf.no_op('restore_sticky_weights')
        cprint("[!] Model built", color="green")

    def forward_with_feed_dict(self, session, questions, answers, is_training=False, ewc_loss_coeff=0):

        encoder_size, decoder_size = self.buckets[0]
        input_feed = {self.is_training: is_training}

        # Instead of an array of dim (batch_size, bucket_length),
        # the model is passed a list of sized batch_size, containing vector of size bucket_length
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = questions[:, l]

        # Same for decoder_input
        for l in range(decoder_size):
            input_feed[self.targets[l].name] = answers[:, l]
            input_feed[self.target_weights[l].name] = np.not_equal(answers[:, l], 0).astype(np.float32)

        if ewc_loss_coeff != 0:
            input_feed[self.ewc_loss_coef] = ewc_loss_coeff

        # Loss, a scalar
        output_feed = [self.losses]

        if is_training:
            output_feed += [
                self.updates  # Backward pass computation
            ]

        # If is not training, retrieve the outputs
        if not is_training:
            for l in range(decoder_size):
                output_feed.append(self.outputs[l])

        outputs = session.run(output_feed, input_feed)

        # Cleaner output dic
        if not is_training:
            outputs_dic = {
                "predictions": outputs[-decoder_size:]
            }
        else:
            outputs_dic = {}

        # If is_training:
        outputs_dic["losses"] = outputs[0]

        return outputs_dic
