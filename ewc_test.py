import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data


class Model:
    def __init__(self):
        self.x = x = tf.placeholder(tf.float32, [None, 784])
        self.y = y = tf.placeholder(tf.int64, [None])
        self.ewc_loss_coef = tf.placeholder_with_default(0., [])
        self.beta = tf.placeholder_with_default(0.95, [])
        self.dropout = tf.placeholder_with_default(1., [])

        x = tf.cond(tf.less(self.dropout, 1.), lambda: tf.nn.dropout(x, 0.8), lambda: x)

        hx = tf.contrib.layers.fully_connected(
            inputs=x, num_outputs=1500, activation_fn=tf.nn.relu)
        hx = tf.nn.dropout(hx, self.dropout)
        hx = tf.contrib.layers.fully_connected(
            inputs=hx, num_outputs=1500, activation_fn=tf.nn.relu)
        hx = tf.nn.dropout(hx, self.dropout)
        hx = tf.contrib.layers.fully_connected(
            inputs=hx, num_outputs=1500, activation_fn=tf.nn.relu)
        hx = tf.nn.dropout(hx, self.dropout)
        self.logits = logits = tf.contrib.layers.fully_connected(
            inputs=hx, num_outputs=10)
        self.softmax = tf.nn.softmax(logits)

        self.var_list = tvs = tf.trainable_variables()
        self.cross_entropy = cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y, logits=self.logits))

        opt = tf.train.GradientDescentOptimizer(0.1)
        self.grads = grads = tf.gradients(cross_entropy, tvs)

        # create gradient variance accumulators and update ops
        grad_variances, fisher, sticky_weights = [], [], []
        update_grad_variances, update_fisher, replace_fisher, update_sticky_weights, restore_sticky_weights = \
            [], [], [], [], []
        ewc_losses = []
        for i, (g, v) in enumerate(zip(grads, tvs)):
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
                    1 - self.beta) * g * g * tf.to_float(tf.shape(x)[0])))
            update_fisher.append(tf.assign(fisher[i], fisher[i] + grad_variances[i]))
            replace_fisher.append(tf.assign(fisher[i], grad_variances[i]))
            update_sticky_weights.append(tf.assign(sticky_weights[i], v))
            restore_sticky_weights.append(tf.assign(v, sticky_weights[i]))
            ewc_losses.append(
                tf.reduce_sum(tf.square(v - sticky_weights[i]) * fisher[i]))

        ewc_loss = cross_entropy + self.ewc_loss_coef * .5 * tf.add_n(ewc_losses)
        grads_ewc = tf.gradients(ewc_loss, tvs)
        self.sticky_weights = sticky_weights
        self.grad_variances = grad_variances

        with tf.control_dependencies(update_grad_variances):
            self.update_grad_variances = tf.no_op('update_grad_variances')

        with tf.control_dependencies(update_grad_variances):
            self.ts = tf.cond(
                tf.equal(self.ewc_loss_coef, tf.constant(0.)),
                lambda: opt.apply_gradients(zip(grads, tvs)),
                lambda: opt.apply_gradients(zip(grads_ewc, tvs)))

        with tf.control_dependencies(update_fisher):
            self.update_fisher = tf.no_op('update_fisher')
        with tf.control_dependencies(replace_fisher):
            self.replace_fisher = tf.no_op('replace_fisher')
        with tf.control_dependencies(update_sticky_weights):
            self.update_sticky_weights = tf.no_op('update_sticky_weights')
        with tf.control_dependencies(restore_sticky_weights):
            self.restore_sticky_weights = tf.no_op('restore_sticky_weights')

        self.acc = tf.reduce_mean(
            tf.cast(tf.equal(y, tf.argmax(logits, 1)), tf.float32))


def mnist_imshow(img):
    plt.imshow(img.reshape([28, 28]), cmap="gray")
    plt.axis('off')


# return a new mnist dataset w/ pixels randomly permuted
def permute_mnist(mnist):
    perm_inds = list(range(mnist.train.images.shape[1]))
    np.random.shuffle(perm_inds)
    mnist2 = deepcopy(mnist)
    sets = ["train", "validation", "test"]
    for set_name in sets:
        this_set = getattr(mnist2, set_name)  # shallow copy
        this_set._images = np.transpose(np.array([this_set.images[:, c] for c in perm_inds]))
    return mnist2


def plot_test_acc(plot_handles):
    plt.legend(handles=plot_handles, loc="center right")
    plt.xlabel("Iterations")
    plt.ylabel("Test Accuracy")
    plt.ylim(0, 1)
    plt.savefig("plot" + str(plot_0))


# train/compare vanilla sgd and ewc
def train_task(sess, m, num_iter, disp_freq, trainset, testsets, lams=[0], restore_weights=False):
    for l in range(len(lams)):
        # lams[l] sets weight on old task(s)
        if restore_weights:
            sess.run(m.restore_sticky_weights)

        # initialize test accuracy array for each task
        test_accs = []
        for task in range(len(testsets)):
            test_accs.append(np.zeros(num_iter // disp_freq))

        # train on current task
        for iter in range(num_iter):
            X, Y = trainset.train.next_batch(50)
            feed_dict = {m.x: X, m.y: Y}
            if lams[l] != 0:
                feed_dict[m.ewc_loss_coef] = lams[l]
            sess.run(m.ts, feed_dict=feed_dict)
            if iter % disp_freq == 0:
                plt.subplot(1, len(lams), l + 1)
                plots = []
                colors = ['r', 'b', 'g']
                for task in range(len(testsets)):
                    feed_dict = {m.x: testsets[task].test.images, m.y: testsets[task].test.labels}
                    test_accs[task][iter // disp_freq] = m.acc.eval(feed_dict=feed_dict)
                    c = chr(ord('A') + task)
                    plot_h, = plt.plot(range(1, iter + 2, disp_freq), test_accs[task][:iter // disp_freq + 1],
                                       colors[task], label="task " + c)
                    plots.append(plot_h)
                plot_test_acc(plots)
                if l == 0:
                    plt.title("vanilla sgd")
                else:
                    plt.title("ewc")
                plt.gcf().set_size_inches(len(lams) * 5, 3.5)

"""
1. First training, no restore weight, simple SGD training.
Train the model, and add a validation training set.

1 bis: Run update Fisher matrix + save the last weights in sticky weights

2 Train the model on a new model, with two validation set (the previous model and the last model)



"""

if __name__ == '__main__':
    plot_0 = 0
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    sess = tf.InteractiveSession()

    m = Model()
    x, y = m.x, m.y

    sess.run(tf.global_variables_initializer())

    train_task(sess, m, 10000, 250, mnist, [mnist], lams=[0])
    plot_0 += 1
    sess.run([m.update_fisher, m.update_sticky_weights])

    print([v.name for v in tf.global_variables()])

    g = tf.get_default_graph()
    v = g.get_tensor_by_name('grad_variance/gv_fully_connected/weights_0:0')

    F_row_mean = np.mean(sess.run(v), 1)
    mnist_imshow(F_row_mean)
    plt.title("W1 row-wise mean Fisher")

    # permuting mnist for 2nd task
    mnist2 = permute_mnist(mnist)

    plt.subplot(1, 2, 1)
    mnist_imshow(mnist.train.images[5])
    plt.title("original task image")
    plt.subplot(1, 2, 2)
    mnist_imshow(mnist2.train.images[5])
    plt.title("new task image")

    train_task(sess, m, 10000, 250, mnist2, [mnist, mnist2], lams=[0, 15], restore_weights=True)
    plot_0 += 1
    sess.run(m.acc, {m.x: mnist.test.images, m.y: mnist.test.labels})

    sess.run([m.update_fisher, m.update_sticky_weights])

    mnist3 = permute_mnist(mnist)

    train_task(sess, m, 10000, 250, mnist3, [mnist, mnist2, mnist3], lams=[0, 15], restore_weights=True)
    plot_0 += 1
