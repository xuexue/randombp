#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

def bi(name, shape, value=0.0, dtype=tf.float32):
    """Declares a bias variable with constant initialization."""
    return tf.get_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        initializer=tf.constant_initializer(
          value, dtype=dtype))


def wi(name, shape, dtype=tf.float32):
    """Declares a weight variable with random normal initialization."""
    return tf.get_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        initializer=tf.truncated_normal_initializer(
          mean=0.0, stddev=1 / np.sqrt(np.prod(shape[:-1])), dtype=dtype))


class BackPropNet(object):
    """Normal 3-layer network for comparison."""
    def __init__(self, num_hidden=100):
        self.define_placeholders()
        # define network
        w1 = wi("bp_w1", [784, num_hidden])
        d1 = bi("bp_d1", [num_hidden])
        h = tf.nn.sigmoid(tf.matmul(self.x, w1) + d1)
        w2 = wi("bp_w2", [num_hidden, 10])
        d2 = bi("bp_d2", [10])
        self.ypred = tf.nn.softmax(tf.matmul(h, w2) + d2)
        # costs
        self.define_costs()
        # training
        self.train_step = tf.train.GradientDescentOptimizer(self.lr) \
                .minimize(self.cross_entropy)

    def define_placeholders(self):
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.lr = tf.placeholder(tf.float32)

    def define_costs(self):
        # cross-entropy
        self.cross_entropy = tf.reduce_mean(
                -tf.reduce_sum(self.y * tf.log(self.ypred), reduction_indices=[1]))
        # acurracy
        correct_prediction = tf.equal(tf.argmax(self.ypred,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self, sess, data, lr=0.5, iter=100):
        for i in range(iter):
            images, labels = data.next_batch(1000)
            _, acc = sess.run([self.train_step, self.accuracy],
                    feed_dict={self.x: images, self.y: labels, self.lr: lr})
            print 'Iter %d, Acc: %.1f %%' % (i, acc*100)

class RandomFeedbackNet(BackPropNet):
    """Three-layer netowrk based on
        'Random feedback weights support learning in deep neural networks'
    There are a few differences:
        * number of hidden units is adjustable
        * cross-entropy is used for training
        * weights are initialized using normal random (instead of uniform)
        * biases are initialized to zero (instead of uniform)
        * did not perform any hyperparamter tuning
    """
    def __init__(self, num_hidden=100):
        self.define_placeholders()
        # define network
        w1 = wi("rf_w1", [784, num_hidden]) # forward weights
        #b1 = wi("rf_b1", [num_hidden, 784]) # backwards weights (unused)
        d1 = bi("rf_d1", [num_hidden])
        z1 = tf.matmul(self.x, w1) + d1
        h = tf.nn.sigmoid(z1)

        w2 = wi("rf_w2", [num_hidden, 10]) # forward weights
        b2 = wi("rf_b2", [10, num_hidden]) # backwards weights
        d2 = bi("rf_d2", [10])
        z2 = tf.matmul(h, w2) + d2
        self.ypred = tf.nn.softmax(z2)
        # costs
        self.define_costs()
        # training: derivative w.r.t. activations
        ypred_grad = tf.gradients(self.cross_entropy, self.ypred)[0]
        z2_grad = tf.gradients(self.cross_entropy, z2)[0] # with softmax inclued
        h_grad = tf.matmul(z2_grad, b2)
        z1_grad = tf.mul(tf.gradients(h, z1)[0], h_grad)
        # training: derivative w.r.t. weights
        self.w2_grad = tf.reduce_sum(
                tf.mul(tf.expand_dims(h, 2),
                       tf.expand_dims(z2_grad, 1)), # order?
                [0])
        self.d2_grad = tf.reduce_sum(z2_grad, [0])
        self.w1_grad = tf.reduce_sum(
                tf.mul(tf.expand_dims(self.x, 2),
                       tf.expand_dims(z1_grad, 1)), # order?
                [0])
        self.d1_grad = tf.reduce_sum(z1_grad, [0])
        # training: assign weights
        self.train_step= [
            tf.assign(w2, w2 - self.lr * self.w2_grad),
            tf.assign(w1, w1 - self.lr * self.w1_grad),
            tf.assign(d2, d2 - self.lr * self.d2_grad),
            tf.assign(d1, d1 - self.lr * self.d1_grad),
        ]


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("mnist/", one_hot=True)

    bpn = BackPropNet()
    rfn = RandomFeedbackNet()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print 'Training normal neural net with backprop:'
        bpn.train(sess, mnist.train)
        print 'Training random feedback neural net:'
        rfn.train(sess, mnist.train, 0.5)

