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
            mean=0.0, stddev=0.2, dtype=dtype))


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
        self.lr = tf.placeholder(tf.float32) # learning rate
        self.decay = tf.placeholder(tf.float32) # weight decay
        self.N = tf.cast(tf.shape(self.x)[0], tf.float32)
        self.alpha = self.lr # / self.N

    def define_costs(self):
        # cross-entropy
        self.cross_entropy = tf.reduce_mean(
                -tf.reduce_sum(self.y * tf.log(self.ypred), reduction_indices=[1]))
        # acurracy
        correct_prediction = tf.equal(tf.argmax(self.ypred,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self, sess, data, lr=0.5, decay=0.0, iter=50):
        for i in range(iter):
            images, labels = data.next_batch(1000)
            _, acc = sess.run([self.train_step, self.accuracy],
                    feed_dict={self.x: images,
                               self.y: labels,
                               self.lr: lr,
                               self.decay: decay
                               })
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
        z1_grad = tf.multiply(tf.gradients(h, z1)[0], h_grad)
        # training: derivative w.r.t. weights
        self.w2_grad = tf.reduce_sum(
                tf.multiply(tf.expand_dims(h, 2),
                       tf.expand_dims(z2_grad, 1)), # order?
                [0])
        self.d2_grad = tf.reduce_sum(z2_grad, [0])
        self.w1_grad = tf.reduce_sum(
                tf.multiply(tf.expand_dims(self.x, 2),
                       tf.expand_dims(z1_grad, 1)), # order?
                [0])
        self.d1_grad = tf.reduce_sum(z1_grad, [0])
        # training: assign weights
        self.train_step= [
            tf.assign(w2, w2 - self.alpha * self.w2_grad - self.decay),
            tf.assign(w1, w1 - self.alpha * self.w1_grad - self.decay),
            tf.assign(d2, d2 - self.alpha * self.d2_grad - self.decay),
            tf.assign(d1, d1 - self.alpha * self.d1_grad - self.decay),
        ]

class RandomFeedback4Layer(RandomFeedbackNet):
    """A 4-layer network, as above
    """
    def __init__(self, num_hidden=[300, 100]):
        self.define_placeholders()
        # define network:
        n1, n2 = num_hidden
        w1 = wi("fa_w1", [784, n1]) # forward weights
        d1 = bi("fa_d1", [n1])
        z1 = tf.matmul(self.x, w1) + d1
        h1 = tf.nn.sigmoid(z1)
        w2 = wi("fa_w2", [n1, n2]) # forward weights
        b2 = wi("fa_b2", [n2, n1]) # backwards weights
        d2 = bi("fa_d2", [n2])
        z2 = tf.matmul(h1, w2) + d2
        h2 = tf.nn.sigmoid(z2)
        w3 = wi("fa_w3", [n2, 10]) # forward weights
        b3 = wi("fa_b3", [10, n2]) # backwards weights
        d3 = bi("fa_d3", [10])
        z3 = tf.matmul(h2, w3) + d3
        self.ypred = tf.nn.softmax(z3)
        # costs
        self.define_costs()

        # training: derivative w.r.t. activations
        ypred_grad = tf.gradients(self.cross_entropy, self.ypred)[0]
        z3_grad = tf.gradients(self.cross_entropy, z3)[0] # softmax
        h2_grad = tf.matmul(z3_grad, b3)
        z2_grad = tf.multiply(tf.gradients(h2, z2)[0], h2_grad) #sigmoid
        h1_grad = tf.matmul(z2_grad, b2)
        z1_grad = tf.multiply(tf.gradients(h1, z1)[0], h1_grad) #sigmoid

        # training: derivative w.r.t. weights
        self.w3_grad = tf.reduce_sum(
                tf.multiply(tf.expand_dims(h2, 2),
                       tf.expand_dims(z3_grad, 1)),
                [0])
        self.d3_grad = tf.reduce_sum(z3_grad, [0])
        self.w2_grad = tf.reduce_sum(
                tf.multiply(tf.expand_dims(h1, 2),
                       tf.expand_dims(z2_grad, 1)),
                [0])
        self.d2_grad = tf.reduce_sum(z2_grad, [0])
        self.w1_grad = tf.reduce_sum(
                tf.multiply(tf.expand_dims(self.x, 2),
                       tf.expand_dims(z1_grad, 1)),
                [0])
        self.d1_grad = tf.reduce_sum(z1_grad, [0])
        # training: assign weights
        self.train_step= [
            tf.assign(w3, w3 - self.alpha * self.w3_grad - self.decay),
            tf.assign(w2, w2 - self.alpha * self.w2_grad - self.decay),
            tf.assign(w1, w1 - self.alpha * self.w1_grad - self.decay),
            tf.assign(d3, d3 - self.alpha * self.d3_grad - self.decay),
            tf.assign(d2, d2 - self.alpha * self.d2_grad - self.decay),
            tf.assign(d1, d1 - self.alpha * self.d1_grad - self.decay),
        ]

class DirectFeedbackNet(BackPropNet):
    """Based on
        'Direct Feedback Alignment Provides Learning in
        Deep Neural Networks'
    There are going to be some differences. I'll know later.
    """
    def __init__(self, num_hidden=[300, 100]):
        self.define_placeholders()
        # define network:
        n1, n2 = num_hidden
        w1 = wi("df_w1", [784, n1]) # forward weights
        d1 = bi("df_d1", [n1])
        z1 = tf.matmul(self.x, w1) + d1
        h1 = tf.nn.sigmoid(z1)
        w2 = wi("df_w2", [n1, n2]) # forward weights
        b2 = wi("df_b2", [10, n1]) # backwards weights <--- SUBTLE DIFFERENCE
        d2 = bi("df_d2", [n2])
        z2 = tf.matmul(h1, w2) + d2
        h2 = tf.nn.sigmoid(z2)
        w3 = wi("df_w3", [n2, 10]) # forward weights
        b3 = wi("df_b3", [10, n2]) # backwards weights
        d3 = bi("df_d3", [10])
        z3 = tf.matmul(h2, w3) + d3
        self.ypred = tf.nn.softmax(z3)
        # costs
        self.define_costs()

        # training: derivative w.r.t. activations
        ypred_grad = tf.gradients(self.cross_entropy, self.ypred)[0]
        z3_grad = tf.gradients(self.cross_entropy, z3)[0] # softmax
        h2_grad = tf.matmul(z3_grad, b3)
        z2_grad = tf.multiply(tf.gradients(h2, z2)[0], h2_grad) #sigmoid
        h1_grad = tf.matmul(z3_grad, b2) # <--- SUBTLE DIFFERENCE HERE
        z1_grad = tf.multiply(tf.gradients(h1, z1)[0], h1_grad) #sigmoid

        # training: derivative w.r.t. weights
        self.w3_grad = tf.reduce_sum(
                tf.multiply(tf.expand_dims(h2, 2),
                       tf.expand_dims(z3_grad, 1)),
                [0])
        self.d3_grad = tf.reduce_sum(z3_grad, [0])
        self.w2_grad = tf.reduce_sum(
                tf.multiply(tf.expand_dims(h1, 2),
                       tf.expand_dims(z2_grad, 1)),
                [0])
        self.d2_grad = tf.reduce_sum(z2_grad, [0])
        self.w1_grad = tf.reduce_sum(
                tf.multiply(tf.expand_dims(self.x, 2),
                       tf.expand_dims(z1_grad, 1)),
                [0])
        self.d1_grad = tf.reduce_sum(z1_grad, [0])
        # training: assign weights
        self.train_step= [
            tf.assign(w3, w3 - self.alpha * self.w3_grad - self.decay),
            tf.assign(w2, w2 - self.alpha * self.w2_grad - self.decay),
            tf.assign(w1, w1 - self.alpha * self.w1_grad - self.decay),
            tf.assign(d3, d3 - self.alpha * self.d3_grad - self.decay),
            tf.assign(d2, d2 - self.alpha * self.d2_grad - self.decay),
            tf.assign(d1, d1 - self.alpha * self.d1_grad - self.decay),
        ]

if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("mnist/", one_hot=True)

    bpn = BackPropNet()
    rfn = RandomFeedbackNet()
    rfn4 = RandomFeedback4Layer([200,100])
    dfb4 = DirectFeedbackNet([200,100])
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        '''
        print 'Training normal neural net with backprop:'
        bpn.train(sess, mnist.train, iter=1000)

        print 'Training random feedback neural net:'
        rfn.train(sess, mnist.train, lr=0.5, decay=0.00001, iter=1000)

        print 'Training 4-layer random feedback net:'
        rfn4.train(sess, mnist.train, lr=0.5, decay=0.0001, iter=1000)
        '''

        print 'Training 4-layer direct feedback net:'
        dfb4.train(sess, mnist.train, lr=0.5, decay=0.0001, iter=1000)
