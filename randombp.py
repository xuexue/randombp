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
    scope = 'bp'

    def __init__(self, num_hidden=100):
        self.define_placeholders()
        self.define_network(num_hidden)
        self.define_costs()
        self.define_train_step(num_hidden)

    def define_network(self, num_hidden):
        with tf.variable_scope(self.scope):
            self.w1 = wi("w1", [784, num_hidden]) # forward weights
            self.d1 = bi("d1", [num_hidden])
            self.z1 = tf.matmul(self.x, self.w1) + self.d1
            self.h = tf.nn.sigmoid(self.z1)

            self.w2 = wi("w2", [num_hidden, 10]) # forward weights
            self.d2 = bi("d2", [10])
            self.z2 = tf.matmul(self.h, self.w2) + self.d2
            self.ypred = tf.nn.softmax(self.z2)

    def define_train_step(self, num_hidden):
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

    def train(self, sess, mnist, lr=0.5, decay=0.0, epochs=50, verbose=False):
        testaccs = []
        trainaccs = []
        for e in range(epochs):
            for b in range(60):
                images, labels = mnist.train.next_batch(1000)
                sess.run(self.train_step,
                        feed_dict={self.x: images,
                                   self.y: labels,
                                   self.lr: lr,
                                   self.decay: decay
                                   })
            train_acc = sess.run(self.accuracy,
                    feed_dict={self.x: mnist.train.images,
                               self.y: mnist.train.labels })
            acc = sess.run(self.accuracy,
                    feed_dict={self.x: mnist.test.images,
                               self.y: mnist.test.labels })
            trainaccs.append(train_acc)
            testaccs.append(acc)
            if verbose:
                print 'Epoch %d, TrainAcc: %.1f %%, TestAcc: %.1f %%' %\
                        (e, train_acc*100, acc*100)
        return trainaccs, testaccs

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
    scope = 'rfn'

    def define_train_step(self, num_hidden):
        # define backward weights
        with tf.variable_scope(self.scope):
            b2 = wi("b2", [10, num_hidden]) # backwards weights
        # training: derivative w.r.t. activations
        ypred_grad = tf.gradients(self.cross_entropy, self.ypred)[0]
        z2_grad = tf.gradients(self.cross_entropy, self.z2)[0] # with softmax inclued
        h_grad = tf.matmul(z2_grad, b2)
        z1_grad = tf.multiply(tf.gradients(self.h, self.z1)[0], h_grad)
        # training: derivative w.r.t. weights
        self.w2_grad = tf.reduce_sum(
                tf.multiply(tf.expand_dims(self.h, 2),
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
            tf.assign(self.w2, self.w2 - self.alpha * self.w2_grad - self.decay),
            tf.assign(self.w1, self.w1 - self.alpha * self.w1_grad - self.decay),
            tf.assign(self.d2, self.d2 - self.alpha * self.d2_grad - self.decay),
            tf.assign(self.d1, self.d1 - self.alpha * self.d1_grad - self.decay),
        ]

class RandomFeedback4Layer(RandomFeedbackNet):
    """A 4-layer network, as above
    """
    scope = 'rf4'

    def layerwise_exp(self, sess, mnist, lr=0.5, decay=0.0, verbpose=False):
        test_acc = []
        stage = 1
        for step in range(4):
            n = 30
            if step == 0: # stage 1: train w1 + w2
                train_step = self.train_step[2:]
                n = 30 * 3
            elif step == 1: # stage 1: train only w3
                train_step = self.train_step[:2]
            elif step == 2: # stage 1: train w1 + w2
                train_step = self.train_step[2:]
            elif step == 3: # stage 1: train all
                train_step = self.train_step

            for iter in range(n):
                images, labels = mnist.train.next_batch(1000)
                sess.run(train_step,
                        feed_dict={self.x: images,
                                   self.y: labels,
                                   self.lr: lr,
                                   self.decay: decay
                                   })
                acc = sess.run(self.accuracy,
                        feed_dict={self.x: mnist.test.images,
                                   self.y: mnist.test.labels })
                test_acc.append(acc)
        return test_acc


    def define_network(self, num_hidden):
        with tf.variable_scope(self.scope):
            n1, n2 = num_hidden
            self.w1 = wi("w1", [784, n1]) # forward weights
            self.d1 = bi("d1", [n1])
            self.z1 = tf.matmul(self.x, self.w1) + self.d1
            self.h1 = tf.nn.tanh(self.z1)
            self.w2 = wi("w2", [n1, n2]) # forward weights
            self.d2 = bi("d2", [n2])
            self.z2 = tf.matmul(self.h1, self.w2) + self.d2
            self.h2 = tf.nn.tanh(self.z2)
            self.w3 = wi("w3", [n2, 10]) # forward weights
            self.d3 = bi("d3", [10])
            self.z3 = tf.matmul(self.h2, self.w3) + self.d3
            self.ypred = tf.nn.softmax(self.z3)

    def define_train_step(self, num_hidden):
        # define backward weights
        with tf.variable_scope(self.scope):
            n1, n2 = num_hidden
            b2 = wi("b2", [n2, n1]) # backwards weights
            b3 = wi("b3", [10, n2]) # backwards weights
        # training: derivative w.r.t. activations
        ypred_grad = tf.gradients(self.cross_entropy, self.ypred)[0]
        z3_grad = tf.gradients(self.cross_entropy, self.z3)[0] # softmax
        h2_grad = tf.matmul(z3_grad, b3)
        z2_grad = tf.multiply(tf.gradients(self.h2, self.z2)[0], h2_grad) #sigmoid
        h1_grad = tf.matmul(z2_grad, b2)
        z1_grad = tf.multiply(tf.gradients(self.h1, self.z1)[0], h1_grad) #sigmoid
        # training: derivative w.r.t. weights
        self.w3_grad = tf.reduce_sum(
                tf.multiply(tf.expand_dims(self.h2, 2),
                       tf.expand_dims(z3_grad, 1)),
                [0])
        self.d3_grad = tf.reduce_sum(z3_grad, [0])
        self.w2_grad = tf.reduce_sum(
                tf.multiply(tf.expand_dims(self.h1, 2),
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
            tf.assign(self.w3, self.w3 - self.alpha * self.w3_grad - self.decay),
            tf.assign(self.d3, self.d3 - self.alpha * self.d3_grad - self.decay),
            tf.assign(self.w2, self.w2 - self.alpha * self.w2_grad - self.decay),
            tf.assign(self.d2, self.d2 - self.alpha * self.d2_grad - self.decay),
            tf.assign(self.w1, self.w1 - self.alpha * self.w1_grad - self.decay),
            tf.assign(self.d1, self.d1 - self.alpha * self.d1_grad - self.decay),
        ]


class BackProp4Layer(RandomFeedback4Layer):
    """A 4-layer network, as above
    """
    scope = 'bp4'
    #def define_train_step(self, num_hidden):
    #    self.train_step = tf.train.GradientDescentOptimizer(self.lr) \
    #            .minimize(self.cross_entropy)
    def define_train_step(self, num_hidden):
        # training: assign weights
        self.w3_grad, self.w2_grad, self.w1_grad, \
                self.d3_grad, self.d2_grad, self.d1_grad = \
                    tf.gradients(self.cross_entropy,
                                 [self.w3, self.w2, self.w1,
                                  self.d3, self.d2, self.d1])
        self.train_step= [
            tf.assign(self.w3, self.w3 - self.alpha * self.w3_grad - self.decay),
            tf.assign(self.d3, self.d3 - self.alpha * self.d3_grad - self.decay),
            tf.assign(self.w2, self.w2 - self.alpha * self.w2_grad - self.decay),
            tf.assign(self.d2, self.d2 - self.alpha * self.d2_grad - self.decay),
            tf.assign(self.w1, self.w1 - self.alpha * self.w1_grad - self.decay),
            tf.assign(self.d1, self.d1 - self.alpha * self.d1_grad - self.decay),
        ]


class DirectFeedbackNet(RandomFeedback4Layer):
    """Based on
        'Direct Feedback Alignment Provides Learning in
        Deep Neural Networks'
    There are going to be some differences. I'll know later.
    """
    scope = 'dfn'
    #def layerwise_exp(self, sess, mnist, lr=0.5, decay=0.0, verbpose=False):
    #    super(DirectFeedbackNet, self).layerwise_exp(sess, mnist, lr=0.5, decay=0.0, verbpose=False):

    def define_train_step(self, num_hidden):
        with tf.variable_scope(self.scope):
            n1, n2 = num_hidden
            b2 = wi("b2", [10, n1]) # <--- SUBTLE DIFFERENCE
            b3 = wi("b3", [10, n2])
        # training: derivative w.r.t. activations
        ypred_grad = tf.gradients(self.cross_entropy, self.ypred)[0]
        z3_grad = tf.gradients(self.cross_entropy, self.z3)[0]
        h2_grad = tf.matmul(z3_grad, b3)
        z2_grad = tf.multiply(tf.gradients(self.h2, self.z2)[0], h2_grad)
        h1_grad = tf.matmul(z3_grad, b2) # <--- SUBTLE DIFFERENCE HERE
        z1_grad = tf.multiply(tf.gradients(self.h1, self.z1)[0], h1_grad)

        # training: derivative w.r.t. weights
        self.w3_grad = tf.reduce_sum(
                tf.multiply(tf.expand_dims(self.h2, 2),
                       tf.expand_dims(z3_grad, 1)),
                [0])
        self.d3_grad = tf.reduce_sum(z3_grad, [0])
        self.w2_grad = tf.reduce_sum(
                tf.multiply(tf.expand_dims(self.h1, 2),
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
            tf.assign(self.w3, self.w3 - self.alpha * self.w3_grad - self.decay),
            tf.assign(self.d3, self.d3 - self.alpha * self.d3_grad - self.decay),
            tf.assign(self.w2, self.w2 - self.alpha * self.w2_grad - self.decay),
            tf.assign(self.d2, self.d2 - self.alpha * self.d2_grad - self.decay),
            tf.assign(self.w1, self.w1 - self.alpha * self.w1_grad - self.decay),
            tf.assign(self.d1, self.d1 - self.alpha * self.d1_grad - self.decay),
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

        print 'Training normal neural net with backprop:'
        bpn.train(sess, mnist, epochs=100)

        print 'Training random feedback neural net:'
        rfn.train(sess, mnist, lr=0.5, decay=0.00001, epochs=100)

        print 'Training 4-layer random feedback net:'
        rfn4.train(sess, mnist, lr=0.5, decay=0.0001, epochs=100)

        print 'Training 4-layer direct feedback net:'
        dfb4.train(sess, mnist, lr=0.5, decay=0.0001, epochs=100)
