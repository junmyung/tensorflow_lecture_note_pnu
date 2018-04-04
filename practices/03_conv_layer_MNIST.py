"""
Tensorflow Lecture Note 03
PNU VISLAB
modified by Junmyung Jimmy Choi

Using convolutional net on MNIST dataset of handwritten digits
MNIST dataset: http://yann.lecun.com/exdb/mnist/
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 07
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops.init_ops import *
import utils
import pdb


def conv2d(inputs, nb_filter, filter_size, strides=1, padding='same',
           activation=tf.nn.relu, bias=True, weights_init=variance_scaling_initializer,
           bias_init=zeros_initializer, trainable=True, scope='Conv2d'):
  with tf.variable_scope(scope,  values=[inputs], reuse=tf.AUTO_REUSE) as scope:
    W_init = weights_init
    bias_init = bias_init

    ################
    # your code here#
    ################

    x = None
  return x

def maxpool(inputs, filter_size=2, strides=1, padding='same', scope='pool'):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
    ################
    #your code here#
    ################
    x = None
  return x

def fully_connected(inputs, nb_filter, activation=tf.nn.relu, bias=True, weights_init=variance_scaling_initializer,
           bias_init=zeros_initializer, trainable=True,  scope='FC'):

  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
    ################
    #your code here#
    ################

    x = None
  return x

class ConvNet(object):
  def __init__(self):
    self.lr = 0.0001
    self.batch_size = 128
    self.keep_prob = tf.constant(0.75)
    self.gstep = tf.Variable(0, dtype=tf.int32,
                             trainable=False, name='global_step')
    self.n_classes = 10
    self.skip_step = 200
    self.n_test = 10000
    self.training = True

  def get_data(self):
    with tf.name_scope('data'):
      # Step 1: Read in data
      self.mnist = input_data.read_data_sets('data/mnist', one_hot=True)
      # Step 2: create placeholders for features and labels
      self.X = tf.placeholder(tf.float32, [None, 784], name='image')
      self.img = tf.reshape(self.X, shape=[-1, 28, 28, 1])
      self.label = tf.placeholder(tf.int32, [None, 10], name='label')

  def inference(self):
    with tf.variable_scope('model') as scope:

      ################
      #your code here#
      ################
      self.logits = None
      pass

  def loss(self):
    '''
    define loss function
    use softmax cross entropy with logits as the loss function
    compute mean cross entropy, softmax is applied internally
    '''
    #
    with tf.name_scope('loss'):
      ################
      # your code here#
      ################
      entropy = None
      self.loss = tf.reduce_mean(entropy, name='loss')

  def optimize(self):
    '''
    Define training op
    using Adam Gradient Descent to minimize cost
    '''
    ################
    # your code here#
    ################
    self.opt = None
  def summary(self):
    '''
    Create summaries to write on TensorBoard
    '''
    with tf.name_scope('summaries'):
      tf.summary.scalar('loss', self.loss)
      tf.summary.scalar('accuracy', self.accuracy_holder)
      tf.summary.histogram('histogram loss', self.loss)
      self.summary_op = tf.summary.merge_all()

  def eval(self):
    '''
    Count the number of right predictions in a batch
    '''
    with tf.name_scope('predict'):
      preds = tf.nn.softmax(self.logits)
      correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
      self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
      self.accuracy_holder = tf.placeholder(tf.float32,shape=None, name='Accuracy')

  def build(self):
    '''
    Build the computation graph
    '''
    self.get_data()
    self.inference()
    self.loss()
    self.optimize()
    self.eval()
    self.summary()

  def train_one_epoch(self, sess, saver, writer, epoch, step):
    start_time = time.time()
    self.training = True
    n_batches = int(self.mnist.train.num_examples/self.batch_size)
    total_loss = 0
    for j in range(n_batches):
      self.X_batch,self.Y_batch = self.mnist.train.next_batch(self.batch_size)
      _, l = sess.run([self.opt, self.loss],feed_dict={self.X:self.X_batch,self.label:self.Y_batch})
      step += 1
      total_loss += l
    saver.save(sess, 'checkpoints/convnet_mnist/mnist-convnet', step)
    print('Average loss at epoch {0}: {1} Took: {2} seconds'.format(epoch, total_loss/n_batches,time.time()-start_time))
    return step

  def eval_once(self, sess, writer, epoch, step):
    start_time = time.time()
    self.training = False
    total_correct_preds = 0
    n_batches = int(self.mnist.test.num_examples/100)
    for j in range(n_batches):
      self.X_batch, self.Y_batch = self.mnist.test.next_batch(100)
      accuracy_batch = sess.run(self.accuracy, feed_dict={self.X: self.X_batch, self.label: self.Y_batch})
      total_correct_preds += accuracy_batch
    summaries = sess.run(self.summary_op, feed_dict={self.X:self.X_batch,self.label:self.Y_batch,self.accuracy_holder:total_correct_preds/n_batches})
    writer.add_summary(summaries, global_step=step)
    print('Accuracy at epoch {0}: {1} Took: {2} seconds'.format(epoch, total_correct_preds/n_batches, time.time()-start_time))

  def train(self, n_epochs):
    '''
    The train function alternates between training one epoch and evaluating
    '''
    utils.safe_mkdir('checkpoints')
    utils.safe_mkdir('checkpoints/convnet_mnist')
    writer = tf.summary.FileWriter('./graphs/convnet', tf.get_default_graph())

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      saver = tf.train.Saver()
      ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_mnist/checkpoint'))
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

      step = self.gstep.eval()

      for epoch in range(n_epochs):
        step = self.train_one_epoch(sess, saver, writer, epoch, step)
        self.eval_once(sess, writer, epoch, step)
    writer.close()


if __name__=='__main__':
  model = ConvNet()
  model.build()
  model.train(n_epochs=30)
