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



def get_module(keys):
  module_arg = {
                "tanh"            : tf.nn.tanh,
                "sigmoid"         : tf.nn.sigmoid,
                "softmax"         : tf.nn.softmax,
                "relu"            : tf.nn.relu,
                "leaky_relu"      : tf.nn.leaky_relu,
                "elu"             : tf.nn.elu,
                "zeros"           : Zeros,
                "uniform"         : RandomUniform,
                "uniform_scaling" : UniformUnitScaling,
                "normal"          : RandomNormal,
                "trucated_normal" : TruncatedNormal,
                "variance_scaling": VarianceScaling(scale=2.0, mode='fan_in',distribution="normal"),
                "xavier"          : VarianceScaling(scale=1.0, mode='fan_avg', distribution="uniform")
               }
  module = module_arg.get(keys)
  return module

def get_shape(inputs):
  if isinstance(inputs, tf.Tensor):
    return inputs.get_shape().as_list()
  elif type(inputs) in [np.array, list, tuple]:
    return np.shape(inputs)
  else:
    raise Exception("Invalid inputs layer.")

def filter_format(fsize, in_depth, out_depth):
  if isinstance(fsize, int):
    return [fsize, fsize, in_depth, out_depth]
  elif isinstance(fsize, (tuple, list)):
    if len(fsize)==2:
      return [fsize[0], fsize[1], in_depth, out_depth]
    else:
      raise Exception("filter length error: "+str(len(fsize))
                      +", only a length of 2 is supported.")
  else:
    raise Exception("filter format error: "+str(type(fsize)))

def stride_format(strides):
  if isinstance(strides, int):
    return [1, strides, strides, 1]
  elif isinstance(strides, (tuple, list)):
    if len(strides)==2:
      return [1, strides[0], strides[1], 1]
    elif len(strides)==4:
      return [strides[0], strides[1], strides[2], strides[3]]
    else:
      raise Exception("strides length error: "+str(len(strides))
                      +", only a length of 2 or 4 is supported.")
  else:
    raise Exception("strides format error: "+str(type(strides)))

def padding_format(padding):
  if padding in ['same', 'SAME', 'valid', 'VALID']:
    return str.upper(padding)
  else:
    raise Exception("Unknown padding! Accepted values: 'same', 'valid'.")

def conv2d(inputs, nb_filter, filter_size, strides=1, padding='same',
           activation='relu', bias=True, weights_init='variance_scaling',
           bias_init='zeros',regularizer=None, weight_decay=0.001,
           trainable=True, reuse=None, scope='Conv2D'):
  # formatting
  input_shape = get_shape(inputs)
  assert len(input_shape)==4, "Incoming Tensor shape must be 4-D"
  if not nb_filter:
    nb_filter = input_shape[-1]
  filter_size = filter_format(filter_size,
                              input_shape[-1],
                              nb_filter)
  strides = stride_format(strides)
  padding = padding_format(padding)

  with tf.variable_scope(scope, values=[inputs], reuse=reuse) as scope:
    if isinstance(weights_init, str):
      W_init = get_module(weights_init)()
    elif isinstance(weights_init, object):
      W_init = weights_init

    W_regul = None
    if regularizer:
      W_regul = lambda x: tf.multiply(tf.nn.l2_loss(x), weight_decay, name='L2-Loss')
    W = tf.get_variable("weight", shape=filter_size, regularizer=W_regul,
                        initializer=W_init, trainable=trainable)

    b = None
    if bias:
      if isinstance(bias_init, str):
        bias_init = get_module(bias_init)()
      b = tf.get_variable('bias', shape=nb_filter, initializer=bias_init,
                          trainable=trainable)

    x = tf.nn.conv2d(inputs, W, strides, padding)
    if b: x = tf.nn.bias_add(x, b)
    if activation:
      if isinstance(activation, str):
        x = get_module(activation)(x)
      elif isinstance(activation,object):
        x = activation(x)
  return x

def maxpool2d(inputs, filter_size, strides=None, padding='same',
              scope="MaxPool2D"):

  # formatting
  input_shape = get_shape(inputs)

  assert len(input_shape)==4, "Incoming Tensor shape must be 4-D"
  filter_size = stride_format(filter_size)
  strides = stride_format(strides)
  padding = padding_format(padding)
  with tf.name_scope(scope):
    x = tf.nn.max_pool(inputs, filter_size, strides, padding)
  return x

def fully_connected(inputs, nb_filter, activation='relu', bias=True, weights_init='variance_scaling',
                    bias_init='zeros', regularizer=None, weight_decay=0.001,
                    trainable=True, reuse=None, scope="fully_connected"):
  # formatting
  input_shape = get_shape(inputs)
  assert len(input_shape)>1, "Incoming Tensor shape must be at least 2-D"
  n_inputs = int(np.prod(input_shape[1:]))

  with tf.variable_scope(scope, values=[inputs],reuse=reuse) as scope:
    W_init = weights_init
    if isinstance(weights_init, str):
      W_init = get_module(weights_init)()
    W_regul = None
    if regularizer:
      W_regul = lambda x: tf.multiply(tf.nn.l2_loss(x), weight_decay, name='L2-Loss')
    W = tf.get_variable("weight", shape=[n_inputs, nb_filter], regularizer=W_regul,
                        initializer=W_init, trainable=trainable)
    b = None
    if bias:
      if isinstance(bias_init, str):
        bias_init = get_module(bias_init)()
      b = tf.get_variable('bias', shape=nb_filter, initializer=bias_init,
                          trainable=trainable)
    # If input is not 2d, flatten it.
    if len(input_shape)>2:
      inputs = tf.reshape(inputs, [-1, n_inputs])
    x = tf.matmul(inputs, W)
    if b: x = tf.nn.bias_add(x, b)
    if activation:
      if isinstance(activation, str):
        x = get_module(activation)(x)

  return x

def conv2d(inputs, nb_filter, filter_size, strides=1, padding='same',
           activation=tf.nn.relu, bias=True, weights_init=variance_scaling_initializer,
           bias_init=zeros_initializer, trainable=True, scope='Conv2d'):
  with tf.variable_scope(scope,  values=[inputs], reuse=tf.AUTO_REUSE) as scope:
    W_init = weights_init
    bias_init = bias_init
    x = tf.layers.conv2d(inputs,filters=nb_filter,kernel_size=filter_size,strides=[strides,strides],padding=padding,
                         kernel_initializer=W_init,bias_initializer=bias_init,activation=activation,trainable=trainable)
  return x

def maxpool(inputs, filter_size=2, strides=1, padding='same', scope='pool'):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
    x = tf.layers.max_pooling2d(inputs,pool_size=[filter_size,filter_size],strides=[strides,strides],padding=padding)
  return x

def fully_connected(inputs, nb_filter, activation=tf.nn.relu, bias=True, weights_init=variance_scaling_initializer,
           bias_init=zeros_initializer, trainable=True,  scope='FC'):

  if len(inputs.get_shape().as_list())>2:
    flatten = int(np.prod(inputs.get_shape().as_list()[1:]))
    inputs = tf.reshape(inputs, [-1, flatten])
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
    x = tf.layers.dense(inputs,units=nb_filter,activation=activation,use_bias=bias,kernel_initializer=weights_init,
                        bias_initializer=bias_init,trainable=trainable)
  return x

class ConvNet(object):
  def __init__(self):
    self.lr = 0.0001
    self.batch_size = 128
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
      conv1 = conv2d(inputs=self.img,nb_filter=32,filter_size=5,trainable=self.training,scope='layer1')
      pool1 = maxpool(conv1, 2, 2, 'same', scope ='layer2')
      conv2 = conv2d(inputs=pool1,nb_filter=64,filter_size=5,trainable=self.training,scope='layer3')
      pool2 = maxpool(conv2, 2, 2, 'same', scope ='layer4')
      fc = fully_connected(pool2, 1024,trainable=self.training, scope='layer5')
      self.logits = fully_connected(fc, self.n_classes,activation=None,trainable=self.training, scope='logits')

  def loss(self):
    '''
    define loss function
    use softmax cross entropy with logits as the loss function
    compute mean cross entropy, softmax is applied internally
    '''
    #
    with tf.name_scope('loss'):
      entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
      self.loss = tf.reduce_mean(entropy, name='loss')

  def optimize(self):
    '''
    Define training op
    using Adam Gradient Descent to minimize cost
    '''
    self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss,
                                                        global_step=self.gstep)

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
