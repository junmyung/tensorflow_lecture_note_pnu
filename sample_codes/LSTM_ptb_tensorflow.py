# -*- coding:utf-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
import pickle
from tensorflow.python.client import device_lib
import pdb
flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", "./data/ptb.pkl",
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", "./checkpoints/tutorials",
                    "Model output directory.")
FLAGS = flags.FLAGS

class SmallConfig(object):
  """Small config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000

def ptb_input_producer(raw_data, batch_size, num_steps, name=None):

  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len]) #[20, ~]

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y

class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size               # 배치사이즈
    self.num_steps = num_steps = config.num_steps                  # RNN 길이
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps # Epoch당 반복 횟수
    self.input_data, self.targets = ptb_input_producer(
        data, batch_size, num_steps, name=name)                    # input data, target data


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self._is_training = is_training
    self._input = input_
    self._rnn_params = None
    self._cell = None
    self.batch_size = input_.batch_size
    self.num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=tf.float32) #[10000, size]
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data) #input_data값(0~9999,index)을 받아서 embedding값으로 줌

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    #### sequential conv1d filtering ####
    # inputs = tf.layers.conv1d(inputs,200,3,padding='same')

    #### depthwise conv1d filtering ####
    # inputs = tf.transpose(inputs, [1, 0, 2]) #[20,35,200] ->[35,20,200]
    # proj = lambda x: tf.layers.conv1d(x, filters=1, kernel_size=3, padding='same')
    # inputs = tf.squeeze(tf.map_fn(proj, tf.expand_dims(inputs,axis=-1)),axis=-1)
    # inputs = tf.transpose(inputs, [1, 0, 2])

    inputs = tf.reshape(inputs, shape=[self.batch_size*self.num_steps, -1])
    inputs = tf.layers.dense(inputs, size)
    inputs = tf.reshape(inputs, shape=[self.batch_size,self.num_steps, -1])
    output, state = self._build_rnn_graph_lstm(inputs, config, is_training)


    ###

    #### depthwise outputs filtering ####
    # output = tf.reshape(output, shape=[self.num_steps, self.batch_size, -1])
    # proj = lambda x: tf.layers.conv1d(x, filters=1, kernel_size=3, padding='same')
    # output = tf.squeeze(tf.map_fn(proj, tf.expand_dims(output,axis=-1)),axis=-1)
    # output = tf.reshape(output, shape=[self.num_steps*self.batch_size, -1])

    logits = tf.layers.dense(output,vocab_size,name='softmax')
     # Reshape logits to be a 3-D tensor for sequence loss
    logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

    # Use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        input_.targets,
        tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
        average_across_timesteps=False,
        average_across_batch=True)

    # Update the cost
    self._cost = tf.reduce_sum(loss)
    self._final_state = state

    if not is_training:
      return
    decay_steps = self._input.epoch_size * 2
    self._global_step = tf.train.get_or_create_global_step()
    self._lr = tf.train.exponential_decay(learning_rate=config.learning_rate, global_step=self._global_step,
                                               decay_steps=decay_steps, decay_rate=0.5,
                                               staircase=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      config.max_grad_norm)
    self._grad_norm = tf.global_norm(grads)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=self._global_step)

  def _get_lstm_cell(self, config, is_training):
    return tf.contrib.rnn.LSTMBlockCell(
        config.hidden_size, forget_bias=0.0)

  def _build_rnn_graph_lstm(self, inputs, config, is_training):
    """Build the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def make_cell():
      cell = self._get_lstm_cell(config, is_training)
      if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=config.keep_prob)
      return cell
    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)
    self._initial_state = cell.zero_state(config.batch_size, tf.float32)
    state = self._initial_state
    outputs = []
    with tf.variable_scope("RNN"):
      for time_step in range(self.num_steps):
        if time_step > 0:
          tf.get_variable_scope().reuse_variables()
          (cell_output, state) = cell(inputs[:, time_step, :], state)
        else:
          (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    # outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=state)
    output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
    return output, state

def run_epoch(sess, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = sess.run(model._initial_state)

  fetches = {
      "cost": model._cost,
      "final_state": model._final_state
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model._input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model._initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = sess.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]
    costs += cost
    iters += model._input.num_steps

    if verbose and step % (model._input.epoch_size // 10) == 10:
      print("step: %d perplexity: %.3f speed: %.0f wps mins: %.2f" %
            (step, np.exp(costs / iters),
             iters * model._input.batch_size/
             (time.time() - start_time), float(time.time()-start_time)/60))

  return np.exp(costs / iters)

def main(_):

  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  with open(FLAGS.data_path) as finp:
    train_data, valid_data, test_data, _, _ = pickle.load(finp)
    print("-"*80)
    print("train_size: {0}".format(np.size(train_data)))
    print("valid_size: {0}".format(np.size(valid_data)))
    print(" test_size: {0}".format(np.size(test_data)))

  config = SmallConfig()
  eval_config = SmallConfig()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m._cost)
      tf.summary.scalar("Learning Rate", m._lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
      tf.summary.scalar("Validation Loss", mvalid._cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(
          config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_input)

    # Session
    saver = tf.train.Saver(max_to_keep=5)
    checkpoint_saver_hook = tf.train.CheckpointSaverHook(FLAGS.save_path, save_steps=m._input.epoch_size,
                                                         saver=saver)
    summary_hook = tf.train.SummarySaverHook(save_secs=2, output_dir=FLAGS.save_path,
                                             scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()))
    hooks = [checkpoint_saver_hook,summary_hook]
    print("Starting session")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    with tf.train.SingularMonitoredSession(hooks=hooks, checkpoint_dir=FLAGS.save_path,
                                           config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
      for i in range(config.max_max_epoch):
        print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(m._lr)))
        train_perplexity = run_epoch(sess, m, eval_op=m._train_op,
                                     verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(sess, mvalid)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(sess, mtest)
        print("Test Perplexity: %.3f" % test_perplexity)

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        saver.save(sess, FLAGS.save_path, global_step=sv.global_step)

if __name__ == "__main__":
  tf.app.run()

