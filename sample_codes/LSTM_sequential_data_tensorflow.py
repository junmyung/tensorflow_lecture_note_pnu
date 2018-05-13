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
import sys
from scipy.stats import randint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph.
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.cross_validation import KFold # use for cross validation
from sklearn.preprocessing import StandardScaler # for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline # pipeline making
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error,r2_score
import pdb
import matplotlib.pyplot as plt

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", "/home/jimmy/PycharmProjects/ENAS-tensorflow/data/household_power_consumption.txt",
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", "./checkpoints/tutorials",
                    "Model output directory.")
FLAGS = flags.FLAGS

class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 0.01
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 100
  max_epoch = 4
  max_max_epoch = 10
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  feature_size = 7

def power_input_producer(raw_data, batch_size, num_steps, name=None):

  with tf.name_scope(name, "powerProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.float32)

    feature_size = raw_data.get_shape()[-1]
    data_len = tf.size(raw_data)//feature_size
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len,raw_data.get_shape()[-1]])
    # data = tf.random_shuffle(data)
    epoch_size = (batch_len - 1) // num_steps

    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=True).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps, 0],
                         [batch_size, (i + 1) * num_steps, feature_size-1])
    x.set_shape([batch_size, num_steps, raw_data.get_shape()[-1]-1])
    y = tf.strided_slice(data, [0, (i + 1) * num_steps-1, feature_size-1],
                         [batch_size, (i + 1) * num_steps, feature_size])
    y.set_shape([batch_size,1, 1])
    y = tf.reshape(y,[batch_size,-1])
    return x, y

class powerInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size               # 배치사이즈
    self.num_steps = num_steps = config.num_steps                  # RNN 길이
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps # Epoch당 반복 횟수
    self.input_data, self.targets = power_input_producer(
        data, batch_size, num_steps, name=name)                    # input data, target data


class powerInput_eval(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = config.batch_size
    self.num_steps = config.num_steps

    test_x = []
    for idx in range(data.shape[0]-config.num_steps):
      test_x.append(np.expand_dims(data[idx:idx+config.num_steps], axis=0))

    test_x = np.concatenate(test_x, axis=0)
    self.test_X, self.test_Y = test_x[:, :, :-1], test_x[:, -1, -1]
    # self.input_data = tf.placeholder(dtype=tf.float32,shape=[self.batch_size,self.num_steps,self.test_X.shape[-1]])
    # self.targets = tf.placeholder(dtype=tf.float32,shape=[self.batch_size,1])
    self.epoch_size = len(self.test_Y)
    test_X_tf = tf.convert_to_tensor(self.test_X, dtype=tf.float32)
    test_Y_tf = tf.convert_to_tensor(self.test_Y, dtype=tf.float32)
    self.input_data, self.targets = tf.train.slice_input_producer(
      [test_X_tf, test_Y_tf], shuffle=False)

    self.input_data = tf.expand_dims(self.input_data, axis=0)
    self.targets = tf.expand_dims(self.targets, axis=0)


class powerModel(object):
  """The power model."""

  def __init__(self, is_training, config, input_):
    self._is_training = is_training
    self._input = input_
    self._rnn_params = None
    self._cell = None
    self.batch_size = input_.batch_size
    self.num_steps = input_.num_steps
    size = config.hidden_size
    feature_size = config.feature_size
    self._targets = tf.reshape(self._input.targets,[-1,1])
    # with tf.device("/cpu:0"):
    #   embedding = tf.get_variable(
    #       "embedding", [feature_size, size], dtype=tf.float32)
    #   inputs = tf.layers.conv1d(input_.input_data,filters=size,kernel_size=1)
    inputs = input_.input_data
    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    #### sequential conv1d filtering ####
    inputs = tf.layers.conv1d(inputs,200,1,padding='same')

    #### depthwise conv1d filtering ####
    # inputs = tf.transpose(inputs, [1, 0, 2]) #[20,35,200] ->[35,20,200]
    # proj = lambda x: tf.layers.conv1d(x, filters=1, kernel_size=3, padding='same')
    # inputs = tf.squeeze(tf.map_fn(proj, tf.expand_dims(inputs,axis=-1)),axis=-1)
    # inputs = tf.transpose(inputs, [1, 0, 2])

    # inputs = tf.reshape(inputs, shape=[self.batch_size*self.num_steps, -1])
    # inputs = tf.layers.dense(inputs, size)
    # inputs = tf.reshape(inputs, shape=[self.batch_size,self.num_steps, -1])

    output, state = self._build_rnn_graph_lstm(inputs, config, is_training)


    ###

    #### depthwise outputs filtering ####
    # output = tf.reshape(output, shape=[self.num_steps, self.batch_size, -1])
    # proj = lambda x: tf.layers.conv1d(x, filters=1, kernel_size=3, padding='same')
    # output = tf.squeeze(tf.map_fn(proj, tf.expand_dims(output,axis=-1)),axis=-1)
    # output = tf.reshape(output, shape=[self.num_steps*self.batch_size, -1])

    self._logits = tf.layers.dense(output,1,name='regression')

    # Use the contrib sequence loss and average over the batches
    loss = tf.losses.mean_squared_error(labels=self._targets, predictions=self._logits)

    # Update the cost
    self._cost = loss
    self._final_state = state

    if not is_training:
      return
    decay_steps = self._input.epoch_size * config.max_max_epoch//5
    self._global_step = tf.train.get_or_create_global_step()
    self._lr = tf.train.exponential_decay(learning_rate=config.learning_rate, global_step=self._global_step,
                                          decay_steps=decay_steps, decay_rate=0.5,
                                          staircase=True)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      config.max_grad_norm)
    self._grad_norm = tf.global_norm(grads)
    optimizer = tf.train.AdamOptimizer(self._lr)
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
      # for time_step in range(self.num_steps):
      #   if time_step > 0: tf.get_variable_scope().reuse_variables()
      #   (cell_output, state) = cell(inputs[:, time_step, :], state)
      #   outputs.append(cell_output)
      #

    # output = outputs[-1]
      outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=state)
      output = tf.reshape(outputs,[config.batch_size,-1])
    return output, state

def run_epoch(sess, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
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
    iters += model._input.num_steps

    if verbose and step % (model._input.epoch_size // 10) == 10:
      print("step: %d loss: %.15f speed: %.0f wps mins: %.2f" %
            (step, cost,
             iters * model._input.batch_size/
             (time.time() - start_time), float(time.time()-start_time)/60))

  return cost

def run_epoch_eval(sess, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  iters = 0
  state = sess.run(model._initial_state)
  history = {"y_hat":[],"y":[]}
  fetches = {
      "cost": model._cost,
      "final_state": model._final_state,
      "y_hat": model._logits,
      "y": model._targets
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
    iters += model._input.num_steps
    history["y_hat"].append(float(np.squeeze(vals["y_hat"])))
    history["y"].append(float(np.squeeze(vals["y"])))
    if verbose and step % (model._input.epoch_size // 10) == 10:
      print("step: %d loss: %.15f speed: %.0f wps mins: %.2f" %
            (step, cost,
             iters * model._input.batch_size/
             (time.time() - start_time), float(time.time()-start_time)/60))

  return cost, history
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  n_vars = 1 if type(data) is list else data.shape[1]
  dff = pd.DataFrame(data)
  cols, names = list(), list()
  # input sequence (t-n, ... t-1)
  for i in range(n_in, 0, -1):
    cols.append(dff.shift(i))
    names += [('var%d(t-%d)'%(j+1, i)) for j in range(n_vars)]
  # forecast sequence (t, t+1, ... t+n)
  for i in range(0, n_out):
    cols.append(dff.shift(-i))
    if i==0:
      names += [('var%d(t)'%(j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)'%(j+1, i)) for j in range(n_vars)]
  # put it all together
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  return agg

def preprosessing_dataset(data):
  """
  :param data: pandas format
  :return: train, valid, test, scaler
  """
  # convert nan to mean_value
  for j in range(0, 7):
    data.iloc[:, j] = data.iloc[:, j].fillna(data.iloc[:, j].mean())
  values = data.resample('h').mean() .values
  values = values.astype('float32')
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled = scaler.fit_transform(values)
  reframed = series_to_supervised(scaled, 1, 1)
  reframed.drop(reframed.columns[[8, 9, 10, 11, 12, 13]], axis=1, inplace=True)
  print(reframed.head())

  # split into train and test sets
  values = reframed.values

  n_train_time = 365*24
  train = values[:n_train_time*2, :]  # 첫 2년만 추출하여 학습 셋
  valid = values[2*n_train_time:3*n_train_time] # 다음 1년만 추출하여 vaild 셋
  test = values[3*n_train_time:, :]  # 나머지를 테스트 셋

  return train, valid, test, scaler

def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to power data directory")

  with open(FLAGS.data_path) as finp:
    df = pd.read_csv(finp, sep=';',
                     parse_dates={'dt': ['Date', 'Time']}, infer_datetime_format=True,
                     low_memory=False, na_values=['nan', '?'], index_col='dt')

    print("-"*80)
    print(df.info())
    print("-"*80)
    train_data, valid_data, test_data , scaler = preprosessing_dataset(df)
    print("train_size: {0}".format(np.size(train_data)))
    print("valid_size: {0}".format(np.size(valid_data)))
    print(" test_size: {0}".format(np.size(test_data)))

  config = SmallConfig()
  eval_config = SmallConfig()
  eval_config.batch_size = 1
  # eval_config.num_steps = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = powerInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = powerModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m._cost)
      tf.summary.scalar("Learning Rate", m._lr)

    with tf.name_scope("Valid"):
      valid_input = powerInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = powerModel(is_training=False, config=config, input_=valid_input)
      tf.summary.scalar("Validation Loss", mvalid._cost)

    with tf.name_scope("Test"):
      test_input = powerInput_eval(
          config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = powerModel(is_training=False, config=eval_config,
                         input_=test_input)
    # Summary
    summary_op = tf.summary.merge_all(scope='summary_op')

    # Session
    saver = tf.train.Saver(max_to_keep=5)
    checkpoint_saver_hook = tf.train.CheckpointSaverHook(FLAGS.save_path, save_steps=m._input.epoch_size,
                                                         saver=saver)
    summary_hook = tf.train.SummarySaverHook(save_steps=1, output_dir=FLAGS.save_path,
                                             scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()))
    hooks = [checkpoint_saver_hook,summary_hook]
    print("Starting session")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    with tf.train.SingularMonitoredSession(hooks=hooks, checkpoint_dir=FLAGS.save_path,
                                           config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
      for i in range(config.max_max_epoch):
        print("Epoch: %d Learning rate: %.5f" % (i + 1, sess.run(m._lr)))
        train_loss = run_epoch(sess, m, eval_op=m._train_op,
                                     verbose=True)
        print("Epoch: %d Train loss: %.15f" % (i + 1, train_loss))
        valid_loss = run_epoch(sess, mvalid)
        print("Epoch: %d Valid loss: %.15f" % (i + 1, valid_loss))

      test_loss, history = run_epoch_eval(sess, mtest)
      print("Test loss: %.15f" % test_loss)

      plt.figure(figsize=(20, 10))
      plt.plot(history['y'][:500], marker='.', label="tf_real")
      plt.plot(history['y_hat'][:500], 'r', marker='.', label="prediction")
      plt.ylabel('Global_active_power', size=15)
      plt.xlabel('Time step', size=15)
      plt.legend(fontsize=15)
      plt.show()

if __name__ == "__main__":
  tf.app.run()

