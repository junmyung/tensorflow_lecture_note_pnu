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

  
########## 잠시 보관
##### quarter#####
#-*- coding:utf-8 -*-
# Parsing dividends data from DART
import urllib.request
import urllib.parse
import xlsxwriter
import os
import time
import sys
import getopt
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re
import xlrd
import yfinance as yf
import pandas_datareader
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pandas as pd
from pykrx import stock as STOCK
from pykrx import e3

# Scrape value
def find_value(text, unit):
	return int(text.replace(" ","").replace("△","-").replace("(-)","-").replace("(","-").replace(")","").replace(",","").replace("=",""))/unit

# Draw figure of cashflows.
def draw_cashflow_figure(income_list, income_list2, year_list, op_cashflow_list, fcf_list, div_list, stock_close):
	
	for i in range(len(income_list)):
		if income_list[i] == 0.0:
			income_list[i] = income_list2[i]

	fig, ax1 = plt.subplots()

	ax1.plot(year_list, op_cashflow_list, label="Op Cashflow", color='r', marker='D')
	ax1.plot(year_list, fcf_list, label="Free Cashflow", color='y', marker='D')
	ax1.plot(year_list, income_list, label="Net Income", color='b', marker='D')
	ax1.plot(year_list, div_list, label="Dividends", color='g', marker='D')
	#ax1.plot(year_list, cash_equivalents_list, label="Cash & Cash Equivalents", color='magenta', marker='D', linestyle ='dashed')
	ax1.set_xlabel("YEAR")
	ax1.set_xticks(year_list)
	plt.legend(loc=2)

	ax2 = ax1.twinx().twiny()
	ax2.plot(stock_close, label="Stock Price", color='gray')

	#plt.title(corp)
	plt.legend(loc=4)
	plt.show()

# Draw figure of net income & assets.
def draw_corp_history(year_list, asset_sum_list, liability_sum_list, equity_sum_list, sales_list, op_income_list, net_income_list):
	
	fig, ax1 = plt.subplots()

	ax1.bar(year_list, equity_sum_list, label="Equity", color='gray')
	#ax1.plot(year_list, equity_sum_list, label="Equity", color='r', marker='D')
	ax1.plot(year_list, asset_sum_list, label="Asset", color='y', marker='D')
	ax1.plot(year_list, liability_sum_list, label="Liability", color='b', marker='D')
	ax1.plot(year_list, sales_list, label="Sales", color='r', marker='D')
	ax1.set_xlabel("YEAR")
	ax1.set_xticks(year_list)
	plt.legend(loc=2)
	
	ax2 = ax1.twinx().twiny()
	#ax2.plot(year_list, sales_list, label="Sales", color='g', marker='D', linestyle ='dashed')
	ax2.plot(year_list, op_income_list, label="Op income", color='magenta', marker='D', linestyle ='dashed')
	ax2.plot(year_list, net_income_list, label="Net income", color='g', marker='D', linestyle ='dashed')
	plt.legend(loc=4)
	
	plt.show()


# Write financial statements to Excel file.
def write_excel_file(workbook_name, dart_post_list, cashflow_list, balance_sheet_list, income_statement_list, corp, stock_code, stock_cat):

	# Write an Excel file

	#workbook = xlsxwriter.Workbook(workbook_name)
	#if os.path.isfile(os.path.join(cur_dir, workbook_name)):
	#	os.remove(os.path.join(cur_dir, workbook_name))
	workbook = xlsxwriter.Workbook(workbook_name)


	############################################################ Format 정렬
	worksheet_result = workbook.add_worksheet('DART사업보고서')
	filter_format = workbook.add_format({'bold':True,
										'fg_color': '#D7E4BC'
										})
	filter_format2 = workbook.add_format({'bold':True
										})

	percent_format = workbook.add_format({'num_format': '0.00%'})

	roe_format = workbook.add_format({'bold':True,
									  'underline': True,
									  'num_format': '0.00%'})

	num_format = workbook.add_format({'num_format':'0.00'})
	num2_format = workbook.add_format({'num_format':'#,##0.0'})
	num3_format = workbook.add_format({'num_format':'#,##0.00',
									  'fg_color':'#FCE4D6'})

	worksheet_result.set_column('A:A', 10)
	worksheet_result.set_column('B:B', 15)
	worksheet_result.set_column('C:C', 15)
	worksheet_result.set_column('D:D', 20)
	worksheet_result.set_column('H:H', 15)
	worksheet_result.set_column('I:I', 15)
	worksheet_result.set_column('J:J', 15)
	worksheet_result.set_column('K:K', 15)

	worksheet_result.write(0, 0, "날짜", filter_format)
	worksheet_result.write(0, 1, "회사명", filter_format)
	worksheet_result.write(0, 2, "분류", filter_format)
	worksheet_result.write(0, 3, "제목", filter_format)
	worksheet_result.write(0, 4, "link", filter_format)
	worksheet_result.write(0, 5, "결산년도", filter_format)
	worksheet_result.write(0, 6, "영업활동 현금흐름", filter_format)
	worksheet_result.write(0, 7, "영업에서 창출된 현금흐름", filter_format)
	worksheet_result.write(0, 8, "당기순이익", filter_format)
	worksheet_result.write(0, 9, "투자활동 현금흐름", filter_format)
	worksheet_result.write(0, 10, "유형자산의 취득", filter_format)
	worksheet_result.write(0, 11, "무형자산의 취득", filter_format)
	worksheet_result.write(0, 12, "토지의 취득", filter_format)
	worksheet_result.write(0, 13, "건물의 취득", filter_format)
	worksheet_result.write(0, 14, "구축물의 취득", filter_format)
	worksheet_result.write(0, 15, "기계장치의 취득", filter_format)
	worksheet_result.write(0, 16, "건설중인자산의 증가", filter_format)
	worksheet_result.write(0, 17, "차량운반구의 취득", filter_format)
	worksheet_result.write(0, 18, "비품의 취득", filter_format)
	worksheet_result.write(0, 19, "공구기구의 취득", filter_format)
	worksheet_result.write(0, 20, "시험 연구 설비의 취득", filter_format)
	worksheet_result.write(0, 21, "렌탈 자산의 취득", filter_format)
	worksheet_result.write(0, 22, "영업권의 취득", filter_format)
	worksheet_result.write(0, 23, "산업재산권의 취득", filter_format)
	worksheet_result.write(0, 24, "소프트웨어의 취득", filter_format)
	worksheet_result.write(0, 25, "기타의무형자산의 취득", filter_format)
	worksheet_result.write(0, 26, "투자부동산의 취득", filter_format)
	worksheet_result.write(0, 27, "관계기업투자의 취득", filter_format)
	worksheet_result.write(0, 28, "재무활동 현금흐름", filter_format)
	worksheet_result.write(0, 29, "단기차입금의 증가", filter_format)
	worksheet_result.write(0, 30, "배당금 지급", filter_format)
	worksheet_result.write(0, 31, "자기주식의 취득", filter_format)
	worksheet_result.write(0, 32, "기초현금 및 현금성자산", filter_format)
	worksheet_result.write(0, 33, "기말현금 및 현금성자산", filter_format)
	############################################################ Format 정렬


	#### 보고서 데이터 입력
	for k in range(len(dart_post_list)):
		worksheet_result.write(k+1,0, dart_post_list[k][0], num2_format) # 보고서 출간 날짜
		worksheet_result.write(k+1,1, dart_post_list[k][1], num2_format) # 종목명
		worksheet_result.write(k+1,2, dart_post_list[k][2], num2_format) # 시장
		worksheet_result.write(k+1,3, dart_post_list[k][3], num2_format) # 보고서명 ex) 분기보고서, 사업보고서
		worksheet_result.write(k+1,4, dart_post_list[k][4], num2_format) # link 주소
		worksheet_result.write(k+1,5, cashflow_list[k]['year']					, num2_format)
		worksheet_result.write(k+1,6, cashflow_list[k]['op_cashflow']				, num2_format)
		worksheet_result.write(k+1,7, cashflow_list[k]['op_cashflow_sub1']		, num2_format)
		worksheet_result.write(k+1,8, cashflow_list[k]['op_cashflow_sub2']		, num2_format)
		worksheet_result.write(k+1,9, cashflow_list[k]['invest_cashflow']			, num2_format)
		worksheet_result.write(k+1,10, cashflow_list[k]['invest_cashflow_sub1']	, num2_format)
		worksheet_result.write(k+1,11, cashflow_list[k]['invest_cashflow_sub2']	, num2_format)
		worksheet_result.write(k+1,12, cashflow_list[k]['invest_cashflow_sub3']	, num2_format)
		worksheet_result.write(k+1,13, cashflow_list[k]['invest_cashflow_sub4']	, num2_format)
		worksheet_result.write(k+1,14, cashflow_list[k]['invest_cashflow_sub5']	, num2_format)
		worksheet_result.write(k+1,15, cashflow_list[k]['invest_cashflow_sub6']	, num2_format)
		worksheet_result.write(k+1,16, cashflow_list[k]['invest_cashflow_sub7']	, num2_format)
		worksheet_result.write(k+1,17, cashflow_list[k]['invest_cashflow_sub8']	, num2_format)
		worksheet_result.write(k+1,18, cashflow_list[k]['invest_cashflow_sub9']	, num2_format)
		worksheet_result.write(k+1,19, cashflow_list[k]['invest_cashflow_sub10']	, num2_format)
		worksheet_result.write(k+1,20, cashflow_list[k]['invest_cashflow_sub11']	, num2_format)
		worksheet_result.write(k+1,21, cashflow_list[k]['invest_cashflow_sub12']	, num2_format)
		worksheet_result.write(k+1,22, cashflow_list[k]['invest_cashflow_sub13']	, num2_format)
		worksheet_result.write(k+1,23, cashflow_list[k]['invest_cashflow_sub14']	, num2_format)
		worksheet_result.write(k+1,24, cashflow_list[k]['invest_cashflow_sub15']	, num2_format)
		worksheet_result.write(k+1,25, cashflow_list[k]['invest_cashflow_sub16']	, num2_format)
		worksheet_result.write(k+1,26, cashflow_list[k]['invest_cashflow_sub17']	, num2_format)
		worksheet_result.write(k+1,27, cashflow_list[k]['invest_cashflow_sub18']	, num2_format)
		worksheet_result.write(k+1,28, cashflow_list[k]['fin_cashflow']			, num2_format)
		worksheet_result.write(k+1,29, cashflow_list[k]['fin_cashflow_sub1']		, num2_format)
		worksheet_result.write(k+1,30, cashflow_list[k]['fin_cashflow_sub2']		, num2_format)
		worksheet_result.write(k+1,31, cashflow_list[k]['fin_cashflow_sub3']		, num2_format)
		worksheet_result.write(k+1,32, cashflow_list[k]['start_cash']				, num2_format)
		worksheet_result.write(k+1,33, cashflow_list[k]['end_cash']				, num2_format)

	#### 보고서 데이터 입력
	cashflow_list.reverse() 
	worksheet_cashflow = workbook.add_worksheet('Cashflow Statement')
	
	worksheet_cashflow.set_column('A:A', 30)
	worksheet_cashflow.write(0, 0, "결산년도", filter_format)
	worksheet_cashflow.write(1, 0, "영업활동 현금흐름", filter_format)
	worksheet_cashflow.write(2, 0, "영업에서 창출된 현금흐름", filter_format2)
	worksheet_cashflow.write(3, 0, "당기순이익", filter_format2)
	worksheet_cashflow.write(4, 0, "감가상각비", filter_format2)
	worksheet_cashflow.write(5, 0, "신탁계정대", filter_format2)
	worksheet_cashflow.write(6, 0, "투자활동 현금흐름", filter_format)
	worksheet_cashflow.write(7, 0, "유형자산의 취득", filter_format2)
	worksheet_cashflow.write(8, 0, "무형자산의 취득", filter_format2)
	worksheet_cashflow.write(9, 0, "토지의 취득", filter_format2)
	worksheet_cashflow.write(10, 0, "건물의 취득", filter_format2)
	worksheet_cashflow.write(11, 0, "구축물의 취득", filter_format2)
	worksheet_cashflow.write(12, 0, "기계장치의 취득", filter_format2)
	worksheet_cashflow.write(13, 0, "건설중인자산의 증가", filter_format2)
	worksheet_cashflow.write(14, 0, "차량운반구의 취득", filter_format2)
	worksheet_cashflow.write(15, 0, "비품의 취득", filter_format2)
	worksheet_cashflow.write(16, 0, "공구기구의 취득", filter_format2)
	worksheet_cashflow.write(17, 0, "시험 연구 설비의 취득", filter_format2)
	worksheet_cashflow.write(18, 0, "렌탈 자산의 취득", filter_format2)
	worksheet_cashflow.write(19, 0, "영업권의 취득", filter_format2)
	worksheet_cashflow.write(20, 0, "산업재산권의 취득", filter_format2)
	worksheet_cashflow.write(21, 0, "소프트웨어의 취득", filter_format2)
	worksheet_cashflow.write(22, 0, "기타의무형자산의 취득", filter_format2)
	worksheet_cashflow.write(23, 0, "투자부동산의 취득", filter_format2)
	worksheet_cashflow.write(24, 0, "관계기업투자의 취득", filter_format2)
	worksheet_cashflow.write(25, 0, "재무활동 현금흐름", filter_format)
	worksheet_cashflow.write(26, 0, "단기차입금의 증가", filter_format2)
	worksheet_cashflow.write(27, 0, "배당금 지급", filter_format2)
	worksheet_cashflow.write(28, 0, "자기주식의 취득", filter_format2)
	worksheet_cashflow.write(29, 0, "기초현금 및 현금성자산", filter_format)
	worksheet_cashflow.write(30, 0, "기말현금 및 현금성자산", filter_format)
	worksheet_cashflow.write(31, 0, "당기순이익 손익계산서", filter_format2)
	worksheet_cashflow.write(32, 0, "잉여현금흐름(FCF)", filter_format)

	prev_year = 0
	j = 0

	year_list = []
	op_cashflow_list = []
	fcf_list = []
	income_list = []
	income_list2 = []
	div_list = []
	cash_equivalents_list = []

	for k in range(len(cashflow_list)):
		fcf = cashflow_list[k]['op_cashflow']
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub1'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub2'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub3'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub4'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub5'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub6'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub7'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub8'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub9'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub10'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub11'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub12'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub13'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub14'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub15'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub16'])
	
		if cashflow_list[k]['op_cashflow_sub1'] != "FINDING LINE NUMBER ERROR":

			cashflow_list[k]["op_cashflow"] = 수정해야함
			cashflow_list[k]["op_cashflow_sub1"] = 0.0
			cashflow_list[k]["op_cashflow_sub2"] = 0.0
			cashflow_list[k]["op_cashflow_sub3"] = 0.0
			cashflow_list[k]["op_cashflow_sub4"] = 0.0
			cashflow_list[k]["invest_cashflow"] = 0.0
			cashflow_list[k]["invest_cashflow_sub1"] = 0.0
			cashflow_list[k]["invest_cashflow_sub2"] = 0.0
			cashflow_list[k]["invest_cashflow_sub3"] = 0.0
			cashflow_list[k]["invest_cashflow_sub4"] = 0.0
			cashflow_list[k]["invest_cashflow_sub5"] = 0.0
			cashflow_list[k]["invest_cashflow_sub6"] = 0.0
			cashflow_list[k]["invest_cashflow_sub7"] = 0.0
			cashflow_list[k]["invest_cashflow_sub8"] = 0.0
			cashflow_list[k]["invest_cashflow_sub9"] = 0.0
			cashflow_list[k]["invest_cashflow_sub10"] = 0.0
			cashflow_list[k]["invest_cashflow_sub11"] = 0.0
			cashflow_list[k]["invest_cashflow_sub12"] = 0.0
			cashflow_list[k]["invest_cashflow_sub13"] = 0.0
			cashflow_list[k]["invest_cashflow_sub14"] = 0.0
			cashflow_list[k]["invest_cashflow_sub15"] = 0.0
			cashflow_list[k]["invest_cashflow_sub16"] = 0.0
			cashflow_list[k]["invest_cashflow_sub17"] = 0.0
			cashflow_list[k]["invest_cashflow_sub18"] = 0.0
			cashflow_list[k]["fin_cashflow"] = 0.0
			cashflow_list[k]["fin_cashflow_sub1"] = 0.0
			cashflow_list[k]["fin_cashflow_sub2"] = 0.0
			cashflow_list[k]["fin_cashflow_sub3"] = 0.0
			cashflow_list[k]["start_cash"] = 0.0
			cashflow_list[k]["end_cash"] = 0.0

			# Overwirting
			if prev_year == cashflow_list[k]['year']:
				worksheet_cashflow.write(0, j, str(cashflow_list[k]['year']))
				worksheet_cashflow.write(1, j, cashflow_list[k]['op_cashflow']				, num2_format)
				worksheet_cashflow.write(2, j, cashflow_list[k]['op_cashflow_sub1']			, num2_format)
				worksheet_cashflow.write(3, j, cashflow_list[k]['op_cashflow_sub2']			, num2_format)
				worksheet_cashflow.write(4, j, cashflow_list[k]['op_cashflow_sub3']			, num2_format)
				worksheet_cashflow.write(5, j, cashflow_list[k]['op_cashflow_sub4']			, num2_format)
				worksheet_cashflow.write(6, j, cashflow_list[k]['invest_cashflow']			, num2_format)
				worksheet_cashflow.write(7, j, cashflow_list[k]['invest_cashflow_sub1']		, num2_format)
				worksheet_cashflow.write(8, j, cashflow_list[k]['invest_cashflow_sub2']		, num2_format)
				worksheet_cashflow.write(9, j, cashflow_list[k]['invest_cashflow_sub3']		, num2_format)
				worksheet_cashflow.write(10, j, cashflow_list[k]['invest_cashflow_sub4']	, num2_format)
				worksheet_cashflow.write(11, j, cashflow_list[k]['invest_cashflow_sub5']	, num2_format)
				worksheet_cashflow.write(12, j, cashflow_list[k]['invest_cashflow_sub6']	, num2_format)
				worksheet_cashflow.write(13, j, cashflow_list[k]['invest_cashflow_sub7']	, num2_format)
				worksheet_cashflow.write(14, j, cashflow_list[k]['invest_cashflow_sub8']	, num2_format)
				worksheet_cashflow.write(15, j, cashflow_list[k]['invest_cashflow_sub9']	, num2_format)
				worksheet_cashflow.write(16, j, cashflow_list[k]['invest_cashflow_sub10']	, num2_format)
				worksheet_cashflow.write(17, j, cashflow_list[k]['invest_cashflow_sub11']	, num2_format)
				worksheet_cashflow.write(18, j, cashflow_list[k]['invest_cashflow_sub12']	, num2_format)
				worksheet_cashflow.write(19, j, cashflow_list[k]['invest_cashflow_sub13']	, num2_format)
				worksheet_cashflow.write(20, j, cashflow_list[k]['invest_cashflow_sub14']	, num2_format)
				worksheet_cashflow.write(21, j, cashflow_list[k]['invest_cashflow_sub15']	, num2_format)
				worksheet_cashflow.write(22, j, cashflow_list[k]['invest_cashflow_sub16']	, num2_format)
				worksheet_cashflow.write(23, j, cashflow_list[k]['invest_cashflow_sub17']	, num2_format)
				worksheet_cashflow.write(24, j, cashflow_list[k]['invest_cashflow_sub18']	, num2_format)
				worksheet_cashflow.write(25, j, cashflow_list[k]['fin_cashflow']			, num2_format)
				worksheet_cashflow.write(26, j, cashflow_list[k]['fin_cashflow_sub1']		, num2_format)
				worksheet_cashflow.write(27, j, cashflow_list[k]['fin_cashflow_sub2']		, num2_format)
				worksheet_cashflow.write(28, j, cashflow_list[k]['fin_cashflow_sub3']		, num2_format)
				worksheet_cashflow.write(29, j, cashflow_list[k]['start_cash']				, num2_format)
				worksheet_cashflow.write(30, j, cashflow_list[k]['end_cash']				, num2_format)
				worksheet_cashflow.write(31, j, cashflow_list[k]['net_income']				, num2_format)
				worksheet_cashflow.write(32, j, fcf, num2_format)
				
				year_list[-1] = cashflow_list[k]['year']
				op_cashflow_list[-1] = cashflow_list[k]['op_cashflow']
				fcf_list[-1] = fcf
				income_list[-1] = cashflow_list[k]['op_cashflow_sub2']
				income_list2[-1] = cashflow_list[k]['net_income']
				div_list[-1] = abs(cashflow_list[k]['fin_cashflow_sub2'])
				cash_equivalents_list[-1] = cashflow_list[k]['end_cash']
			else:
				worksheet_cashflow.write(0, j+1, str(cashflow_list[k]['year']))
				worksheet_cashflow.write(1, j+1, cashflow_list[k]['op_cashflow']			, num2_format)
				worksheet_cashflow.write(2, j+1, cashflow_list[k]['op_cashflow_sub1']		, num2_format)
				worksheet_cashflow.write(3, j+1, cashflow_list[k]['op_cashflow_sub2']		, num2_format)
				worksheet_cashflow.write(4, j+1, cashflow_list[k]['op_cashflow_sub3']		, num2_format)
				worksheet_cashflow.write(5, j+1, cashflow_list[k]['op_cashflow_sub4']		, num2_format)
				worksheet_cashflow.write(6, j+1, cashflow_list[k]['invest_cashflow']		, num2_format)
				worksheet_cashflow.write(7, j+1, cashflow_list[k]['invest_cashflow_sub1']	, num2_format)
				worksheet_cashflow.write(8, j+1, cashflow_list[k]['invest_cashflow_sub2']	, num2_format)
				worksheet_cashflow.write(9, j+1, cashflow_list[k]['invest_cashflow_sub3']	, num2_format)
				worksheet_cashflow.write(10, j+1, cashflow_list[k]['invest_cashflow_sub4']	, num2_format)
				worksheet_cashflow.write(11, j+1, cashflow_list[k]['invest_cashflow_sub5']	, num2_format)
				worksheet_cashflow.write(12, j+1, cashflow_list[k]['invest_cashflow_sub6']	, num2_format)
				worksheet_cashflow.write(13, j+1, cashflow_list[k]['invest_cashflow_sub7']	, num2_format)
				worksheet_cashflow.write(14, j+1, cashflow_list[k]['invest_cashflow_sub8']	, num2_format)
				worksheet_cashflow.write(15, j+1, cashflow_list[k]['invest_cashflow_sub9']	, num2_format)
				worksheet_cashflow.write(16, j+1, cashflow_list[k]['invest_cashflow_sub10']	, num2_format)
				worksheet_cashflow.write(17, j+1, cashflow_list[k]['invest_cashflow_sub11']	, num2_format)
				worksheet_cashflow.write(18, j+1, cashflow_list[k]['invest_cashflow_sub12']	, num2_format)
				worksheet_cashflow.write(19, j+1, cashflow_list[k]['invest_cashflow_sub13']	, num2_format)
				worksheet_cashflow.write(20, j+1, cashflow_list[k]['invest_cashflow_sub14']	, num2_format)
				worksheet_cashflow.write(21, j+1, cashflow_list[k]['invest_cashflow_sub15']	, num2_format)
				worksheet_cashflow.write(22, j+1, cashflow_list[k]['invest_cashflow_sub16']	, num2_format)
				worksheet_cashflow.write(23, j+1, cashflow_list[k]['invest_cashflow_sub17']	, num2_format)
				worksheet_cashflow.write(24, j+1, cashflow_list[k]['invest_cashflow_sub18']	, num2_format)
				worksheet_cashflow.write(25, j+1, cashflow_list[k]['fin_cashflow']			, num2_format)
				worksheet_cashflow.write(26, j+1, cashflow_list[k]['fin_cashflow_sub1']		, num2_format)
				worksheet_cashflow.write(27, j+1, cashflow_list[k]['fin_cashflow_sub2']		, num2_format)
				worksheet_cashflow.write(28, j+1, cashflow_list[k]['fin_cashflow_sub3']		, num2_format)
				worksheet_cashflow.write(29, j+1, cashflow_list[k]['start_cash']			, num2_format)
				worksheet_cashflow.write(30, j+1, cashflow_list[k]['end_cash']				, num2_format)
				worksheet_cashflow.write(31, j+1, cashflow_list[k]['net_income']			, num2_format)
				worksheet_cashflow.write(32, j+1, fcf, num2_format)
			
				year_list.append(cashflow_list[k]['year'])
				op_cashflow_list.append(cashflow_list[k]['op_cashflow'])
				fcf_list.append(fcf)
				income_list.append(cashflow_list[k]['op_cashflow_sub2'])
				income_list2.append(cashflow_list[k]['net_income'])
				div_list.append(abs(cashflow_list[k]['fin_cashflow_sub2']))
				cash_equivalents_list.append(cashflow_list[k]['end_cash'])
				j = j+1
		
			prev_year = cashflow_list[k]['year']

	# Balance sheet
	balance_sheet_list.reverse() 
	worksheet_bs= workbook.add_worksheet('Balance Sheet')
	
	prev_year = 0
	j = 0

	asset_sum_list = []
	liability_sum_list = []
	equity_sum_list = []

	worksheet_bs.set_column('A:A', 30)
	worksheet_bs.write(0, 0, "결산년도", filter_format)
	worksheet_bs.write(1, 0, "유동자산", filter_format)
	worksheet_bs.write(2, 0, "현금 및 현금성 자산", filter_format2)
	worksheet_bs.write(3, 0, "매출채권", filter_format2)
	worksheet_bs.write(4, 0, "재고자산", filter_format2)
	worksheet_bs.write(5, 0, "비유동자산", filter_format)
	worksheet_bs.write(6, 0, "유형자산", filter_format2)
	worksheet_bs.write(7, 0, "무형자산", filter_format2)
	worksheet_bs.write(8, 0, "자산총계", filter_format)
	worksheet_bs.write(9, 0, "유동부채", filter_format)
	worksheet_bs.write(10, 0, "매입채무", filter_format2)
	worksheet_bs.write(11, 0, "단기차입금", filter_format2)
	worksheet_bs.write(12, 0, "미지급금", filter_format2)
	worksheet_bs.write(13, 0, "비유동부채", filter_format)
	worksheet_bs.write(14, 0, "사채", filter_format2)
	worksheet_bs.write(15, 0, "장기차입금", filter_format2)
	worksheet_bs.write(16, 0, "장기미지급금", filter_format2)
	worksheet_bs.write(17, 0, "이연법인세부채", filter_format2)
	worksheet_bs.write(18, 0, "부채총계", filter_format)
	worksheet_bs.write(19, 0, "자본금", filter_format2)
	worksheet_bs.write(20, 0, "주식발행초과금", filter_format2)
	worksheet_bs.write(21, 0, "자본잉여금", filter_format2)
	worksheet_bs.write(22, 0, "이익잉여금", filter_format2)
	worksheet_bs.write(23, 0, "자본총계", filter_format)
	
	for k in range(len(balance_sheet_list)):
		if balance_sheet_list[k]['asset_current_sub1'] != "FINDING LINE NUMBER ERROR":
			# Overwirting
			if prev_year == balance_sheet_list[k]['year']:
				asset_sum_list[-1] = balance_sheet_list[k]['asset_sum']
				liability_sum_list[-1] = balance_sheet_list[k]['liability_sum']
				equity_sum_list[-1] = balance_sheet_list[k]['equity_sum']
				w = j
			else:
				asset_sum_list.append(balance_sheet_list[k]['asset_sum'])
				liability_sum_list.append(balance_sheet_list[k]['liability_sum'])
				equity_sum_list.append(balance_sheet_list[k]['equity_sum'])
				w = j+1

			worksheet_bs.write(0, w, str(balance_sheet_list[k]['year']))
			worksheet_bs.write(1, w, balance_sheet_list[k]['asset_current']					, num2_format)
			worksheet_bs.write(2, w, balance_sheet_list[k]['asset_current_sub1']			, num2_format)
			worksheet_bs.write(3, w, balance_sheet_list[k]['asset_current_sub2']			, num2_format)
			worksheet_bs.write(4, w, balance_sheet_list[k]['asset_current_sub3']			, num2_format)
			worksheet_bs.write(5, w, balance_sheet_list[k]['asset_non_current']				, num2_format)
			worksheet_bs.write(6, w, balance_sheet_list[k]['asset_non_current_sub1']		, num2_format)
			worksheet_bs.write(7, w, balance_sheet_list[k]['asset_non_current_sub2']		, num2_format)
			worksheet_bs.write(8, w, balance_sheet_list[k]['asset_sum']						, num2_format)
			worksheet_bs.write(9, w, balance_sheet_list[k]['liability_current']				, num2_format)
			worksheet_bs.write(10, w, balance_sheet_list[k]['liability_current_sub1']		, num2_format)
			worksheet_bs.write(11, w, balance_sheet_list[k]['liability_current_sub2']		, num2_format)
			worksheet_bs.write(12, w, balance_sheet_list[k]['liability_current_sub3']		, num2_format)
			worksheet_bs.write(13, w, balance_sheet_list[k]['liability_non_current']		, num2_format)
			worksheet_bs.write(14, w, balance_sheet_list[k]['liability_non_current_sub1']	, num2_format)
			worksheet_bs.write(15, w, balance_sheet_list[k]['liability_non_current_sub2']	, num2_format)
			worksheet_bs.write(16, w, balance_sheet_list[k]['liability_non_current_sub3']	, num2_format)
			worksheet_bs.write(17, w, balance_sheet_list[k]['liability_non_current_sub4']	, num2_format)
			worksheet_bs.write(18, w, balance_sheet_list[k]['liability_sum']				, num2_format)
			worksheet_bs.write(19, w, balance_sheet_list[k]['equity']						, num2_format)
			worksheet_bs.write(20, w, balance_sheet_list[k]['equity_sub1']					, num2_format)
			worksheet_bs.write(21, w, balance_sheet_list[k]['equity_sub3']					, num2_format)
			worksheet_bs.write(22, w, balance_sheet_list[k]['equity_sub2']					, num2_format)
			worksheet_bs.write(23, w, balance_sheet_list[k]['equity_sum']					, num2_format)
			
			if prev_year != balance_sheet_list[k]['year']:
				j = j+1
			prev_year = balance_sheet_list[k]['year']

	# Income statement
	income_statement_list.reverse() 
	worksheet_income= workbook.add_worksheet('Income Statement')

	prev_year = 0
	j = 0

	sales_list = []
	op_income_list = []
	net_income_list = []
	
	worksheet_income.set_column('A:A', 30)
	worksheet_income.write(0, 0, "결산년도", filter_format)
	worksheet_income.write(1, 0, "매출액", filter_format)
	worksheet_income.write(2, 0, "매출원가", filter_format2)
	worksheet_income.write(3, 0, "매출총이익", filter_format2)
	worksheet_income.write(4, 0, "판매비와관리비", filter_format2)
	worksheet_income.write(5, 0, "영업수익", filter_format)
	worksheet_income.write(6, 0, "영업비용", filter_format2)
	worksheet_income.write(7, 0, "영업이익", filter_format)
	worksheet_income.write(8, 0, "기타수익", filter_format2)
	worksheet_income.write(9, 0, "기타비용", filter_format2)
	worksheet_income.write(10, 0, "금융수익", filter_format2)
	worksheet_income.write(11, 0, "금융비용", filter_format2)
	worksheet_income.write(12, 0, "영업외수익", filter_format2)
	worksheet_income.write(13, 0, "영업외비용", filter_format2)
	worksheet_income.write(14, 0, "법인세비용차감전순이익", filter_format)
	worksheet_income.write(15, 0, "법인세비용", filter_format2)
	worksheet_income.write(16, 0, "당기순이익", filter_format)
	#worksheet_income.write(17, 0, "기본주당이익", filter_format)

	for k in range(len(income_statement_list)):
		if income_statement_list[k]['sales_sub1'] != "FINDING LINE NUMBER ERROR":
			# Overwirting
			if prev_year == income_statement_list[k]['year']:
				w = j
				sales_list[-1]			= income_statement_list[k]['sales']
				op_income_list[-1]		= income_statement_list[k]['op_income']
				net_income_list[-1]		= income_statement_list[k]['net_income']
			else:
				sales_list.append(income_statement_list[k]['sales'])
				op_income_list.append(income_statement_list[k]['op_income'])
				net_income_list.append(income_statement_list[k]['net_income'])
				w = j+1

			if int(income_statement_list[k]['year'][-2:]) == 12:
				worksheet_income.write(0, w, str(income_statement_list[k]['year']))
				worksheet_income.write(1, w, income_statement_list[k]['sales']-income_statement_list[k-1]['sales']-income_statement_list[k-2]['sales']-income_statement_list[k-3]['sales'] 			, num2_format)
				worksheet_income.write(2, w, income_statement_list[k]['sales_sub1']-income_statement_list[k-1]['sales_sub1']-income_statement_list[k-2]['sales_sub1']	-income_statement_list[k-3]['sales_sub1']	, num2_format)
				worksheet_income.write(3, w, income_statement_list[k]['sales_sub2']-income_statement_list[k-1]['sales_sub2']-income_statement_list[k-2]['sales_sub2']-income_statement_list[k-3]['sales_sub2']			, num2_format)
				worksheet_income.write(4, w, income_statement_list[k]['sales_sub3']-income_statement_list[k-1]['sales_sub3']-income_statement_list[k-2]['sales_sub3']-income_statement_list[k-3]['sales_sub3']		, num2_format)
				worksheet_income.write(5, w, income_statement_list[k]['sales2']	-income_statement_list[k-1]['sales2']-income_statement_list[k-2]['sales2']	-income_statement_list[k-3]['sales2']	, num2_format)
				worksheet_income.write(6, w, income_statement_list[k]['sales2_sub1']-income_statement_list[k-1]['sales2_sub1']-income_statement_list[k-2]['sales2_sub1']-income_statement_list[k-3]['sales2_sub1']		, num2_format)
				worksheet_income.write(7, w, income_statement_list[k]['op_income']	-income_statement_list[k-1]['op_income']-income_statement_list[k-2]['op_income']-income_statement_list[k-3]['op_income']		, num2_format)
				worksheet_income.write(8, w, income_statement_list[k]['op_income_sub1']-income_statement_list[k-1]['op_income_sub1']-income_statement_list[k-2]['op_income_sub1']-income_statement_list[k-3]['op_income_sub1']	, num2_format)
				worksheet_income.write(9, w, income_statement_list[k]['op_income_sub2']-income_statement_list[k-1]['op_income_sub2']-income_statement_list[k-2]['op_income_sub2']-income_statement_list[k-3]['op_income_sub2']	, num2_format)
				worksheet_income.write(10, w, income_statement_list[k]['op_income_sub3']-income_statement_list[k-1]['op_income_sub3']-income_statement_list[k-2]['op_income_sub3']-income_statement_list[k-3]['op_income_sub3']	, num2_format)
				worksheet_income.write(11, w, income_statement_list[k]['op_income_sub4']-income_statement_list[k-1]['op_income_sub4']-income_statement_list[k-2]['op_income_sub4']-income_statement_list[k-3]['op_income_sub4']	, num2_format)
				worksheet_income.write(12, w, income_statement_list[k]['op_income_sub6']-income_statement_list[k-1]['op_income_sub6']-income_statement_list[k-2]['op_income_sub6']-income_statement_list[k-3]['op_income_sub6']	, num2_format)
				worksheet_income.write(13, w, income_statement_list[k]['op_income_sub7']-income_statement_list[k-1]['op_income_sub7']-income_statement_list[k-2]['op_income_sub7']-income_statement_list[k-3]['op_income_sub7']	, num2_format)
				worksheet_income.write(14, w, income_statement_list[k]['op_income_sub5']-income_statement_list[k-1]['op_income_sub5']-income_statement_list[k-2]['op_income_sub5']-income_statement_list[k-3]['op_income_sub5']	, num2_format)
				worksheet_income.write(15, w, income_statement_list[k]['tax']-income_statement_list[k-1]['tax']-income_statement_list[k-2]['tax']-income_statement_list[k-3]['tax']				, num2_format)
				worksheet_income.write(16, w, income_statement_list[k]['net_income']-income_statement_list[k-1]['net_income']-income_statement_list[k-2]['net_income']-income_statement_list[k-3]['net_income']		, num2_format)
				#worksheet_income.write(17, w, income_statement_list[k]['eps']				, num2_format)
			else:
				worksheet_income.write(0, w, str(income_statement_list[k]['year']))
				worksheet_income.write(1, w, income_statement_list[k] ['sales']  			, num2_format)
				worksheet_income.write(2, w, income_statement_list[k] ['sales_sub1']		, num2_format)
				worksheet_income.write(3, w, income_statement_list[k] ['sales_sub2']		, num2_format)
				worksheet_income.write(4, w, income_statement_list[k] ['sales_sub3']		, num2_format)
				worksheet_income.write(5, w, income_statement_list[k] ['sales2']			, num2_format)
				worksheet_income.write(6, w, income_statement_list[k] ['sales2_sub1']		, num2_format)
				worksheet_income.write(7, w, income_statement_list[k] ['op_income']			, num2_format)
				worksheet_income.write(8, w, income_statement_list[k] ['op_income_sub1']	, num2_format)
				worksheet_income.write(9, w, income_statement_list[k] ['op_income_sub2']	, num2_format)
				worksheet_income.write(10, w, income_statement_list[k] ['op_income_sub3']	, num2_format)
				worksheet_income.write(11, w, income_statement_list[k] ['op_income_sub4']	, num2_format)
				worksheet_income.write(12, w, income_statement_list[k]['op_income_sub6']	, num2_format)
				worksheet_income.write(13, w, income_statement_list[k]['op_income_sub7']	, num2_format)
				worksheet_income.write(14, w, income_statement_list[k]['op_income_sub5']	, num2_format)
				worksheet_income.write(15, w, income_statement_list[k]['tax']				, num2_format)
				worksheet_income.write(16, w, income_statement_list[k]['net_income']		, num2_format)
				#worksheet_income.write(17, w, income_statement_list[k]['eps']				, num2_format)
			
			if prev_year != income_statement_list[k]['year']:
				j = j+1
			prev_year = income_statement_list[k]['year']
	
	j = 0
	
	# Chart WORKSHEET	
	#chart = workbook.add_chart({'type':'line'})
	#chart.add_series({
	#				'categories':'=cashflow!$B$1:$Q$1',
	#				'name':'=cashflow!A2',
	#				'values':'=cashflow!$B$2:$Q$2',
	#				'marker':{'type': 'diamond'}
	#				})
	#chart.add_series({
	#				'name':'=cashflow!A4',
	#				'values':'=cashflow!$B$4:$Q$4',
	#				'marker':{'type': 'diamond'}
	#				})
	#chart.add_series({
	#				'name':'=cashflow!A26',
	#				'values':'=cashflow!$B$26:$Q$26',
	#				'marker':{'type': 'diamond'}
	#				})
	#chart.set_legend({'font':{'bold':1}})
	#chart.set_x_axis({'name':"결산년도"})
	#chart.set_y_axis({'name':"단위:억원"})
	#chart.set_title({'name':corp})

	#worksheet_cashflow.insert_chart('C30', chart)

	old_year = cashflow_list[0]['year']

	if (stock_code != ""):
		yf.pdr_override()
		start_date = str(old_year).replace('.', '-')+'-01'
		if stock_cat == "KOSPI":
			ticker = stock_code+'.KS'
		elif stock_cat == 'KOSDAQ':
			ticker = stock_code+'.KQ'

		print("ticker", ticker)
		print("start date", start_date)
		stock_read = pandas_datareader.data.get_data_yahoo(ticker, start_date)
		stock_close = stock_read['Close'].values
		stock_datetime64 = stock_read.index.values

		stock_date = []

		for date in stock_datetime64:
			unix_epoch = np.datetime64(0, 's')
			one_second = np.timedelta64(1, 's')
			seconds_since_epoch = (date - unix_epoch) / one_second
			
			day = datetime.utcfromtimestamp(seconds_since_epoch)
			stock_date.append(day.strftime('%Y-%m-%d'))

		worksheet_stock = workbook.add_worksheet('stock_chart')

		worksheet_stock.write(0, 0, "date")
		worksheet_stock.write(0, 1, "Close")
		
		for i in range(len(stock_close)):
			worksheet_stock.write(i+1, 0, stock_date[i])
			worksheet_stock.write(i+1, 1, stock_close[i])
		
		chart = workbook.add_chart({'type':'line'})
		chart.add_series({
						'categories':'=stock_chart!$A$2:$A$'+str(len(stock_close)+1),
						'name':'=stock_chart!B1',
						'values':'=stock_chart!$B$2:$B$'+str(len(stock_close)+1)
						})
		chart.set_size({'x_scale': 2, 'y_scale': 1})
		worksheet_stock.insert_chart('D3', chart)

	workbook.close()
	# Deactivate
	# draw_cashflow_figure(income_list, income_list2, year_list, op_cashflow_list, fcf_list, div_list, stock_close)
	# draw_corp_history(year_list, asset_sum_list, liability_sum_list, equity_sum_list, sales_list, op_income_list, net_income_list)

# Get information of balance sheet
def scrape_balance_sheet(balance_sheet_table, year, unit):

	#유동자산
	##현금및현금성자산
	##매출채권
	##재고자산
	#비유동자산
	##유형자산
	##무형자산
	#자산총계
	#유동부채
	##매입채무
	##단기차입금
	##미지급금
	#비유동부채
	##사채
	##장기차입금
	##장기미지급금
	##이연법인세부채
	#부채총계
	##자본금
	##주식발행초과금
	##이익잉여금
	#자본총계

	re_asset_list = []

	re_asset_current				=	re.compile("^유[ \s]*동[ \s]*자[ \s]*산([ \s]*합[ \s]*계)*|\.[ \s]*유[ \s]*동[ \s]*자[ \s]*산([ \s]*합[ \s]*계)*")
	re_asset_current_sub1			=	re.compile("현[ \s]*금[ \s]*및[ \s]*현[ \s]*금[ \s]*((성[ \s]*자[ \s]*산)|(등[ \s]*가[ \s]*물))")
	re_asset_current_sub2			=	re.compile("매[ \s]*출[ \s]*채[ \s]*권")
	re_asset_current_sub3			=	re.compile("재[ \s]*고[ \s]*자[ \s]*산")
	re_asset_non_current			=	re.compile("비[ \s]*유[ \s]*동[ \s]*자[ \s]*산|고[ \s]*정[ \s]*자[ \s]*산([ \s]*합[ \s]*계)*")
	re_asset_non_current_sub1		=	re.compile("유[ \s]*형[ \s]*자[ \s]*산")
	re_asset_non_current_sub2		=	re.compile("무[ \s]*형[ \s]*자[ \s]*산")
	re_asset_sum					=	re.compile("자[ \s]*산[ \s]*총[ \s]*계([ \s]*합[ \s]*계)*")
	re_liability_current			=	re.compile("^유[ \s]*동[ \s]*부[ \s]*채([ \s]*합[ \s]*계)*|\.[ \s]*유[ \s]*동[ \s]*부[ \s]*채([ \s]*합[ \s]*계)*")
	re_liability_current_sub1		=	re.compile("매[ \s]*입[ \s]*채[ \s]*무[ \s]*")
	re_liability_current_sub2		=	re.compile("단[ \s]*기[ \s]*차[ \s]*입[ \s]*금")
	re_liability_current_sub3		=	re.compile("^미[ \s]*지[ \s]*급[ \s]*금[ \s]*")
	re_liability_non_current		=	re.compile("^비[ \s]*유[ \s]*동[ \s]*부[ \s]*채|\.[ \s]*비[ \s]*유[ \s]*동[ \s]*부[ \s]*채|고[ \s]*정[ \s]*부[ \s]*채")
	re_liability_non_current_sub1	=	re.compile("사[ \s]*채[ \s]*")
	re_liability_non_current_sub2	=	re.compile("장[ \s]*기[ \s]*차[ \s]*입[ \s]*금")
	re_liability_non_current_sub3	=	re.compile("장[ \s]*기[ \s]*미[ \s]*지[ \s]*급[ \s]*금")
	re_liability_non_current_sub4	=	re.compile("이[ \s]*연[ \s]*법[ \s]*인[ \s]*세[ \s]*부[ \s]*채")
	re_liability_sum				=	re.compile("^부[ \s]*채[ \s]*총[ \s]*계([ \s]*합[ \s]*계)*|\.[ \s]*부[ \s]*채[ \s]*총[ \s]*계([ \s]*합[ \s]*계)*")
	re_equity						=	re.compile("자[ \s]*본[ \s]*금")
	re_equity_sub1					=	re.compile("주[ \s]*식[ \s]*발[ \s]*행[ \s]*초[ \s]*과[ \s]*금")
	re_equity_sub3					=	re.compile("자[ \s]*본[ \s]*잉[ \s]*여[ \s]*금")
	re_equity_sub2					=	re.compile("이[ \s]*익[ \s]*잉[ \s]*여[ \s]*금")
	re_equity_sum					=	re.compile("^자[ \s]*본[ \s]*총[ \s]*계([ \s]*합[ \s]*계)*|\.[ \s]*자[ \s]*본[ \s]*총[ \s]*계([ \s]*합[ \s]*계)*")

	re_asset_list.append(re_asset_current)
	re_asset_list.append(re_asset_current_sub1)
	re_asset_list.append(re_asset_current_sub2)		
	re_asset_list.append(re_asset_current_sub3)		
	re_asset_list.append(re_asset_non_current)
	re_asset_list.append(re_asset_non_current_sub1)	
	re_asset_list.append(re_asset_non_current_sub2)	
	re_asset_list.append(re_asset_sum)
	re_asset_list.append(re_liability_current)
	re_asset_list.append(re_liability_current_sub1)
	re_asset_list.append(re_liability_current_sub2)		
	re_asset_list.append(re_liability_current_sub3)		
	re_asset_list.append(re_liability_non_current)
	re_asset_list.append(re_liability_non_current_sub1)	
	re_asset_list.append(re_liability_non_current_sub2)	
	re_asset_list.append(re_liability_non_current_sub3)	
	re_asset_list.append(re_liability_non_current_sub4)	
	re_asset_list.append(re_liability_sum)
	re_asset_list.append(re_equity)
	re_asset_list.append(re_equity_sub1)
	re_asset_list.append(re_equity_sub3)
	re_asset_list.append(re_equity_sub2)		
	re_asset_list.append(re_equity_sum)

	balance_sheet_sub_list = {}
	balance_sheet_sub_list["asset_current"]					=	0.0
	balance_sheet_sub_list["asset_current_sub1"]			=	0.0
	balance_sheet_sub_list["asset_current_sub2"]			=	0.0
	balance_sheet_sub_list["asset_current_sub3"]			=	0.0
	balance_sheet_sub_list["asset_non_current"]				=	0.0
	balance_sheet_sub_list["asset_non_current_sub1"]		=	0.0
	balance_sheet_sub_list["asset_non_current_sub2"]		=	0.0
	balance_sheet_sub_list["asset_sum"]						=	0.0
	balance_sheet_sub_list['year']							=	year
	balance_sheet_sub_list["liability_current"]				=	0.0
	balance_sheet_sub_list["liability_current_sub1"]		=	0.0
	balance_sheet_sub_list["liability_current_sub2"]		=	0.0
	balance_sheet_sub_list["liability_current_sub3"]		=	0.0
	balance_sheet_sub_list["liability_non_current"]			=	0.0
	balance_sheet_sub_list["liability_non_current_sub1"]	=	0.0
	balance_sheet_sub_list["liability_non_current_sub2"]	=	0.0
	balance_sheet_sub_list["liability_non_current_sub3"]	=	0.0
	balance_sheet_sub_list["liability_non_current_sub4"]	=	0.0
	balance_sheet_sub_list["liability_sum"]					=	0.0
	balance_sheet_sub_list["equity"]						=	0.0
	balance_sheet_sub_list["equity_sub1"]					=	0.0
	balance_sheet_sub_list["equity_sub3"]					=	0.0
	balance_sheet_sub_list["equity_sub2"]					=	0.0
	balance_sheet_sub_list["equity_sum"]					=	0.0

	balance_sheet_key_list = []
	
	balance_sheet_key_list.append("asset_current")
	balance_sheet_key_list.append("asset_current_sub1")
	balance_sheet_key_list.append("asset_current_sub2")
	balance_sheet_key_list.append("asset_current_sub3")
	balance_sheet_key_list.append("asset_non_current")
	balance_sheet_key_list.append("asset_non_current_sub1")
	balance_sheet_key_list.append("asset_non_current_sub2")
	balance_sheet_key_list.append("asset_sum")
	balance_sheet_key_list.append("liability_current")			
	balance_sheet_key_list.append("liability_current_sub1")		
	balance_sheet_key_list.append("liability_current_sub2")		
	balance_sheet_key_list.append("liability_current_sub3")		
	balance_sheet_key_list.append("liability_non_current")		
	balance_sheet_key_list.append("liability_non_current_sub1")	
	balance_sheet_key_list.append("liability_non_current_sub2")	
	balance_sheet_key_list.append("liability_non_current_sub3")	
	balance_sheet_key_list.append("liability_non_current_sub4")	
	balance_sheet_key_list.append("liability_sum")				
	balance_sheet_key_list.append("equity")						
	balance_sheet_key_list.append("equity_sub1")				
	balance_sheet_key_list.append("equity_sub3")				
	balance_sheet_key_list.append("equity_sub2")				
	balance_sheet_key_list.append("equity_sum")					
	
	trs = balance_sheet_table.findAll("tr")

	# Balance sheet statement
	if (len(trs) != 2):
		for tr in trs:
			#print("trs", len(trs))
			tds = tr.findAll("td")
			#print("tds", len(tds))
			try:
				if (len(tds) != 0):
					#print(tds[0].text.strip())
					value = 0.0
					for i in range(len(re_asset_list)):
						if re_asset_list[i].search(tds[0].text.strip()):
							if len(tds)>4:
								if (tds[1].text.strip() != '') and (tds[1].text.strip() != '-'):
									value = find_value(tds[1].text.strip(), unit)
									break # for i in len(re_asset_list)
								elif (tds[2].text.strip() != '') and (tds[2].text.strip() != '-'):
									value = find_value(tds[2].text.strip(), unit)
									break # for i in len(re_asset_list)
							else:
								if (tds[1].text.strip() != '') and (tds[1].text.strip() != '-'):
									value = find_value(tds[1].text.strip(), unit)
									break # for i in len(re_asset_list)
					if value != 0.0 and balance_sheet_sub_list[balance_sheet_key_list[i]] == 0.0:
						balance_sheet_sub_list[balance_sheet_key_list[i]] = value
			except Exception as e:
				print("NET INCOME PARSING ERROR in Balance sheet")
				print(e)
	# Special case
	## if (len(trs) != 2):
	else:	
		tr = trs[1]
		tds = tr.findAll("td")
		
		index_col = []
		prev = 0
		for a in tds[0].childGenerator():
			if (str(a) == "<br/>"):
				if (prev == 1):
					index_col.append('')	
				prev = 1
			else:
				prev = 0
				index_col.append(str(a).strip())	
		data_col = []
		prev = 0
		for b in tds[1].childGenerator():
			if (str(b) == "<br/>"):
				if (prev == 1):
					data_col.append('')	
				prev = 1
			else:
				data_col.append(str(b))	
				prev = 0
		data_col2 = []
		prev = 0
		for b in tds[2].childGenerator():
			if (str(b) == "<br/>"):
				if (prev == 1):
					data_col2.append('')	
				prev = 1
			else:
				data_col2.append(str(b))	
				prev = 0

		print("##################################################")
		print(index_col)
		print(data_col)
		print(data_col2)
		print(len(index_col))
		print(len(data_col))
		index_cnt = 0

		for (index) in (index_col):
			try:
				value = 0.0
				for i in range(len(re_asset_list)):
					if re_asset_list[i].search(index):
						if len(tds)>4:
							if (data_col[index_cnt].strip() != '') and (data_col[index_cnt].strip() != '-'):
								value = find_value(data_col[index_cnt], unit)
								break
							elif (data_col2[index_cnt].strip() != '') and (data_col2[index_cnt].strip() != '-'):
								value = find_value(data_col2[index_cnt], unit)
								break
						else:
							if (data_col[index_cnt].strip() != '') and (data_col[index_cnt].strip() != '-'):
								value = find_value(data_col[index_cnt], unit)
								break
				if value != 0.0 and balance_sheet_sub_list[balance_sheet_key_list[i]] == 0.0:
					balance_sheet_sub_list[balance_sheet_key_list[i]] = value
			except Exception as e:
				print("PARSING ERROR in BALANCE SHEET")
				print(e)
			index_cnt = index_cnt + 1

	print("balance_sheet \t",balance_sheet_sub_list)
	return balance_sheet_sub_list


# Get information of cashflows statements
def scrape_cashflows(cashflow_table, year, unit):

	error_cashflows_list = []
	re_cashflow_list = []

	# Regular expression
	re_op_cashflow			= re.compile("((영업활동)|(영업활동으로[ \s]*인한)|(영업활동으로부터의))[ \s]*([순]*현금[ \s]*흐름)")
	re_op_cashflow_sub1 	= re.compile("((영업에서)|(영업으로부터))[ \s]*창출된[ \s]*현금(흐름)*")
	re_op_cashflow_sub2 	= re.compile("(연[ \s]*결[ \s]*)*당[ \s]*기[ \s]*순[ \s]*((이[ \s]*익)|(손[ \s]*익))")
	re_op_cashflow_sub3 	= re.compile("감[ \s]*가[ \s]*상[ \s]*각[ \s]*비")
	re_op_cashflow_sub4 	= re.compile("신[ \s]*탁[ \s]*계[ \s]*정[ \s]*대")
	
	re_invest_cashflow		= re.compile("투자[ \s]*활동[ \s]*현금[ \s]*흐름|투[ \s]*자[ \s]*활[ \s]*동[ \s]*으[ \s]*로[ \s]*인[ \s]*한[ \s]*[순]*현[ \s]*금[ \s]*흐[ \s]*름")
	re_invest_cashflow_sub1 = re.compile("유[ \s]*형[ \s]*자[ \s]*산[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub2 = re.compile("무[ \s]*형[ \s]*자[ \s]*산[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub3 = re.compile("토[ \s]*지[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub4 = re.compile("건[ \s]*물[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub5 = re.compile("구[ \s]*축[ \s]*물[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub6 = re.compile("기[ \s]*계[ \s]*장[ \s]*치[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub7 = re.compile("건[ \s]*설[ \s]*중[ \s]*인[ \s]*자[ \s]*산[ \s]*의[ \s]*((증[ \s]*가)|(취[ \s]*득))")
	re_invest_cashflow_sub8 = re.compile("차[ \s]*량[ \s]*운[ \s]*반[ \s]*구[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub9 = re.compile("비[ \s]*품[ \s]*의[ \s]*취[ \s]*득|비[ \s]*품[ \s]*의[ \s]*((증[ \s]*가)|(취[ \s]*득))")
	re_invest_cashflow_sub10= re.compile("공[ \s]*구[ \s]*기[ \s]*구[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub11= re.compile("시[ \s]*험[ \s]*연[ \s]*구[ \s]*설[ \s]*비[ \s]*의[ \s]*취[ \s]*득")
	re_invest_cashflow_sub12= re.compile("렌[ \s]*탈[ \s]*자[ \s]*산[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub13= re.compile("영[ \s]*업[ \s]*권[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub14= re.compile("산[ \s]*업[ \s]*재[ \s]*산[ \s]*권[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub15= re.compile("소[ \s]*프[ \s]*트[ \s]*웨[ \s]*어[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub16= re.compile("기[ \s]*타[ \s]*무[ \s]*형[ \s]*자[ \s]*산[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub17= re.compile("투[ \s]*자[ \s]*부[ \s]*통[ \s]*산[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub18= re.compile("관[ \s]*계[ \s]*기[ \s]*업[ \s]*투[ \s]*자[ \s]*의[ \s]*취[ \s]*득|관계[ \s]*기업[ \s]*투자[ \s]*주식의[ \s]*취득|지분법[ \s]*적용[ \s]*투자[ \s]*주식의[ \s]*취득")
	
	re_fin_cashflow			= re.compile("재무[ \s]*활동[ \s]*현금[ \s]*흐름|재무활동으로[ \s]*인한[ \s]*현금흐름")
	re_fin_cashflow_sub1	= re.compile("단기차입금의[ \s]*순증가")
	re_fin_cashflow_sub2	= re.compile("배당금[ \s]*지급|현금배당금의[ \s]*지급|배당금의[ \s]*지급|현금배당|보통주[ ]*배당[ ]*지급")
	re_fin_cashflow_sub3	= re.compile("자기주식의[ \s]*취득")
	re_start_cash			= re.compile("기초[ ]*현금[ ]*및[ ]*현금성[ ]*자산|기초의[ \s]*현금[ ]*및[ ]*현금성[ ]*자산|기[ \s]*초[ \s]*의[ \s]*현[ \s]*금|기[ \s]*초[ \s]*현[ \s]*금")
	re_end_cash				= re.compile("기말[ ]*현금[ ]*및[ ]*현금성[ ]*자산|기말의[ \s]*현금[ ]*및[ ]*현금성[ ]*자산|기[ \s]*말[ \s]*의[ \s]*현[ \s]*금|기[ \s]*말[ \s]*현[ \s]*금")

	re_cashflow_list.append(re_op_cashflow)
	re_cashflow_list.append(re_op_cashflow_sub1) 	
	re_cashflow_list.append(re_op_cashflow_sub2) 	
	re_cashflow_list.append(re_op_cashflow_sub3) 	
	re_cashflow_list.append(re_op_cashflow_sub4) 	
	
	re_cashflow_list.append(re_invest_cashflow)		
	re_cashflow_list.append(re_invest_cashflow_sub1) 
	re_cashflow_list.append(re_invest_cashflow_sub2) 
	re_cashflow_list.append(re_invest_cashflow_sub3) 
	re_cashflow_list.append(re_invest_cashflow_sub4) 
	re_cashflow_list.append(re_invest_cashflow_sub5) 
	re_cashflow_list.append(re_invest_cashflow_sub6) 
	re_cashflow_list.append(re_invest_cashflow_sub7) 
	re_cashflow_list.append(re_invest_cashflow_sub8) 
	re_cashflow_list.append(re_invest_cashflow_sub9) 
	re_cashflow_list.append(re_invest_cashflow_sub10)
	re_cashflow_list.append(re_invest_cashflow_sub11)
	re_cashflow_list.append(re_invest_cashflow_sub12)
	re_cashflow_list.append(re_invest_cashflow_sub13)
	re_cashflow_list.append(re_invest_cashflow_sub14)
	re_cashflow_list.append(re_invest_cashflow_sub15)
	re_cashflow_list.append(re_invest_cashflow_sub16)
	re_cashflow_list.append(re_invest_cashflow_sub17)
	re_cashflow_list.append(re_invest_cashflow_sub18)
	
	re_cashflow_list.append(re_fin_cashflow)		
	re_cashflow_list.append(re_fin_cashflow_sub1)	
	re_cashflow_list.append(re_fin_cashflow_sub2)	
	re_cashflow_list.append(re_fin_cashflow_sub3)	
	re_cashflow_list.append(re_start_cash)
	re_cashflow_list.append(re_end_cash)


	# 영업현금흐름
	## 영업에서 창출된 현금흐름
	## 당기순이익
	## 신탁계정대
	# 투자현금흐름
	## 유형자산의 취득
	## 무형자산의 취득
	## 토지의 취득
	## 건물의 취득
	## 구축물의 취득
	## 기계장치의 취득
	## 건설중인자산의증가
	## 차량운반구의 취득
	## 영업권의 취득
	## 산업재산권의 취득
	## 기타의무형자산의취득
	## 투자부동산의 취득
	## 관계기업투자의취득
	# 재무현금흐름
	## 단기차입금의 순증가
	## 배당금 지급
	## 자기주식의 취득
	# 기초 현금 및 현금성자산
	# 기말 현금 및 현금성자산

	cashflow_sub_list = {}
	
	cashflow_sub_list['year']					= year
	cashflow_sub_list["op_cashflow"]			= 0.0
	cashflow_sub_list["op_cashflow_sub1"]		= 0.0
	cashflow_sub_list["op_cashflow_sub2"]		= 0.0
	cashflow_sub_list["op_cashflow_sub3"]		= 0.0
	cashflow_sub_list["op_cashflow_sub4"]		= 0.0
	cashflow_sub_list["invest_cashflow"]		= 0.0
	cashflow_sub_list["invest_cashflow_sub1"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub2"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub3"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub4"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub5"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub6"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub7"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub8"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub9"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub10"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub11"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub12"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub13"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub14"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub15"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub16"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub17"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub18"]	= 0.0
	cashflow_sub_list["fin_cashflow"]			= 0.0
	cashflow_sub_list["fin_cashflow_sub1"]		= 0.0
	cashflow_sub_list["fin_cashflow_sub2"]		= 0.0
	cashflow_sub_list["fin_cashflow_sub3"]		= 0.0
	cashflow_sub_list["start_cash"]				= 0.0
	cashflow_sub_list["end_cash"]				= 0.0

	cashflow_key_list = []

	cashflow_key_list.append("op_cashflow")
	cashflow_key_list.append("op_cashflow_sub1")
	cashflow_key_list.append("op_cashflow_sub2")
	cashflow_key_list.append("op_cashflow_sub3")
	cashflow_key_list.append("op_cashflow_sub4")
	cashflow_key_list.append("invest_cashflow")
	cashflow_key_list.append("invest_cashflow_sub1")
	cashflow_key_list.append("invest_cashflow_sub2")
	cashflow_key_list.append("invest_cashflow_sub3")
	cashflow_key_list.append("invest_cashflow_sub4")
	cashflow_key_list.append("invest_cashflow_sub5")
	cashflow_key_list.append("invest_cashflow_sub6")
	cashflow_key_list.append("invest_cashflow_sub7")
	cashflow_key_list.append("invest_cashflow_sub8")
	cashflow_key_list.append("invest_cashflow_sub9")
	cashflow_key_list.append("invest_cashflow_sub10")
	cashflow_key_list.append("invest_cashflow_sub11")
	cashflow_key_list.append("invest_cashflow_sub12")
	cashflow_key_list.append("invest_cashflow_sub13")
	cashflow_key_list.append("invest_cashflow_sub14")
	cashflow_key_list.append("invest_cashflow_sub15")
	cashflow_key_list.append("invest_cashflow_sub16")
	cashflow_key_list.append("invest_cashflow_sub17")
	cashflow_key_list.append("invest_cashflow_sub18")
	cashflow_key_list.append("fin_cashflow")
	cashflow_key_list.append("fin_cashflow_sub1")
	cashflow_key_list.append("fin_cashflow_sub2")
	cashflow_key_list.append("fin_cashflow_sub3")
	cashflow_key_list.append("start_cash")
	cashflow_key_list.append("end_cash")

	#net_income = 0.0
	#print("len(trs)", len(trs))
	
	trs = cashflow_table.findAll("tr")
			
	# CASHFLOW statement
	if (len(trs) != 2):
		for tr in trs:
			#print("trs", len(trs))
			tds = tr.findAll("td")
			#print("tds", len(tds))
			try:
				if (len(tds) != 0):
					#print(tds[0].text.strip())

					value = 0.0
					for i in range(len(re_cashflow_list)):
						if re_cashflow_list[i].search(tds[0].text.strip()):
							if len(tds)>4:
								if (tds[1].text.strip() != '') and (tds[1].text.strip() != '-'):
									value = find_value(tds[1].text.strip(), unit)
									break # for i in len(re_cashflow_list)
								elif (tds[2].text.strip() != '') and (tds[2].text.strip() != '-'):
									value = find_value(tds[2].text.strip(), unit)
									break # for i in len(re_cashflow_list)
							else:
								if (tds[1].text.strip() != '') and (tds[1].text.strip() != '-'):
									value = find_value(tds[1].text.strip(), unit)
									break # for i in len(re_cashflow_list)
					if value != 0.0 and cashflow_sub_list[cashflow_key_list[i]] == 0.0:
						cashflow_sub_list[cashflow_key_list[i]] = value
					# No matching case
					else:
						error_cashflows_list.append(tds[0].text.strip())
			except Exception as e:
				print("NET INCOME PARSING ERROR in Cashflows")
				cashflow_sub_list["op_cashflow_sub1"] = "PARSING ERROR"
				print(e)
	# Special case
	## if (len(trs) != 2):
	else:	
		tr = trs[1]
		tds = tr.findAll("td")
		
		index_col = []
		prev = 0
		for a in tds[0].childGenerator():
			if (str(a) == "<br/>"):
				if (prev == 1):
					index_col.append('')	
				prev = 1
			else:
				prev = 0
				index_col.append(str(a).strip())	
		data_col = []
		prev = 0
		for b in tds[1].childGenerator():
			if (str(b) == "<br/>"):
				if (prev == 1):
					data_col.append('0')	
				prev = 1
			else:
				data_col.append(str(b))	
				prev = 0
		data_col2 = []
		prev = 0
		for b in tds[2].childGenerator():
			if (str(b) == "<br/>"):
				if (prev == 1):
					data_col2.append('')	
				prev = 1
			else:
				data_col2.append(str(b))	
				prev = 0

		#print(index_col)
		#print(data_col)
		print(len(index_col))
		print(len(data_col))
		index_cnt = 0

		for (index) in (index_col):
			try:
				value = 0.0
				for i in range(len(re_cashflow_list)):
					if re_cashflow_list[i].search(index):
						if len(tds)>4:
							if (data_col[index_cnt].strip() != '') and (data_col[index_cnt].strip() != '-'):
								value = find_value(data_col[index_cnt], unit)
								break
							elif (data_col2[index_cnt].strip() != '') and (data_col2[index_cnt].strip() != '-'):
								value = find_value(data_col2[index_cnt], unit)
								break
						else:
							if (data_col[index_cnt].strip() != '') and (data_col[index_cnt].strip() != '-'):
								value = find_value(data_col[index_cnt], unit)
								break
				if value != 0.0 and cashflow_sub_list[cashflow_key_list[i]] == 0.0:
					cashflow_sub_list[cashflow_key_list[i]] = value
			except Exception as e:
				print("PARSING ERROR")
				cashflow_sub_list["op_cashflow_sub1"] = "PARSING ERROR"
				print(e)
			index_cnt = index_cnt + 1

	print("cashflow_sheet \t",cashflow_sub_list)
	print("error_cashflow_sheet \t",error_cashflows_list)
	return cashflow_sub_list

# Get information of income statements
def scrape_income_statement(income_table, year, unit, mode):

	#매출액
	#매출원가
	#매출총이익
	#판매비와관리비
	#영업이익
	#기타수익
	#기타비용
	#금융수익
	#금융비용
	#법인세비용차감전순이익
	#번인세비용
	#당기순이익
	#기본주당이익

	re_income_list = []
	
	# Regular expression
	re_sales			=	re.compile("^매[ \s]*출[ \s]*액|\.[ \s]*매[ \s]*출[ \s]*액|\(매출액\)")
	re_sales_sub1		= 	re.compile("^매[ \s]*출[ \s]*원[ \s]*가|\.[ \s]*매[ \s]*출[ \s]*원[ \s]*가")
	re_sales_sub2		= 	re.compile("^매[ \s]*출[ \s]*총[ \s]*이[ \s]*익|\.[ \s]*매[ \s]*출[ \s]*총[ \s]*이[ \s]*익")
	re_sales_sub3		= 	re.compile("판[ \s]*매[ \s]*비[ \s]*와[ \s]*관[ \s]*리[ \s]*비")
	re_sales2			=	re.compile("^영[ \s]*업[ \s]*수[ \s]*익|\.[ \s]*영[ \s]*업[ \s]*수[ \s]*익")
	re_sales2_sub1		= 	re.compile("^영[ \s]*업[ \s]*비[ \s]*용|\.[ \s]*영[ \s]*업[ \s]*비[ \s]*용")
	re_op_income		= 	re.compile("^영[ \s]*업[ \s]*이[ \s]*익|\.[ \s]*영[ \s]*업[ \s]*이[ \s]*익")
	re_op_income_sub1	= 	re.compile("기[ \s]*타[ \s]*수[ \s]*익")
	re_op_income_sub2	= 	re.compile("기[ \s]*타[ \s]*비[ \s]*용")
	re_op_income_sub3	= 	re.compile("금[ \s]*융[ \s]*수[ \s]*익")
	re_op_income_sub4	= 	re.compile("금[ \s]*융[ \s]*비[ \s]*용")
	re_op_income_sub6	= 	re.compile("영[ \s]*업[ \s]*외[ \s]*수[ \s]*익")
	re_op_income_sub7	= 	re.compile("영[ \s]*업[ \s]*외[ \s]*비[ \s]*용")
	re_op_income_sub5	= 	re.compile("법[ \s]*인[ \s]*세[ \s]*비[ \s]*용[ \s]*차[ \s]*감[ \s]*전[ \s]*순[ \s]*((이[ \s]*익)|(손[ \s]*실))|법[ \s]*인[ \s]*세[ \s]*차[ \s]*감[ \s]*전[ \s]*계[ \s]*속[ \s]*영[ \s]*업[ \s]*순[ \s]*이[ \s]*익|법인세[ \s]*차감전[ \s]*순이익|법인세차감전계속영업이익|법인세비용차감전이익|법인세비용차감전계속영업[순]*이익|법인세비용차감전당기순이익|법인세비용차감전순이익|법인세비용차감전[ \s]*계속사업이익|법인세비용차감전순손익")
	re_tax				=	re.compile("법[ \s]*인[ \s]*세[ \s]*비[ \s]*용")
	re_net_income		=	re.compile("^순[ \s]*이[ \s]*익|^당[ \s]*기[ \s]*순[ \s]*이[ \s]*익|^연[ ]*결[ ]*[총 ]*당[ ]*기[ ]*순[ ]*이[ ]*익|지배기업의 소유주에게 귀속되는 당기순이익|분기순이익|당\(분\)기순이익|\.[ \s]*당[ \s]*기[ \s]*순[ \s]*이[ \s]*익|당분기연결순이익")
	re_eps				=	re.compile("기[ \s]*본[ \s]*주[ \s]*당[ \s]*((수[ \s]*익)|([순 \s]*이[ \s]*익))")

	re_income_list.append(re_sales)	
	re_income_list.append(re_sales_sub1)		 	
	re_income_list.append(re_sales_sub2)		 	
	re_income_list.append(re_sales_sub3)		 	
	re_income_list.append(re_sales2)	
	re_income_list.append(re_sales2_sub1)		 	
	re_income_list.append(re_op_income)		 	
	re_income_list.append(re_op_income_sub1)	 	
	re_income_list.append(re_op_income_sub2)	 	
	re_income_list.append(re_op_income_sub3)	 	
	re_income_list.append(re_op_income_sub4)	 	
	re_income_list.append(re_op_income_sub5)	 	
	re_income_list.append(re_op_income_sub6)	 	
	re_income_list.append(re_op_income_sub7)	 	
	re_income_list.append(re_tax)
	re_income_list.append(re_net_income)
	re_income_list.append(re_eps)				

	income_statement_sub_list = {}
	income_statement_sub_list["sales"]				=	0.0
	income_statement_sub_list["sales_sub1"]			=	0.0
	income_statement_sub_list["sales_sub2"]			=	0.0
	income_statement_sub_list["sales_sub3"]			=	0.0
	income_statement_sub_list["sales2"]				=	0.0
	income_statement_sub_list["sales2_sub1"]		=	0.0
	income_statement_sub_list["op_income"]		 	=	0.0
	income_statement_sub_list["op_income_sub1"]		=	0.0
	income_statement_sub_list["op_income_sub2"]		=	0.0
	income_statement_sub_list["op_income_sub3"]		=	0.0
	income_statement_sub_list["op_income_sub4"]		=	0.0
	income_statement_sub_list["op_income_sub5"]		=	0.0
	income_statement_sub_list["op_income_sub6"]		=	0.0
	income_statement_sub_list["op_income_sub7"]		=	0.0
	income_statement_sub_list["tax"]				=	0.0
	income_statement_sub_list["net_income"]			=	0.0
	income_statement_sub_list["eps"]				=	0.0
	income_statement_sub_list['year']				=	year

	income_statement_key_list = []
	income_statement_key_list.append("sales")			
	income_statement_key_list.append("sales_sub1")		
	income_statement_key_list.append("sales_sub2")		
	income_statement_key_list.append("sales_sub3")		
	income_statement_key_list.append("sales2")			
	income_statement_key_list.append("sales2_sub1")		
	income_statement_key_list.append("op_income")		
	income_statement_key_list.append("op_income_sub1")	
	income_statement_key_list.append("op_income_sub2")	
	income_statement_key_list.append("op_income_sub3")	
	income_statement_key_list.append("op_income_sub4")	
	income_statement_key_list.append("op_income_sub5")	
	income_statement_key_list.append("op_income_sub6")	
	income_statement_key_list.append("op_income_sub7")	
	income_statement_key_list.append("tax")			
	income_statement_key_list.append("net_income")		
	income_statement_key_list.append("eps")			

	trs = income_table.findAll("tr")

	# Income statement

	if (len(trs) != 2):
		for income_tr in trs:
			tds = income_tr.findAll("td")
			try:
				if (len(tds) != 0):
					#print(tds[0].text.strip())
					value = 0.0
					for i in range(len(re_income_list)):
						if re_income_list[i].search(tds[0].text.strip()):
							if mode == 0:
								if len(tds)>4:
									if (tds[1].text.strip() != '') and (tds[1].text.strip() != '-'):
										value = find_value(tds[1].text.strip(), unit)
										break # for i in len(re_income_list)
									elif (tds[2].text.strip() != '') and (tds[2].text.strip() != '-'):
										value = find_value(tds[2].text.strip(), unit)
										break # for i in len(re_income_list)
								else:
									if (tds[1].text.strip() != '') and (tds[1].text.strip() != '-'):
										value = find_value(tds[1].text.strip(), unit)
										break # for i in len(re_income_list)
							# mode 1
							else:
								if len(tds)>4:
									if (tds[3].text.strip() != '') and (tds[3].text.strip() != '-'):
										value = find_value(tds[2].text.strip(), unit)
										break # for i in len(re_income_list)
								else:
									if (tds[2].text.strip() != '') and (tds[2].text.strip() != '-'):
										value = find_value(tds[1].text.strip(), unit)
										break # for i in len(re_income_list)
					if value != 0.0 and income_statement_sub_list[income_statement_key_list[i]] == 0.0:
						income_statement_sub_list[income_statement_key_list[i]] = value
			except Exception as e:
				print("NET INCOME PARSING ERROR in Income statement")
				print(e)
				net_income = 0.0
	## if (len(trs) != 2):
	else:	
		income_tr = trs[1]
		tds = income_tr.findAll("td")
		
		index_col = []
		prev = 0
		for a in tds[0].childGenerator():
			if (str(a) == "<br/>"):
				if (prev == 1):
					index_col.append('')	
				prev = 1
			else:
				prev = 0
				index_col.append(str(a).strip())	
		data_col = []
		prev = 0
		for b in tds[1].childGenerator():
			if (str(b) == "<br/>"):
				if (prev == 1):
					data_col.append('0')	
				prev = 1
			else:
				data_col.append(str(b))	
				prev = 0
		data_col2 = []
		prev = 0
		for b in tds[2].childGenerator():
			if (str(b) == "<br/>"):
				if (prev == 1):
					data_col2.append('')	
				prev = 1
			else:
				data_col2.append(str(b))	
				prev = 0

		
		print(len(index_col))
		print(len(data_col))
		index_cnt = 0

		for (index) in (index_col):
			try:
				value = 0.0
				for i in range(len(re_income_list)):
					if re_income_list[i].search(index):
						if len(tds)>4:
							if (data_col[index_cnt].strip() != '') and (data_col[index_cnt].strip() != '-'):
								value = find_value(data_col[index_cnt], unit)
								break
							elif (data_col2[index_cnt].strip() != '') and (data_col2[index_cnt].strip() != '-'):
								value = find_value(data_col2[index_cnt], unit)
								break
						else:
							if (data_col[index_cnt].strip() != '') and (data_col[index_cnt].strip() != '-'):
								value = find_value(data_col[index_cnt], unit)
								break
				if value != 0.0 and income_statement_sub_list[income_statement_key_list[i]] == 0.0:
					income_statement_sub_list[income_statement_key_list[i]] = value
			except Exception as e:
				print("PARSING ERROR in INCOME STATEMENT")
				print(e)
			index_cnt = index_cnt + 1

	print("income_sheet \t",income_statement_sub_list)
	return income_statement_sub_list

# Main function
def main():

	# Default
	corp = "민앤지"
	term = {"사업":"%EC%82%AC%EC%97%85%EB%B3%B4%EA%B3%A0%EC%84%9C",
			"분기":"%EB%B6%84%EA%B8%B0%EB%B3%B4%EA%B3%A0%EC%84%9C",
			"반기":"%EB%B0%98%EA%B8%B0%EB%B3%B4%EA%B3%A0%EC%84%9C",
			"복합":""}
	period = '복합'
	# 사업보고서 "%EB %B3%B4 %EA%B3 %A0%EC %84%9C"
	report = "%EC%82%AC%EC%97%85%EB%B3%B4%EA%B3%A0%EC%84%9C"
	# 분기보고서
	report2 = "%EB%B6%84%EA%B8%B0%EB%B3%B4%EA%B3%A0%EC%84%9C"
	# 반기보고서
	report3 = "%EB%B0%98%EA%B8%B0%EB%B3%B4%EA%B3%A0%EC%84%9C"
	# print(report2.decode())
	report_ = term["{}".format(period)]


	workbook_name = "{}_Dart_financial_statement_{}.xlsx".format(corp,period)

	re_income_find = re.compile("법[ \s]*인[ \s]*세[ \s]*비[ \s]*용(\(이익\))*[ \s]*차[ \s]*감[ \s]*전[ \s]*순[ \s]*((이[ \s]*익)|(손[ \s]*실))|법[ \s]*인[ \s]*세[ \s]*차[ \s]*감[ \s]*전[ \s]*계[ \s]*속[ \s]*영[ \s]*업[ \s]*순[ \s]*이[ \s]*익|법인세[ \s]*차감전[ \s]*순이익|법인세차감전계속영업이익|법인세비용차감전이익|법인세비용차감전계속영업[순]*이익|법인세비용차감전당기순이익|법인세(비용차감|손익가감)전순이익|법인세비용차감전[ \s]*계속사업이익|법인세비용차감전순손익")
	re_cashflow_find = re.compile("영업활동[ \s]*현금[ \s]*흐름|영업활동으로[ \s]*인한[ \s]*[순]*현금[ \s]*흐름|영업활동으로부터의[ \s]*현금흐름|영업활동으로 인한 자산부채의 변동")
	re_balance_sheet_find = re.compile("현[ \s]*금[ \s]*및[ \s]*현[ \s]*금[ \s]*((성[ \s]*자[ \s]*산)|(등[ \s]*가[ \s]*물))")

	### PART I - Read pickle file for stock lists
	if os.path.isfile("tickers.pk"):
		tickers = pd.read_pickle("tickers.pk")
		print("load url pickle")
	else:
		tickers_code = STOCK.get_market_ticker_code_list()
		tickers_etf = e3.get_etf_ticker_list()
		tickers = tickers_code['종목'].append(tickers_etf)
		tickers = pd.DataFrame(tickers.values, index=tickers.index, columns=['종목'])
		tickers['시장'] = tickers_code['시장']
		tickers['시장'] = tickers['시장'].replace(np.nan, 'ETF')
		tickers.to_pickle("tickers.pk")
		tickers.to_csv("tickers.csv", encoding='ms949')

	corp_info = tickers[tickers['종목']==corp]
	if len(corp_info) != 0:
		stock_code = corp_info.index[0]
		stock_cat = corp_info['시장'].values[0]
	else:
		print("STOCK CODE ERROR")
		raise AssertionError("STOCK CODE ERROR")
	# URL
	#url_templete = "http://dart.fss.or.kr/dsab002/search.ax?reportName=%s&&maxResults=100&&textCrpNm=%s"
	url_templete = "http://dart.fss.or.kr/dsab002/search.ax?reportName=%s&&maxResults=100&&textCrpNm=%s&&startDate=%s&&endDate=%s"
	headers = {'Cookie':'DSAB002_MAXRESULTS=5000;'}
	
	dart_post_list = []
	cashflow_list = []
	balance_sheet_list = []
	income_statement_list = []

	# 최근

	year = 2010
	start_day = datetime(2000,1,1)
	end_day = datetime(2019,5,28)
	delta = end_day - start_day



	# start_day2 = datetime(2005,1,1)
	# end_day2 = datetime(2019,5,28)
	#
	#
	# # 최신 분기보고서 읽기
	# handle = urllib.request.urlopen(url_templete % (term['분기'], urllib.parse.quote(corp), start_day2.strftime('%Y%m%d'), end_day2.strftime('%Y%m%d')))
	#
	# data = handle.read()
	# soup = BeautifulSoup(data, 'html.parser', from_encoding='utf-8')
	#
	# table = soup.find('table')
	# trs = table.findAll('tr')
	# tds = table.findAll('td')
	# counts = len(tds)
	#
	# if counts > 2:
	# 	# Delay operation
	# 	#time.sleep(20)
	#
	# 	link_list = []
	# 	date_list = []
	# 	corp_list = []
	# 	market_list = []
	# 	title_list = []
	# 	reporter_list = []
	#
	# 	# recent report 최신 분기보고서만 읽기
	# 	tr = trs[1]
	# 	# time.sleep(2)
	# 	tds = tr.findAll('td')
	# 	link = 'http://dart.fss.or.kr' + tds[2].a['href']
	# 	date = tds[4].text.strip().replace('.', '-')
	# 	corp_name = tds[1].text.strip()
	# 	market = tds[1].img['title']
	# 	title = " ".join(tds[2].text.split())
	# 	reporter = tds[3].text.strip()
	#
	# 	link_list.append(link)
	# 	date_list.append(date)
	# 	corp_list.append(corp_name)
	# 	market_list.append(market)
	# 	title_list.append(title)
	# 	reporter_list.append(reporter)
	#
	# 	dart_post_sublist = []
	#
	# 	year = title[-8:-1] ##첫번째 분기값 year 결정하는 곳
	# 	print(year)
	# 	print(corp_name)
	# 	print(title)
	# 	print(date)
	# 	handle = urllib.request.urlopen(link)
	# 	data = handle.read()
	# 	soup2 = BeautifulSoup(data, 'html.parser', from_encoding='utf-8')
	# 	print(link)
	#
	# 	head_lines = soup2.find('head').text.split("\n")
	# 	#print(head_lines)
	# 	re_tree_find1 = re.compile("2.[ ]*연결재무제표")
	# 	re_tree_find1_bak = re.compile("4.[ ]*재무제표")
	# 	line_num = 0
	# 	line_find = 0
	# 	for head_line in head_lines:
	# 		#print(head_line)
	# 		if (re_tree_find1.search(head_line)):
	# 			line_find = line_num
	# 			break
	# 		line_num = line_num + 1
	#
	# 	line_num = 0
	# 	line_find_bak = 0
	# 	for head_line in head_lines:
	# 		if (re_tree_find1_bak.search(head_line)):
	# 			line_find_bak = line_num
	# 			break
	# 		line_num = line_num + 1
	# 	if(line_find != 0):
	# 		line_words = head_lines[line_find+4].split("'")
	# 		#print(line_words)
	# 		rcpNo = line_words[1]
	# 		dcmNo = line_words[3]
	# 		eleId = line_words[5]
	# 		offset = line_words[7]
	# 		length = line_words[9]
	#
	# 		dart = soup2.find_all(string=re.compile('dart.dtd'))
	# 		dart2 = soup2.find_all(string=re.compile('dart2.dtd'))
	# 		dart3 = soup2.find_all(string=re.compile('dart3.xsd'))
	#
	# 		if len(dart3) != 0:
	# 			link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=" + eleId + "&offset=" + offset + "&length=" + length + "&dtd=dart3.xsd"
	# 		elif len(dart2) != 0:
	# 			link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=" + eleId + "&offset=" + offset + "&length=" + length + "&dtd=dart2.dtd"
	# 		elif len(dart) != 0:
	# 			link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=" + eleId + "&offset=" + offset + "&length=" + length + "&dtd=dart.dtd"
	# 		else:
	# 			link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=0&offset=0&length=0&dtd=HTML"
	#
	# 		print("link2 : {}".format(link2))
	#
	# 		#try:
	# 		handle = urllib.request.urlopen(link2)
	# 		print(handle) ## <http.client.HTTPResponse object at 0x0D86EC90>
	# 		data = handle.read()
	# 		soup3 = BeautifulSoup(data, 'html.parser', from_encoding='utf-8')
	#
	# 		tables = soup3.findAll("table")
	#
	# 		# 2. 연결재무제표가 비어 있는 경우
	# 		if (len(tables) == 0):
	# 			line_words = head_lines[line_find_bak+4].split("'")
	# 			#print(line_words)
	# 			rcpNo = line_words[1]
	# 			dcmNo = line_words[3]
	# 			eleId = line_words[5]
	# 			offset = line_words[7]
	# 			length = line_words[9]
	#
	# 			dart = soup2.find_all(string=re.compile('dart.dtd'))
	# 			dart2 = soup2.find_all(string=re.compile('dart2.dtd'))
	# 			dart3 = soup2.find_all(string=re.compile('dart3.xsd'))
	#
	# 			if len(dart3) != 0:
	# 				link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=" + eleId + "&offset=" + offset + "&length=" + length + "&dtd=dart3.xsd"
	# 			elif len(dart2) != 0:
	# 				link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=" + eleId + "&offset=" + offset + "&length=" + length + "&dtd=dart2.dtd"
	# 			elif len(dart) != 0:
	# 				link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=" + eleId + "&offset=" + offset + "&length=" + length + "&dtd=dart.dtd"
	# 			else:
	# 				link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=0&offset=0&length=0&dtd=HTML"
	#
	# 			print("link2 : {}".format(link2))
	#
	# 			handle = urllib.request.urlopen(link2)
	# 			print(handle) ## <http.client.HTTPResponse object at 0x0D86EC90>
	# 			data = handle.read()
	# 			soup3 = BeautifulSoup(data, 'html.parser', from_encoding='utf-8')
	# 			tables = soup3.findAll("table")
	#
	# 		cnt = 0
	# 		table_num = 0
	#
	# 		for table in tables:
	# 			if (re_cashflow_find.search(table.text)):
	# 				table_num = cnt
	# 				break
	# 			cnt = cnt + 1
	#
	# 		print("table_num", table_num, "Tables", len(tables)) ## table_num 7 Tables 8
	# 		cashflow_table = soup3.findAll("table")[table_num]
	#
	# 		cnt = 0
	# 		table_income_num = 0
	# 		for table in tables:
	# 			if (re_income_find.search(table.text)):
	# 				table_income_num = cnt
	# 				break
	# 			cnt = cnt + 1
	# 		income_table = soup3.findAll("table")[table_income_num]
	# 		#print("table_income_num", table_income_num, "Tables", len(tables))
	#
	# 		cnt = 0
	# 		table_balance_num = 0
	# 		for table in tables:
	# 			if (re_balance_sheet_find.search(table.text)):
	# 				table_balance_num = cnt
	# 				break
	# 			cnt = cnt + 1
	# 		balance_table = soup3.findAll("table")[table_balance_num]
	# 		print("table_balance_num", table_balance_num, "Tables", len(tables)) ## table_balance_num 1 Tables 8
	#
	# 		unit = 100.0
	# 		unit_find = 0
	# 		re_unit1 = re.compile('단위[ \s]*:[ \s]*원')
	# 		re_unit2 = re.compile('단위[ \s]*:[ \s]*백만원')
	# 		re_unit3 = re.compile('단위[ \s]*:[ \s]*천원')
	#
	# 		# 원
	# 		if len(soup3.findAll("table")[table_num-1](string=re_unit1)) != 0:
	# 			unit = 100000000.0
	# 			unit_find = 1
	# 			#print("Unit ###1")
	# 		# 백만원
	# 		elif len(soup3.findAll("table")[table_num-1](string=re_unit2)) != 0:
	# 			unit = 100.0
	# 			unit_find = 1
	# 			#print("Unit ###2")
	# 		elif len(soup3.findAll("table")[table_num-1](string=re_unit3)) != 0:
	# 			unit = 100000.0
	# 			unit_find = 1
	# 			#print("Unit ###3")
	#
	# 		if unit_find == 0:
	# 			print ("UNIT NOT FOUND")
	# 			if len(soup3.findAll(string=re_unit1)) != 0:
	# 				print("Unit ###1")
	# 				unit = 100000000.0
	# 			elif len(soup3.findAll(string=re_unit2)) != 0:
	# 				print("Unit ###2")
	# 				unit = 100.0
	# 			elif len(soup3.findAll(string=re_unit3)) != 0:
	# 				print("Unit ###3")
	# 				unit = 100000.0
	#
	# 		cashflow_sub_list = scrape_cashflows(cashflow_table, year, unit)
	# 		income_statement_sub_list = scrape_income_statement(income_table, year, unit, 1)
	# 		balance_sheet_sub_list = scrape_balance_sheet(balance_table, year, unit)
	#
	# 		cashflow_sub_list['net_income'] = income_statement_sub_list['net_income']
	#
	# 	## if(line_find != 0):
	# 	else: ## Error
	# 		print("FINDING LINE NUMBER ERROR")
	# 		cashflow_sub_list = {}
	#
	# 		cashflow_sub_list['year']				= 2010
	# 		cashflow_sub_list['op_cashflow']		= 0.0
	# 		cashflow_sub_list['op_cashflow_sub1']	= "FINDING LINE NUMBER ERROR"
	# 		cashflow_sub_list['op_cashflow_sub2']	= 0.0
	#
	# 		cashflow_sub_list['invest_cashflow']		= 0.0
	# 		cashflow_sub_list['invest_cashflow_sub1']	= 0.0
	# 		cashflow_sub_list['invest_cashflow_sub2'] 	= 0.0
	# 		cashflow_sub_list['invest_cashflow_sub3'] 	= 0.0
	# 		cashflow_sub_list['invest_cashflow_sub4'] 	= 0.0
	# 		cashflow_sub_list['invest_cashflow_sub5'] 	= 0.0
	# 		cashflow_sub_list['invest_cashflow_sub6'] 	= 0.0
	# 		cashflow_sub_list['invest_cashflow_sub7'] 	= 0.0
	# 		cashflow_sub_list['invest_cashflow_sub8'] 	= 0.0
	# 		cashflow_sub_list['invest_cashflow_sub9'] 	= 0.0
	# 		cashflow_sub_list['invest_cashflow_sub10']	= 0.0
	# 		cashflow_sub_list['invest_cashflow_sub11'] 	= 0.0
	# 		cashflow_sub_list['invest_cashflow_sub12'] 	= 0.0
	# 		cashflow_sub_list['invest_cashflow_sub13'] 	= 0.0
	# 		cashflow_sub_list['invest_cashflow_sub14'] 	= 0.0
	# 		cashflow_sub_list['invest_cashflow_sub15'] 	= 0.0
	# 		cashflow_sub_list['invest_cashflow_sub16'] 	= 0.0
	# 		cashflow_sub_list['invest_cashflow_sub17'] 	= 0.0
	# 		cashflow_sub_list['invest_cashflow_sub18'] 	= 0.0
	#
	# 		cashflow_sub_list['fin_cashflow']		= 0.0
	# 		cashflow_sub_list['fin_cashflow_sub1']	= 0.0
	# 		cashflow_sub_list['fin_cashflow_sub2'] 	= 0.0
	# 		cashflow_sub_list['fin_cashflow_sub3'] 	= 0.0
	#
	# 		cashflow_sub_list['start_cash']		= 0.0
	# 		cashflow_sub_list['end_cash']		= 0.0
	# 		cashflow_sub_list['net_income']		= 0.0
	#
	# 		#print(cashflow_sub_list)
	#
	# 		balance_sheet_sub_list = {}
	# 		balance_sheet_sub_list['year']						=	2010
	# 		balance_sheet_sub_list["asset_current"]				=	0.0
	# 		balance_sheet_sub_list["asset_current_sub1"]		=	"FINDING LINE NUMBER ERROR"
	# 		balance_sheet_sub_list["asset_current_sub2"]		=	0.0
	# 		balance_sheet_sub_list["asset_current_sub3"]		=	0.0
	# 		balance_sheet_sub_list["asset_non_current"]			=	0.0
	# 		balance_sheet_sub_list["asset_non_current_sub1"]	=	0.0
	# 		balance_sheet_sub_list["asset_non_current_sub2"]	=	0.0
	# 		balance_sheet_sub_list["asset_sum"]					=	0.0
	# 		balance_sheet_sub_list["liability_current"]				=	0.0
	# 		balance_sheet_sub_list["liability_current_sub1"]		=	0.0
	# 		balance_sheet_sub_list["liability_current_sub2"]		=	0.0
	# 		balance_sheet_sub_list["liability_current_sub3"]		=	0.0
	# 		balance_sheet_sub_list["liability_non_current"]			=	0.0
	# 		balance_sheet_sub_list["liability_non_current_sub1"]	=	0.0
	# 		balance_sheet_sub_list["liability_non_current_sub2"]	=	0.0
	# 		balance_sheet_sub_list["liability_non_current_sub3"]	=	0.0
	# 		balance_sheet_sub_list["liability_non_current_sub4"]	=	0.0
	# 		balance_sheet_sub_list["liability_sum"]					=	0.0
	# 		balance_sheet_sub_list["equity"]						=	0.0
	# 		balance_sheet_sub_list["equity_sub1"]					=	0.0
	# 		balance_sheet_sub_list["equity_sub3"]					=	0.0
	# 		balance_sheet_sub_list["equity_sub2"]					=	0.0
	# 		balance_sheet_sub_list["equity_sum"]					=	0.0
	#
	# 		income_statement_sub_list = {}
	# 		income_statement_sub_list['year']				=	2010
	# 		income_statement_sub_list["sales"]				=	0.0
	# 		income_statement_sub_list["sales_sub1"]			=	"FINDING LINE NUMBER ERROR"
	# 		income_statement_sub_list["sales_sub2"]			=	0.0
	# 		income_statement_sub_list["sales_sub3"]			=	0.0
	# 		income_statement_sub_list["sales2"]				=	0.0
	# 		income_statement_sub_list["sales2_sub1"]		=	0.0
	# 		income_statement_sub_list["op_income"]		 	=	0.0
	# 		income_statement_sub_list["op_income_sub1"]		=	0.0
	# 		income_statement_sub_list["op_income_sub2"]		=	0.0
	# 		income_statement_sub_list["op_income_sub3"]		=	0.0
	# 		income_statement_sub_list["op_income_sub4"]		=	0.0
	# 		income_statement_sub_list["op_income_sub5"]		=	0.0
	# 		income_statement_sub_list["op_income_sub6"]		=	0.0
	# 		income_statement_sub_list["op_income_sub7"]		=	0.0
	# 		income_statement_sub_list["tax"]				=	0.0
	# 		income_statement_sub_list["net_income"]			=	0.0
	# 		income_statement_sub_list["eps"]				=	0.0
	#
	# 	dart_post_sublist.append(date)
	# 	dart_post_sublist.append(corp_name)
	# 	dart_post_sublist.append(market)
	# 	dart_post_sublist.append(title)
	# 	dart_post_sublist.append(link)
	#
	# 	dart_post_list.append(dart_post_sublist)
	# 	cashflow_list.append(cashflow_sub_list)
	# 	balance_sheet_list.append(balance_sheet_sub_list)
	# 	income_statement_list.append(income_statement_sub_list)
	#
	# #handle = urllib.request.urlopen(url_templete % (report, urllib.parse.quote(corp)))
	# #print("URL" + url_templete % (report, corp))
	#
	#
	#

	#################### 여기서부터 시작 connect#################
	## 선택


	handle = urllib.request.urlopen(url_templete % (report, urllib.parse.quote(corp), start_day.strftime('%Y%m%d'), end_day.strftime('%Y%m%d')))
	# print("URL" + url_templete % (report, corp, start_day.strftime('%Y%m%d'), end_day.strftime('%Y%m%d')))


	data = handle.read()
	soup = BeautifulSoup(data, 'html.parser', from_encoding='utf-8')
	
	table = soup.find('table')
	trs = table.findAll('tr')
	tds = table.findAll('td')
	counts = len(tds)

	handle2 = urllib.request.urlopen(
		url_templete % (report2, urllib.parse.quote(corp), start_day.strftime('%Y%m%d'), end_day.strftime('%Y%m%d')))
	data2 = handle2.read()
	soup2 = BeautifulSoup(data2, 'html.parser', from_encoding='utf-8')

	table2 = soup2.find('table')
	trs2 = table2.findAll('tr')

	handle3 = urllib.request.urlopen(
		url_templete % (report3, urllib.parse.quote(corp), start_day.strftime('%Y%m%d'), end_day.strftime('%Y%m%d')))
	data3 = handle3.read()
	soup3 = BeautifulSoup(data3, 'html.parser', from_encoding='utf-8')

	table3 = soup3.find('table')
	trs3 = table3.findAll('tr')

	#print(counts)
	# datetime(int(date.split("-")[0]), int(date.split("-")[1]), int(date.split("-")[2]))
	#if counts > 0:
	if counts > 2:
		# Delay operation
		# time.sleep(20)

		link_list = []
		date_list = []
		corp_list = []
		market_list = []
		title_list = []
		reporter_list = []
		tr_cnt = 0
		title = ""

		aaa = trs[1:]+trs2[1:]+trs3[1:]
		cc = []

		for aa in aaa:
			# cc.append(int(aa.findAll('td')[4].text.replace('.','').strip()))
			cc.append(int(aa.findAll('td')[2].text.strip().split('.')[0][-4:]+aa.findAll('td')[2].text.strip().split('.')[1][:2]))
		redun = []
		for idx in range(len(cc)-1):
			if cc[idx] == cc[idx+1]:
				redun.append(idx+1)
		ord = np.argsort(cc).tolist()[::-1]


		## 3월부터 시작
		for ordr in ord[::-1]:
			if int(aaa[ordr].findAll('td')[2].text.strip().split('.')[1][:2]) != 3:
				ord.pop()
			else:
				break

		for tr_idx in ord:
			if tr_idx in redun:
				continue
			tr = aaa[tr_idx]
			tr_cnt = tr_cnt + 1
			# time.sleep(2)
			tds = tr.findAll('td')
			link = 'http://dart.fss.or.kr' + tds[2].a['href']
			date = tds[4].text.strip().replace('.', '-')
			corp_name = tds[1].text.strip()
			market = tds[1].img['title']
			title = " ".join(tds[2].text.split())
			print("current",title)
			reporter = tds[3].text.strip()

			re_pass = re.compile("해외증권거래소등에신고한사업보고서등의국내신고")
			if (not re_pass.search(title)):
				link_list.append(link)
				date_list.append(date)
				corp_list.append(corp_name)
				market_list.append(market)
				title_list.append(title)
				reporter_list.append(reporter)

				dart_post_sublist = []

				# year = int(date[0:4])
				year = title[-8:-1]
				print(year)
				print(corp_name)
				print(title)
				print(date)
				handle = urllib.request.urlopen(link)
				#print(link)
				data = handle.read()
				soup2 = BeautifulSoup(data, 'html.parser', from_encoding='utf-8')
				#print(soup2)
				
				#print(type(soup2.find('head').text))
				head_lines = soup2.find('head').text.split("\n")
				#print(head_words)

				# From 2015 ~ now
				#if (year>2014):
				#	re_tree_find = re.compile("2. 연결재무제표")
				## From 2010 to 2014
				#elif (year>2009):
				#	re_tree_find = re.compile("재무제표 등")
				## From 2008 to 2009
				#elif (year>2007):
				#	re_tree_find = re.compile("1. 연결재무제표에 관한 사항")
				## From 2002 to 2007
				#elif (year>2001):
				#	re_tree_find = re.compile("4. 재무제표")
				#else:
				#	re_tree_find = re.compile("3. 재무제표")

				re_tree_find1 = re.compile("2. 연결재무제표")
				re_tree_find2 = re.compile("재무제표 등")
				re_tree_find3 = re.compile("1. 연결재무제표에 관한 사항")
				re_tree_find4 = re.compile("4. 재무제표")
				re_tree_find5 = re.compile("3. 재무제표")
				
				re_tree_find1_bak = re.compile("4.[ ]*재무제표")
				
				line_num = 0
				line_find = 0
				for head_line in head_lines:
					if (re_tree_find1.search(head_line)):
						line_find = line_num
						break
						#print(head_line)
					elif (re_tree_find2.search(head_line)):
						line_find = line_num
						break
					elif (re_tree_find3.search(head_line)):
						line_find = line_num
						break
					elif (re_tree_find4.search(head_line)):
						line_find = line_num
						break
					elif (re_tree_find5.search(head_line)):
						line_find = line_num
						break
					line_num = line_num + 1

				line_num = 0
				line_find_bak = 0
				for head_line in head_lines:
					if (re_tree_find1_bak.search(head_line)):
						line_find_bak = line_num
						break
					line_num = line_num + 1


				if(line_find != 0):
		
					#print(head_lines[line_find])
					#print(head_lines[line_find+1])
					#print(head_lines[line_find+2])
					#print(head_lines[line_find+3])
					#print(head_lines[line_find+4])

					line_words = head_lines[line_find+4].split("'")
					#print(line_words)
					rcpNo = line_words[1]
					dcmNo = line_words[3]
					eleId = line_words[5]
					offset = line_words[7]
					length = line_words[9]

					#test = soup2.find('a', {'href' : '#download'})['onclick']
					#words = test.split("'")
					#rcpNo = words[1]
					#dcmNo = words[3]
					
					dart = soup2.find_all(string=re.compile('dart.dtd'))
					dart2 = soup2.find_all(string=re.compile('dart2.dtd'))
					dart3 = soup2.find_all(string=re.compile('dart3.xsd'))

					if len(dart3) != 0:
						link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=" + eleId + "&offset=" + offset + "&length=" + length + "&dtd=dart3.xsd"
					elif len(dart2) != 0:
						link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=" + eleId + "&offset=" + offset + "&length=" + length + "&dtd=dart2.dtd"
					elif len(dart) != 0:
						link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=" + eleId + "&offset=" + offset + "&length=" + length + "&dtd=dart.dtd"
					else:
						link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=0&offset=0&length=0&dtd=HTML"  
					
					print(link2)

					#try:
					handle = urllib.request.urlopen(link2)
					#print(handle)
					data = handle.read()
					soup3 = BeautifulSoup(data, 'html.parser', from_encoding='utf-8')
					#print(soup3)

					tables = soup3.findAll("table")
			
					# 2. 연결재무제표가 비어 있는 경우
					if (len(tables) == 0):
						line_words = head_lines[line_find_bak+4].split("'")
						#print(line_words)
						rcpNo = line_words[1]
						dcmNo = line_words[3]
						eleId = line_words[5]
						offset = line_words[7]
						length = line_words[9]

						dart = soup2.find_all(string=re.compile('dart.dtd'))
						dart2 = soup2.find_all(string=re.compile('dart2.dtd'))
						dart3 = soup2.find_all(string=re.compile('dart3.xsd'))

						if len(dart3) != 0:
							link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=" + eleId + "&offset=" + offset + "&length=" + length + "&dtd=dart3.xsd"
						elif len(dart2) != 0:
							link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=" + eleId + "&offset=" + offset + "&length=" + length + "&dtd=dart2.dtd"
						elif len(dart) != 0:
							link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=" + eleId + "&offset=" + offset + "&length=" + length + "&dtd=dart.dtd"
						else:
							link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=0&offset=0&length=0&dtd=HTML"  
						
						print(link2)
						
						handle = urllib.request.urlopen(link2)
						print(handle)
						data = handle.read()
						soup3 = BeautifulSoup(data, 'html.parser', from_encoding='utf-8')
						tables = soup3.findAll("table")


					cnt = 0
					table_num = 0

					for table in tables:
						if (re_cashflow_find.search(table.text)):
							table_num = cnt
							break
						cnt = cnt + 1
					
					print("table_num", table_num, "Tables", len(tables))
					cashflow_table = soup3.findAll("table")[table_num]
					
					trs = cashflow_table.findAll("tr")
					
					cnt = 0
					table_income_num = 0
					for table in tables:
						if (re_income_find.search(table.text)):
							table_income_num = cnt
							break
						cnt = cnt + 1
					income_table = soup3.findAll("table")[table_income_num]
					print("table_income_num", table_income_num, "Tables", len(tables))
					
					cnt = 0
					table_balance_num = 0
					for table in tables:
						if (re_balance_sheet_find.search(table.text)):
							table_balance_num = cnt
							break
						cnt = cnt + 1
					balance_table = soup3.findAll("table")[table_balance_num]
					print("table_balance_num", table_balance_num, "Tables", len(tables))
			
					unit = 100.0
					unit_find = 0
					re_unit1 = re.compile('단위[ \s]*:[ \s]*원')
					re_unit2 = re.compile('단위[ \s]*:[ \s]*백만원')
					re_unit3 = re.compile('단위[ \s]*:[ \s]*천원')

					# 원
					if len(soup3.findAll("table")[table_num-1](string=re_unit1)) != 0:
						unit = 100000000.0
						unit_find = 1
						#print("Unit ###1")
					# 백만원
					elif len(soup3.findAll("table")[table_num-1](string=re_unit2)) != 0:
						unit = 100.0
						unit_find = 1
						#print("Unit ###2")
					elif len(soup3.findAll("table")[table_num-1](string=re_unit3)) != 0:
						unit = 100000.0
						unit_find = 1
						#print("Unit ###3")

					if unit_find == 0:
						print ("UNIT NOT FOUND")
						if len(soup3.findAll(string=re_unit1)) != 0:
							print("Unit ###1")
							unit = 100000000.0
						elif len(soup3.findAll(string=re_unit2)) != 0:
							print("Unit ###2")
							unit = 100.0
						elif len(soup3.findAll(string=re_unit3)) != 0:
							print("Unit ###3")
							unit = 100000.0
			
					## 원
					#if len(soup3.findAll("table")[table_num-1](string=re.compile('단위[ ]*:[ ]*원'))) != 0:
					#	unit = 100000000.0
					## 백만원
					#elif len(soup3.findAll("table")[table_num-1](string=re.compile('단위[ ]*:[ ]*백만원'))) != 0:
					#	unit = 100.0
					#elif len(soup3.findAll("table")[table_num-1](string=re.compile('단위[ ]*:[ ]*천원'))) != 0:
					#	unit = 100000.0
				
					# Scrape data
					cashflow_sub_list = scrape_cashflows(cashflow_table, title[-8:-1], unit) ##title[-8:-1]->year-1
					income_statement_sub_list = scrape_income_statement(income_table, title[-8:-1], unit, 0)
					income_statement_sub_list['net_income'] = income_statement_sub_list['op_income_sub5'] - income_statement_sub_list['tax']
					balance_sheet_sub_list = scrape_balance_sheet(balance_table, title[-8:-1], unit)
					# print("cashflow_sheet \t",cashflow_sub_list)
					
					cashflow_sub_list['net_income'] = income_statement_sub_list['net_income']

				## if(line_find != 0):
				else:
					print("FINDING LINE NUMBER ERROR")
					cashflow_sub_list = {}
					
					cashflow_sub_list['year']				= title[-8:-1]
					cashflow_sub_list['op_cashflow']		= 0.0
					cashflow_sub_list['op_cashflow_sub1']	= "FINDING LINE NUMBER ERROR"
					cashflow_sub_list['op_cashflow_sub2']	= 0.0

					cashflow_sub_list['invest_cashflow']		= 0.0
					cashflow_sub_list['invest_cashflow_sub1']	= 0.0
					cashflow_sub_list['invest_cashflow_sub2'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub3'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub4'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub5'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub6'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub7'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub8'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub9'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub10']	= 0.0
					cashflow_sub_list['invest_cashflow_sub11'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub12'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub13'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub14'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub15'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub16'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub17'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub18'] 	= 0.0
					
					cashflow_sub_list['fin_cashflow']		= 0.0
					cashflow_sub_list['fin_cashflow_sub1']	= 0.0
					cashflow_sub_list['fin_cashflow_sub2'] 	= 0.0
					cashflow_sub_list['fin_cashflow_sub3'] 	= 0.0

					cashflow_sub_list['start_cash']		= 0.0
					cashflow_sub_list['end_cash']		= 0.0
					cashflow_sub_list['net_income']		= 0.0
			
					balance_sheet_sub_list = {}
					balance_sheet_sub_list['year']						=	title[-8:-1]
					balance_sheet_sub_list["asset_current"]				=	0.0
					balance_sheet_sub_list["asset_current_sub1"]		=	"FINDING LINE NUMBER ERROR"
					balance_sheet_sub_list["asset_current_sub2"]		=	0.0
					balance_sheet_sub_list["asset_current_sub3"]		=	0.0
					balance_sheet_sub_list["asset_non_current"]			=	0.0
					balance_sheet_sub_list["asset_non_current_sub1"]	=	0.0
					balance_sheet_sub_list["asset_non_current_sub2"]	=	0.0
					balance_sheet_sub_list["asset_sum"]					=	0.0
					balance_sheet_sub_list["liability_current"]				=	0.0
					balance_sheet_sub_list["liability_current_sub1"]		=	0.0
					balance_sheet_sub_list["liability_current_sub2"]		=	0.0
					balance_sheet_sub_list["liability_current_sub3"]		=	0.0
					balance_sheet_sub_list["liability_non_current"]			=	0.0
					balance_sheet_sub_list["liability_non_current_sub1"]	=	0.0
					balance_sheet_sub_list["liability_non_current_sub2"]	=	0.0
					balance_sheet_sub_list["liability_non_current_sub3"]	=	0.0
					balance_sheet_sub_list["liability_non_current_sub4"]	=	0.0
					balance_sheet_sub_list["liability_sum"]					=	0.0
					balance_sheet_sub_list["equity"]						=	0.0
					balance_sheet_sub_list["equity_sub1"]					=	0.0
					balance_sheet_sub_list["equity_sub3"]					=	0.0
					balance_sheet_sub_list["equity_sub2"]					=	0.0
					balance_sheet_sub_list["equity_sum"]					=	0.0

					income_statement_sub_list = {}
					income_statement_sub_list["year"]				=	title[-8:-1]
					income_statement_sub_list["sales"]				=	0.0
					income_statement_sub_list["sales_sub1"]			=	"FINDING LINE NUMBER ERROR"
					income_statement_sub_list["sales_sub2"]			=	0.0
					income_statement_sub_list["sales_sub3"]			=	0.0
					income_statement_sub_list["sales2"]				=	0.0
					income_statement_sub_list["sales2_sub1"]		=	0.0
					income_statement_sub_list["op_income"]		 	=	0.0
					income_statement_sub_list["op_income_sub1"]		=	0.0
					income_statement_sub_list["op_income_sub2"]		=	0.0
					income_statement_sub_list["op_income_sub3"]		=	0.0
					income_statement_sub_list["op_income_sub4"]		=	0.0
					income_statement_sub_list["op_income_sub5"]		=	0.0
					income_statement_sub_list["op_income_sub6"]		=	0.0
					income_statement_sub_list["op_income_sub7"]		=	0.0
					income_statement_sub_list["tax"]				=	0.0
					income_statement_sub_list["net_income"]			=	0.0
					income_statement_sub_list["eps"]				=	0.0

				dart_post_sublist.append(date)
				dart_post_sublist.append(corp_name)
				dart_post_sublist.append(market)
				dart_post_sublist.append(title)
				dart_post_sublist.append(link)
				
				dart_post_list.append(dart_post_sublist)
				cashflow_list.append(cashflow_sub_list)
				balance_sheet_list.append(balance_sheet_sub_list)
				income_statement_list.append(income_statement_sub_list)

	write_excel_file(workbook_name, dart_post_list, cashflow_list, balance_sheet_list, income_statement_list, corp, stock_code, stock_cat)

# Main
if __name__ == "__main__":
	main()


###### anualy ##########
#-*- coding:utf-8 -*-
# Parsing dividends data from DART
import urllib.request
import urllib.parse
import xlsxwriter
import os
import time
import sys
import getopt
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re
import xlrd
import yfinance as yf
import pandas_datareader
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pandas as pd
from pykrx import stock as STOCK
from pykrx import e3


# Scrape value
def find_value(text, unit):
	return int(text.replace(" ","").replace("△","-").replace("(-)","-").replace("(","-").replace(")","").replace(",","").replace("=",""))/unit

# Draw figure of cashflows.
def draw_cashflow_figure(income_list, income_list2, year_list, op_cashflow_list, fcf_list, div_list, stock_close):
	
	for i in range(len(income_list)):
		if income_list[i] == 0.0:
			income_list[i] = income_list2[i]

	fig, ax1 = plt.subplots()

	ax1.plot(year_list, op_cashflow_list, label="Op Cashflow", color='r', marker='D')
	ax1.plot(year_list, fcf_list, label="Free Cashflow", color='y', marker='D')
	ax1.plot(year_list, income_list, label="Net Income", color='b', marker='D')
	ax1.plot(year_list, div_list, label="Dividends", color='g', marker='D')
	#ax1.plot(year_list, cash_equivalents_list, label="Cash & Cash Equivalents", color='magenta', marker='D', linestyle ='dashed')
	ax1.set_xlabel("YEAR")
	ax1.set_xticks(year_list)
	plt.legend(loc=2)

	ax2 = ax1.twinx().twiny()
	ax2.plot(stock_close, label="Stock Price", color='gray')

	#plt.title(corp)
	plt.legend(loc=4)
	plt.show()

# Draw figure of net income & assets.
def draw_corp_history(year_list, asset_sum_list, liability_sum_list, equity_sum_list, sales_list, op_income_list, net_income_list):
	
	fig, ax1 = plt.subplots()

	ax1.bar(year_list, equity_sum_list, label="Equity", color='gray')
	#ax1.plot(year_list, equity_sum_list, label="Equity", color='r', marker='D')
	ax1.plot(year_list, asset_sum_list, label="Asset", color='y', marker='D')
	ax1.plot(year_list, liability_sum_list, label="Liability", color='b', marker='D')
	ax1.plot(year_list, sales_list, label="Sales", color='r', marker='D')
	ax1.set_xlabel("YEAR")
	ax1.set_xticks(year_list)
	plt.legend(loc=2)
	
	ax2 = ax1.twinx().twiny()
	#ax2.plot(year_list, sales_list, label="Sales", color='g', marker='D', linestyle ='dashed')
	ax2.plot(year_list, op_income_list, label="Op income", color='magenta', marker='D', linestyle ='dashed')
	ax2.plot(year_list, net_income_list, label="Net income", color='g', marker='D', linestyle ='dashed')
	plt.legend(loc=4)
	
	plt.show()


# Write financial statements to Excel file.
def write_excel_file(workbook_name, dart_post_list, cashflow_list, balance_sheet_list, income_statement_list, corp, stock_code, stock_cat):
	# Write an Excel file

	#workbook = xlsxwriter.Workbook(workbook_name)
	#if os.path.isfile(os.path.join(cur_dir, workbook_name)):
	#	os.remove(os.path.join(cur_dir, workbook_name))
	workbook = xlsxwriter.Workbook(workbook_name)

	worksheet_result = workbook.add_worksheet('DART사업보고서')
	filter_format = workbook.add_format({'bold':True,
										'fg_color': '#D7E4BC'
										})
	filter_format2 = workbook.add_format({'bold':True
										})

	percent_format = workbook.add_format({'num_format': '0.00%'})

	roe_format = workbook.add_format({'bold':True,
									  'underline': True,
									  'num_format': '0.00%'})

	num_format = workbook.add_format({'num_format':'0.00'})
	num2_format = workbook.add_format({'num_format':'#,##0'})
	num3_format = workbook.add_format({'num_format':'#,##0.00',
									  'fg_color':'#FCE4D6'})

	worksheet_result.set_column('A:A', 10)
	worksheet_result.set_column('B:B', 15)
	worksheet_result.set_column('C:C', 15)
	worksheet_result.set_column('D:D', 20)
	worksheet_result.set_column('H:H', 15)
	worksheet_result.set_column('I:I', 15)
	worksheet_result.set_column('J:J', 15)
	worksheet_result.set_column('K:K', 15)

	worksheet_result.write(0, 0, "날짜", filter_format)
	worksheet_result.write(0, 1, "회사명", filter_format)
	worksheet_result.write(0, 2, "분류", filter_format)
	worksheet_result.write(0, 3, "제목", filter_format)
	worksheet_result.write(0, 4, "link", filter_format)
	worksheet_result.write(0, 5, "결산년도", filter_format)
	worksheet_result.write(0, 6, "영업활동 현금흐름", filter_format)
	worksheet_result.write(0, 7, "영업에서 창출된 현금흐름", filter_format)
	worksheet_result.write(0, 8, "당기순이익", filter_format)
	worksheet_result.write(0, 9, "투자활동 현금흐름", filter_format)
	worksheet_result.write(0, 10, "유형자산의 취득", filter_format)
	worksheet_result.write(0, 11, "무형자산의 취득", filter_format)
	worksheet_result.write(0, 12, "토지의 취득", filter_format)
	worksheet_result.write(0, 13, "건물의 취득", filter_format)
	worksheet_result.write(0, 14, "구축물의 취득", filter_format)
	worksheet_result.write(0, 15, "기계장치의 취득", filter_format)
	worksheet_result.write(0, 16, "건설중인자산의 증가", filter_format)
	worksheet_result.write(0, 17, "차량운반구의 취득", filter_format)
	worksheet_result.write(0, 18, "비품의 취득", filter_format)
	worksheet_result.write(0, 19, "공구기구의 취득", filter_format)
	worksheet_result.write(0, 20, "시험 연구 설비의 취득", filter_format)
	worksheet_result.write(0, 21, "렌탈 자산의 취득", filter_format)
	worksheet_result.write(0, 22, "영업권의 취득", filter_format)
	worksheet_result.write(0, 23, "산업재산권의 취득", filter_format)
	worksheet_result.write(0, 24, "소프트웨어의 취득", filter_format)
	worksheet_result.write(0, 25, "기타의무형자산의 취득", filter_format)
	worksheet_result.write(0, 26, "투자부동산의 취득", filter_format)
	worksheet_result.write(0, 27, "관계기업투자의 취득", filter_format)
	worksheet_result.write(0, 28, "재무활동 현금흐름", filter_format)
	worksheet_result.write(0, 29, "단기차입금의 증가", filter_format)
	worksheet_result.write(0, 30, "배당금 지급", filter_format)
	worksheet_result.write(0, 31, "자기주식의 취득", filter_format)
	worksheet_result.write(0, 32, "기초현금 및 현금성자산", filter_format)
	worksheet_result.write(0, 33, "기말현금 및 현금성자산", filter_format)

	for k in range(len(dart_post_list)):
		worksheet_result.write(k+1,0, dart_post_list[k][0], num2_format)
		worksheet_result.write(k+1,1, dart_post_list[k][1], num2_format)
		worksheet_result.write(k+1,2, dart_post_list[k][2], num2_format)
		worksheet_result.write(k+1,3, dart_post_list[k][3], num2_format)
		worksheet_result.write(k+1,4, dart_post_list[k][4], num2_format)
		worksheet_result.write(k+1,5, cashflow_list[k]	['year']					, num2_format)
		worksheet_result.write(k+1,6, cashflow_list[k]	['op_cashflow']				, num2_format)
		worksheet_result.write(k+1,7, cashflow_list[k]	['op_cashflow_sub1']		, num2_format)
		worksheet_result.write(k+1,8, cashflow_list[k]	['op_cashflow_sub2']		, num2_format)
		worksheet_result.write(k+1,9, cashflow_list[k]	['invest_cashflow']			, num2_format)
		worksheet_result.write(k+1,10, cashflow_list[k]	['invest_cashflow_sub1']	, num2_format)
		worksheet_result.write(k+1,11, cashflow_list[k]	['invest_cashflow_sub2']	, num2_format)
		worksheet_result.write(k+1,12, cashflow_list[k]	['invest_cashflow_sub3']	, num2_format)
		worksheet_result.write(k+1,13, cashflow_list[k]	['invest_cashflow_sub4']	, num2_format)
		worksheet_result.write(k+1,14, cashflow_list[k]	['invest_cashflow_sub5']	, num2_format)
		worksheet_result.write(k+1,15, cashflow_list[k]	['invest_cashflow_sub6']	, num2_format)
		worksheet_result.write(k+1,16, cashflow_list[k]	['invest_cashflow_sub7']	, num2_format)
		worksheet_result.write(k+1,17, cashflow_list[k]	['invest_cashflow_sub8']	, num2_format)
		worksheet_result.write(k+1,18, cashflow_list[k]	['invest_cashflow_sub9']	, num2_format)
		worksheet_result.write(k+1,19, cashflow_list[k]	['invest_cashflow_sub10']	, num2_format)
		worksheet_result.write(k+1,20, cashflow_list[k]	['invest_cashflow_sub11']	, num2_format)
		worksheet_result.write(k+1,21, cashflow_list[k]	['invest_cashflow_sub12']	, num2_format)
		worksheet_result.write(k+1,22, cashflow_list[k]	['invest_cashflow_sub13']	, num2_format)
		worksheet_result.write(k+1,23, cashflow_list[k]	['invest_cashflow_sub14']	, num2_format)
		worksheet_result.write(k+1,24, cashflow_list[k]	['invest_cashflow_sub15']	, num2_format)
		worksheet_result.write(k+1,25, cashflow_list[k]	['invest_cashflow_sub16']	, num2_format)
		worksheet_result.write(k+1,26, cashflow_list[k]	['invest_cashflow_sub17']	, num2_format)
		worksheet_result.write(k+1,27, cashflow_list[k]	['invest_cashflow_sub18']	, num2_format)
		worksheet_result.write(k+1,28, cashflow_list[k]	['fin_cashflow']			, num2_format)
		worksheet_result.write(k+1,29, cashflow_list[k]	['fin_cashflow_sub1']		, num2_format)
		worksheet_result.write(k+1,30, cashflow_list[k]	['fin_cashflow_sub2']		, num2_format)
		worksheet_result.write(k+1,31, cashflow_list[k]	['fin_cashflow_sub3']		, num2_format)
		worksheet_result.write(k+1,32, cashflow_list[k]	['start_cash']				, num2_format)
		worksheet_result.write(k+1,33, cashflow_list[k]	['end_cash']				, num2_format)

	cashflow_list.reverse() 
	worksheet_cashflow = workbook.add_worksheet('Cashflow Statement')
	
	worksheet_cashflow.set_column('A:A', 30)
	worksheet_cashflow.write(0, 0, "결산년도", filter_format)
	worksheet_cashflow.write(1, 0, "영업활동 현금흐름", filter_format)
	worksheet_cashflow.write(2, 0, "영업에서 창출된 현금흐름", filter_format2)
	worksheet_cashflow.write(3, 0, "당기순이익", filter_format2)
	worksheet_cashflow.write(4, 0, "감가상각비", filter_format2)
	worksheet_cashflow.write(5, 0, "신탁계정대", filter_format2)
	worksheet_cashflow.write(6, 0, "투자활동 현금흐름", filter_format)
	worksheet_cashflow.write(7, 0, "유형자산의 취득", filter_format2)
	worksheet_cashflow.write(8, 0, "무형자산의 취득", filter_format2)
	worksheet_cashflow.write(9, 0, "토지의 취득", filter_format2)
	worksheet_cashflow.write(10, 0, "건물의 취득", filter_format2)
	worksheet_cashflow.write(11, 0, "구축물의 취득", filter_format2)
	worksheet_cashflow.write(12, 0, "기계장치의 취득", filter_format2)
	worksheet_cashflow.write(13, 0, "건설중인자산의 증가", filter_format2)
	worksheet_cashflow.write(14, 0, "차량운반구의 취득", filter_format2)
	worksheet_cashflow.write(15, 0, "비품의 취득", filter_format2)
	worksheet_cashflow.write(16, 0, "공구기구의 취득", filter_format2)
	worksheet_cashflow.write(17, 0, "시험 연구 설비의 취득", filter_format2)
	worksheet_cashflow.write(18, 0, "렌탈 자산의 취득", filter_format2)
	worksheet_cashflow.write(19, 0, "영업권의 취득", filter_format2)
	worksheet_cashflow.write(20, 0, "산업재산권의 취득", filter_format2)
	worksheet_cashflow.write(21, 0, "소프트웨어의 취득", filter_format2)
	worksheet_cashflow.write(22, 0, "기타의무형자산의 취득", filter_format2)
	worksheet_cashflow.write(23, 0, "투자부동산의 취득", filter_format2)
	worksheet_cashflow.write(24, 0, "관계기업투자의 취득", filter_format2)
	worksheet_cashflow.write(25, 0, "재무활동 현금흐름", filter_format)
	worksheet_cashflow.write(26, 0, "단기차입금의 증가", filter_format2)
	worksheet_cashflow.write(27, 0, "배당금 지급", filter_format2)
	worksheet_cashflow.write(28, 0, "자기주식의 취득", filter_format2)
	worksheet_cashflow.write(29, 0, "기초현금 및 현금성자산", filter_format)
	worksheet_cashflow.write(30, 0, "기말현금 및 현금성자산", filter_format)
	worksheet_cashflow.write(31, 0, "당기순이익 손익계산서", filter_format2)
	worksheet_cashflow.write(32, 0, "잉여현금흐름(FCF)", filter_format)

	prev_year = 0
	j = 0

	year_list = []
	op_cashflow_list = []
	fcf_list = []
	income_list = []
	income_list2 = []
	div_list = []
	cash_equivalents_list = []

	for k in range(len(cashflow_list)):
		fcf = cashflow_list[k]['op_cashflow']
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub1'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub2'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub3'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub4'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub5'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub6'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub7'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub8'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub9'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub10'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub11'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub12'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub13'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub14'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub15'])
		fcf = fcf - abs(cashflow_list[k]['invest_cashflow_sub16'])
	
		if cashflow_list[k]['op_cashflow_sub1'] != "FINDING LINE NUMBER ERROR":
			# Overwirting
			if prev_year == cashflow_list[k]['year']:
				worksheet_cashflow.write(0, j, str(cashflow_list[k]['year']))
				worksheet_cashflow.write(1, j, cashflow_list[k]['op_cashflow']				, num2_format)
				worksheet_cashflow.write(2, j, cashflow_list[k]['op_cashflow_sub1']			, num2_format)
				worksheet_cashflow.write(3, j, cashflow_list[k]['op_cashflow_sub2']			, num2_format)
				worksheet_cashflow.write(4, j, cashflow_list[k]['op_cashflow_sub3']			, num2_format)
				worksheet_cashflow.write(5, j, cashflow_list[k]['op_cashflow_sub4']			, num2_format)
				worksheet_cashflow.write(6, j, cashflow_list[k]['invest_cashflow']			, num2_format)
				worksheet_cashflow.write(7, j, cashflow_list[k]['invest_cashflow_sub1']		, num2_format)
				worksheet_cashflow.write(8, j, cashflow_list[k]['invest_cashflow_sub2']		, num2_format)
				worksheet_cashflow.write(9, j, cashflow_list[k]['invest_cashflow_sub3']		, num2_format)
				worksheet_cashflow.write(10, j, cashflow_list[k]['invest_cashflow_sub4']	, num2_format)
				worksheet_cashflow.write(11, j, cashflow_list[k]['invest_cashflow_sub5']	, num2_format)
				worksheet_cashflow.write(12, j, cashflow_list[k]['invest_cashflow_sub6']	, num2_format)
				worksheet_cashflow.write(13, j, cashflow_list[k]['invest_cashflow_sub7']	, num2_format)
				worksheet_cashflow.write(14, j, cashflow_list[k]['invest_cashflow_sub8']	, num2_format)
				worksheet_cashflow.write(15, j, cashflow_list[k]['invest_cashflow_sub9']	, num2_format)
				worksheet_cashflow.write(16, j, cashflow_list[k]['invest_cashflow_sub10']	, num2_format)
				worksheet_cashflow.write(17, j, cashflow_list[k]['invest_cashflow_sub11']	, num2_format)
				worksheet_cashflow.write(18, j, cashflow_list[k]['invest_cashflow_sub12']	, num2_format)
				worksheet_cashflow.write(19, j, cashflow_list[k]['invest_cashflow_sub13']	, num2_format)
				worksheet_cashflow.write(20, j, cashflow_list[k]['invest_cashflow_sub14']	, num2_format)
				worksheet_cashflow.write(21, j, cashflow_list[k]['invest_cashflow_sub15']	, num2_format)
				worksheet_cashflow.write(22, j, cashflow_list[k]['invest_cashflow_sub16']	, num2_format)
				worksheet_cashflow.write(23, j, cashflow_list[k]['invest_cashflow_sub17']	, num2_format)
				worksheet_cashflow.write(24, j, cashflow_list[k]['invest_cashflow_sub18']	, num2_format)
				worksheet_cashflow.write(25, j, cashflow_list[k]['fin_cashflow']			, num2_format)
				worksheet_cashflow.write(26, j, cashflow_list[k]['fin_cashflow_sub1']		, num2_format)
				worksheet_cashflow.write(27, j, cashflow_list[k]['fin_cashflow_sub2']		, num2_format)
				worksheet_cashflow.write(28, j, cashflow_list[k]['fin_cashflow_sub3']		, num2_format)
				worksheet_cashflow.write(29, j, cashflow_list[k]['start_cash']				, num2_format)
				worksheet_cashflow.write(30, j, cashflow_list[k]['end_cash']				, num2_format)
				worksheet_cashflow.write(31, j, cashflow_list[k]['net_income']				, num2_format)
				worksheet_cashflow.write(32, j, fcf, num2_format)
				
				year_list[-1] = cashflow_list[k]['year']
				op_cashflow_list[-1] = cashflow_list[k]['op_cashflow']
				fcf_list[-1] = fcf
				income_list[-1] = cashflow_list[k]['op_cashflow_sub2']
				income_list2[-1] = cashflow_list[k]['net_income']
				div_list[-1] = abs(cashflow_list[k]['fin_cashflow_sub2'])
				cash_equivalents_list[-1] = cashflow_list[k]['end_cash']
			else:
				worksheet_cashflow.write(0, j+1, str(cashflow_list[k]['year']))
				worksheet_cashflow.write(1, j+1, cashflow_list[k]['op_cashflow']			, num2_format)
				worksheet_cashflow.write(2, j+1, cashflow_list[k]['op_cashflow_sub1']		, num2_format)
				worksheet_cashflow.write(3, j+1, cashflow_list[k]['op_cashflow_sub2']		, num2_format)
				worksheet_cashflow.write(4, j+1, cashflow_list[k]['op_cashflow_sub3']		, num2_format)
				worksheet_cashflow.write(5, j+1, cashflow_list[k]['op_cashflow_sub4']		, num2_format)
				worksheet_cashflow.write(6, j+1, cashflow_list[k]['invest_cashflow']		, num2_format)
				worksheet_cashflow.write(7, j+1, cashflow_list[k]['invest_cashflow_sub1']	, num2_format)
				worksheet_cashflow.write(8, j+1, cashflow_list[k]['invest_cashflow_sub2']	, num2_format)
				worksheet_cashflow.write(9, j+1, cashflow_list[k]['invest_cashflow_sub3']	, num2_format)
				worksheet_cashflow.write(10, j+1, cashflow_list[k]['invest_cashflow_sub4']	, num2_format)
				worksheet_cashflow.write(11, j+1, cashflow_list[k]['invest_cashflow_sub5']	, num2_format)
				worksheet_cashflow.write(12, j+1, cashflow_list[k]['invest_cashflow_sub6']	, num2_format)
				worksheet_cashflow.write(13, j+1, cashflow_list[k]['invest_cashflow_sub7']	, num2_format)
				worksheet_cashflow.write(14, j+1, cashflow_list[k]['invest_cashflow_sub8']	, num2_format)
				worksheet_cashflow.write(15, j+1, cashflow_list[k]['invest_cashflow_sub9']	, num2_format)
				worksheet_cashflow.write(16, j+1, cashflow_list[k]['invest_cashflow_sub10']	, num2_format)
				worksheet_cashflow.write(17, j+1, cashflow_list[k]['invest_cashflow_sub11']	, num2_format)
				worksheet_cashflow.write(18, j+1, cashflow_list[k]['invest_cashflow_sub12']	, num2_format)
				worksheet_cashflow.write(19, j+1, cashflow_list[k]['invest_cashflow_sub13']	, num2_format)
				worksheet_cashflow.write(20, j+1, cashflow_list[k]['invest_cashflow_sub14']	, num2_format)
				worksheet_cashflow.write(21, j+1, cashflow_list[k]['invest_cashflow_sub15']	, num2_format)
				worksheet_cashflow.write(22, j+1, cashflow_list[k]['invest_cashflow_sub16']	, num2_format)
				worksheet_cashflow.write(23, j+1, cashflow_list[k]['invest_cashflow_sub17']	, num2_format)
				worksheet_cashflow.write(24, j+1, cashflow_list[k]['invest_cashflow_sub18']	, num2_format)
				worksheet_cashflow.write(25, j+1, cashflow_list[k]['fin_cashflow']			, num2_format)
				worksheet_cashflow.write(26, j+1, cashflow_list[k]['fin_cashflow_sub1']		, num2_format)
				worksheet_cashflow.write(27, j+1, cashflow_list[k]['fin_cashflow_sub2']		, num2_format)
				worksheet_cashflow.write(28, j+1, cashflow_list[k]['fin_cashflow_sub3']		, num2_format)
				worksheet_cashflow.write(29, j+1, cashflow_list[k]['start_cash']			, num2_format)
				worksheet_cashflow.write(30, j+1, cashflow_list[k]['end_cash']				, num2_format)
				worksheet_cashflow.write(31, j+1, cashflow_list[k]['net_income']			, num2_format)
				worksheet_cashflow.write(32, j+1, fcf, num2_format)
			
				year_list.append(cashflow_list[k]['year'])
				op_cashflow_list.append(cashflow_list[k]['op_cashflow'])
				fcf_list.append(fcf)
				income_list.append(cashflow_list[k]['op_cashflow_sub2'])
				income_list2.append(cashflow_list[k]['net_income'])
				div_list.append(abs(cashflow_list[k]['fin_cashflow_sub2']))
				cash_equivalents_list.append(cashflow_list[k]['end_cash'])
				j = j+1
		
			prev_year = cashflow_list[k]['year']

	# Balance sheet
	balance_sheet_list.reverse() 
	worksheet_bs= workbook.add_worksheet('Balance Sheet')
	
	prev_year = 0
	j = 0

	asset_sum_list = []
	liability_sum_list = []
	equity_sum_list = []

	worksheet_bs.set_column('A:A', 30)
	worksheet_bs.write(0, 0, "결산년도", filter_format)
	worksheet_bs.write(1, 0, "유동자산", filter_format)
	worksheet_bs.write(2, 0, "현금 및 현금성 자산", filter_format2)
	worksheet_bs.write(3, 0, "매출채권", filter_format2)
	worksheet_bs.write(4, 0, "재고자산", filter_format2)
	worksheet_bs.write(5, 0, "비유동자산", filter_format)
	worksheet_bs.write(6, 0, "유형자산", filter_format2)
	worksheet_bs.write(7, 0, "무형자산", filter_format2)
	worksheet_bs.write(8, 0, "자산총계", filter_format)
	worksheet_bs.write(9, 0, "유동부채", filter_format)
	worksheet_bs.write(10, 0, "매입채무", filter_format2)
	worksheet_bs.write(11, 0, "단기차입금", filter_format2)
	worksheet_bs.write(12, 0, "미지급금", filter_format2)
	worksheet_bs.write(13, 0, "비유동부채", filter_format)
	worksheet_bs.write(14, 0, "사채", filter_format2)
	worksheet_bs.write(15, 0, "장기차입금", filter_format2)
	worksheet_bs.write(16, 0, "장기미지급금", filter_format2)
	worksheet_bs.write(17, 0, "이연법인세부채", filter_format2)
	worksheet_bs.write(18, 0, "부채총계", filter_format)
	worksheet_bs.write(19, 0, "자본금", filter_format2)
	worksheet_bs.write(20, 0, "주식발행초과금", filter_format2)
	worksheet_bs.write(21, 0, "자본잉여금", filter_format2)
	worksheet_bs.write(22, 0, "이익잉여금", filter_format2)
	worksheet_bs.write(23, 0, "자본총계", filter_format)
	
	for k in range(len(balance_sheet_list)):
		if balance_sheet_list[k]['asset_current_sub1'] != "FINDING LINE NUMBER ERROR":
			# Overwirting
			if prev_year == balance_sheet_list[k]['year']:
				asset_sum_list[-1] = balance_sheet_list[k]['asset_sum']
				liability_sum_list[-1] = balance_sheet_list[k]['liability_sum']
				equity_sum_list[-1] = balance_sheet_list[k]['equity_sum']
				w = j
			else:
				asset_sum_list.append(balance_sheet_list[k]['asset_sum'])
				liability_sum_list.append(balance_sheet_list[k]['liability_sum'])
				equity_sum_list.append(balance_sheet_list[k]['equity_sum'])
				w = j+1

			worksheet_bs.write(0, w, str(balance_sheet_list[k]['year']))
			worksheet_bs.write(1, w, balance_sheet_list[k]['asset_current']					, num2_format)
			worksheet_bs.write(2, w, balance_sheet_list[k]['asset_current_sub1']			, num2_format)
			worksheet_bs.write(3, w, balance_sheet_list[k]['asset_current_sub2']			, num2_format)
			worksheet_bs.write(4, w, balance_sheet_list[k]['asset_current_sub3']			, num2_format)
			worksheet_bs.write(5, w, balance_sheet_list[k]['asset_non_current']				, num2_format)
			worksheet_bs.write(6, w, balance_sheet_list[k]['asset_non_current_sub1']		, num2_format)
			worksheet_bs.write(7, w, balance_sheet_list[k]['asset_non_current_sub2']		, num2_format)
			worksheet_bs.write(8, w, balance_sheet_list[k]['asset_sum']						, num2_format)
			worksheet_bs.write(9, w, balance_sheet_list[k]['liability_current']				, num2_format)
			worksheet_bs.write(10, w, balance_sheet_list[k]['liability_current_sub1']		, num2_format)
			worksheet_bs.write(11, w, balance_sheet_list[k]['liability_current_sub2']		, num2_format)
			worksheet_bs.write(12, w, balance_sheet_list[k]['liability_current_sub3']		, num2_format)
			worksheet_bs.write(13, w, balance_sheet_list[k]['liability_non_current']		, num2_format)
			worksheet_bs.write(14, w, balance_sheet_list[k]['liability_non_current_sub1']	, num2_format)
			worksheet_bs.write(15, w, balance_sheet_list[k]['liability_non_current_sub2']	, num2_format)
			worksheet_bs.write(16, w, balance_sheet_list[k]['liability_non_current_sub3']	, num2_format)
			worksheet_bs.write(17, w, balance_sheet_list[k]['liability_non_current_sub4']	, num2_format)
			worksheet_bs.write(18, w, balance_sheet_list[k]['liability_sum']				, num2_format)
			worksheet_bs.write(19, w, balance_sheet_list[k]['equity']						, num2_format)
			worksheet_bs.write(20, w, balance_sheet_list[k]['equity_sub1']					, num2_format)
			worksheet_bs.write(21, w, balance_sheet_list[k]['equity_sub3']					, num2_format)
			worksheet_bs.write(22, w, balance_sheet_list[k]['equity_sub2']					, num2_format)
			worksheet_bs.write(23, w, balance_sheet_list[k]['equity_sum']					, num2_format)
			
			if prev_year != balance_sheet_list[k]['year']:
				j = j+1
			prev_year = balance_sheet_list[k]['year']

	# Income statement
	income_statement_list.reverse() 
	worksheet_income= workbook.add_worksheet('Income Statement')

	prev_year = 0
	j = 0

	sales_list = []
	op_income_list = []
	net_income_list = []
	
	worksheet_income.set_column('A:A', 30)
	worksheet_income.write(0, 0, "결산년도", filter_format)
	worksheet_income.write(1, 0, "매출액", filter_format)
	worksheet_income.write(2, 0, "매출원가", filter_format2)
	worksheet_income.write(3, 0, "매출총이익", filter_format2)
	worksheet_income.write(4, 0, "판매비와관리비", filter_format2)
	worksheet_income.write(5, 0, "영업수익", filter_format)
	worksheet_income.write(6, 0, "영업비용", filter_format2)
	worksheet_income.write(7, 0, "영업이익", filter_format)
	worksheet_income.write(8, 0, "기타수익", filter_format2)
	worksheet_income.write(9, 0, "기타비용", filter_format2)
	worksheet_income.write(10, 0, "금융수익", filter_format2)
	worksheet_income.write(11, 0, "금융비용", filter_format2)
	worksheet_income.write(12, 0, "영업외수익", filter_format2)
	worksheet_income.write(13, 0, "영업외비용", filter_format2)
	worksheet_income.write(14, 0, "법인세비용차감전순이익", filter_format)
	worksheet_income.write(15, 0, "법인세비용", filter_format2)
	worksheet_income.write(16, 0, "당기순이익", filter_format)
	#worksheet_income.write(17, 0, "기본주당이익", filter_format)

	for k in range(len(income_statement_list)):
		if income_statement_list[k]['sales_sub1'] != "FINDING LINE NUMBER ERROR":
			# Overwirting
			if prev_year == income_statement_list[k]['year']:
				w = j
				sales_list[-1]			= income_statement_list[k]['sales']
				op_income_list[-1]		= income_statement_list[k]['op_income']
				net_income_list[-1]		= income_statement_list[k]['net_income']
			else:
				sales_list.append(income_statement_list[k]['sales'])
				op_income_list.append(income_statement_list[k]['op_income'])
				net_income_list.append(income_statement_list[k]['net_income'])
				w = j+1

			worksheet_income.write(0, w, str(income_statement_list[k]['year']))
			worksheet_income.write(1, w, income_statement_list[k] ['sales']				, num2_format)
			worksheet_income.write(2, w, income_statement_list[k] ['sales_sub1']		, num2_format)
			worksheet_income.write(3, w, income_statement_list[k] ['sales_sub2']		, num2_format)
			worksheet_income.write(4, w, income_statement_list[k] ['sales_sub3']		, num2_format)
			worksheet_income.write(5, w, income_statement_list[k] ['sales2']			, num2_format)
			worksheet_income.write(6, w, income_statement_list[k] ['sales2_sub1']		, num2_format)
			worksheet_income.write(7, w, income_statement_list[k] ['op_income']			, num2_format)
			worksheet_income.write(8, w, income_statement_list[k] ['op_income_sub1']	, num2_format)
			worksheet_income.write(9, w, income_statement_list[k] ['op_income_sub2']	, num2_format)
			worksheet_income.write(10, w, income_statement_list[k] ['op_income_sub3']	, num2_format)
			worksheet_income.write(11, w, income_statement_list[k] ['op_income_sub4']	, num2_format)
			worksheet_income.write(12, w, income_statement_list[k]['op_income_sub6']	, num2_format)
			worksheet_income.write(13, w, income_statement_list[k]['op_income_sub7']	, num2_format)
			worksheet_income.write(14, w, income_statement_list[k]['op_income_sub5']	, num2_format)
			worksheet_income.write(15, w, income_statement_list[k]['tax']				, num2_format)
			worksheet_income.write(16, w, income_statement_list[k]['net_income']		, num2_format)
			#worksheet_income.write(17, w, income_statement_list[k]['eps']				, num2_format)
			
			if prev_year != income_statement_list[k]['year']:
				j = j+1
			prev_year = income_statement_list[k]['year']
	
	j = 0
	
	# Chart WORKSHEET	
	#chart = workbook.add_chart({'type':'line'})
	#chart.add_series({
	#				'categories':'=cashflow!$B$1:$Q$1',
	#				'name':'=cashflow!A2',
	#				'values':'=cashflow!$B$2:$Q$2',
	#				'marker':{'type': 'diamond'}
	#				})
	#chart.add_series({
	#				'name':'=cashflow!A4',
	#				'values':'=cashflow!$B$4:$Q$4',
	#				'marker':{'type': 'diamond'}
	#				})
	#chart.add_series({
	#				'name':'=cashflow!A26',
	#				'values':'=cashflow!$B$26:$Q$26',
	#				'marker':{'type': 'diamond'}
	#				})
	#chart.set_legend({'font':{'bold':1}})
	#chart.set_x_axis({'name':"결산년도"})
	#chart.set_y_axis({'name':"단위:억원"})
	#chart.set_title({'name':corp})

	#worksheet_cashflow.insert_chart('C30', chart)

	old_year = cashflow_list[0]['year']

	if (stock_code != ""):
		yf.pdr_override()
		start_date = str(old_year).replace('.', '-')+'-01'
		if stock_cat == "KOSPI":
			ticker = stock_code+'.KS'
		elif stock_cat == 'KOSDAQ':
			ticker = stock_code+'.KQ'

		print("ticker", ticker)
		print("start date", start_date)
		stock_read = pandas_datareader.data.get_data_yahoo(ticker, start_date)
		stock_close = stock_read['Close'].values
		stock_datetime64 = stock_read.index.values

		stock_date = []

		for date in stock_datetime64:
			unix_epoch = np.datetime64(0, 's')
			one_second = np.timedelta64(1, 's')
			seconds_since_epoch = (date - unix_epoch) / one_second
			
			day = datetime.utcfromtimestamp(seconds_since_epoch)
			stock_date.append(day.strftime('%Y-%m-%d'))

		worksheet_stock = workbook.add_worksheet('stock_chart')

		worksheet_stock.write(0, 0, "date")
		worksheet_stock.write(0, 1, "Close")
		
		for i in range(len(stock_close)):
			worksheet_stock.write(i+1, 0, stock_date[i])
			worksheet_stock.write(i+1, 1, stock_close[i])
		
		chart = workbook.add_chart({'type':'line'})
		chart.add_series({
						'categories':'=stock_chart!$A$2:$A$'+str(len(stock_close)+1),
						'name':'=stock_chart!B1',
						'values':'=stock_chart!$B$2:$B$'+str(len(stock_close)+1)
						})
		chart.set_size({'x_scale': 2, 'y_scale': 1})
		worksheet_stock.insert_chart('D3', chart)

	workbook.close()
	# Deactivate
	# draw_cashflow_figure(income_list, income_list2, year_list, op_cashflow_list, fcf_list, div_list, stock_close)
	# draw_corp_history(year_list, asset_sum_list, liability_sum_list, equity_sum_list, sales_list, op_income_list, net_income_list)

# Get information of balance sheet
def scrape_balance_sheet(balance_sheet_table, year, unit):

	#유동자산
	##현금및현금성자산
	##매출채권
	##재고자산
	#비유동자산
	##유형자산
	##무형자산
	#자산총계
	#유동부채
	##매입채무
	##단기차입금
	##미지급금
	#비유동부채
	##사채
	##장기차입금
	##장기미지급금
	##이연법인세부채
	#부채총계
	##자본금
	##주식발행초과금
	##이익잉여금
	#자본총계

	re_asset_list = []

	re_asset_current				=	re.compile("^유[ \s]*동[ \s]*자[ \s]*산([ \s]*합[ \s]*계)*|\.[ \s]*유[ \s]*동[ \s]*자[ \s]*산([ \s]*합[ \s]*계)*")
	re_asset_current_sub1			=	re.compile("현[ \s]*금[ \s]*및[ \s]*현[ \s]*금[ \s]*((성[ \s]*자[ \s]*산)|(등[ \s]*가[ \s]*물))")
	re_asset_current_sub2			=	re.compile("매[ \s]*출[ \s]*채[ \s]*권")
	re_asset_current_sub3			=	re.compile("재[ \s]*고[ \s]*자[ \s]*산")
	re_asset_non_current			=	re.compile("비[ \s]*유[ \s]*동[ \s]*자[ \s]*산|고[ \s]*정[ \s]*자[ \s]*산([ \s]*합[ \s]*계)*")
	re_asset_non_current_sub1		=	re.compile("유[ \s]*형[ \s]*자[ \s]*산")
	re_asset_non_current_sub2		=	re.compile("무[ \s]*형[ \s]*자[ \s]*산")
	re_asset_sum					=	re.compile("자[ \s]*산[ \s]*총[ \s]*계([ \s]*합[ \s]*계)*")
	re_liability_current			=	re.compile("^유[ \s]*동[ \s]*부[ \s]*채([ \s]*합[ \s]*계)*|\.[ \s]*유[ \s]*동[ \s]*부[ \s]*채([ \s]*합[ \s]*계)*")
	re_liability_current_sub1		=	re.compile("매[ \s]*입[ \s]*채[ \s]*무[ \s]*")
	re_liability_current_sub2		=	re.compile("단[ \s]*기[ \s]*차[ \s]*입[ \s]*금")
	re_liability_current_sub3		=	re.compile("^미[ \s]*지[ \s]*급[ \s]*금[ \s]*")
	re_liability_non_current		=	re.compile("^비[ \s]*유[ \s]*동[ \s]*부[ \s]*채|\.[ \s]*비[ \s]*유[ \s]*동[ \s]*부[ \s]*채|고[ \s]*정[ \s]*부[ \s]*채")
	re_liability_non_current_sub1	=	re.compile("사[ \s]*채[ \s]*")
	re_liability_non_current_sub2	=	re.compile("장[ \s]*기[ \s]*차[ \s]*입[ \s]*금")
	re_liability_non_current_sub3	=	re.compile("장[ \s]*기[ \s]*미[ \s]*지[ \s]*급[ \s]*금")
	re_liability_non_current_sub4	=	re.compile("이[ \s]*연[ \s]*법[ \s]*인[ \s]*세[ \s]*부[ \s]*채")
	re_liability_sum				=	re.compile("^부[ \s]*채[ \s]*총[ \s]*계([ \s]*합[ \s]*계)*|\.[ \s]*부[ \s]*채[ \s]*총[ \s]*계([ \s]*합[ \s]*계)*")
	re_equity						=	re.compile("자[ \s]*본[ \s]*금")
	re_equity_sub1					=	re.compile("주[ \s]*식[ \s]*발[ \s]*행[ \s]*초[ \s]*과[ \s]*금")
	re_equity_sub3					=	re.compile("자[ \s]*본[ \s]*잉[ \s]*여[ \s]*금")
	re_equity_sub2					=	re.compile("이[ \s]*익[ \s]*잉[ \s]*여[ \s]*금")
	re_equity_sum					=	re.compile("^자[ \s]*본[ \s]*총[ \s]*계([ \s]*합[ \s]*계)*|\.[ \s]*자[ \s]*본[ \s]*총[ \s]*계([ \s]*합[ \s]*계)*")

	re_asset_list.append(re_asset_current)
	re_asset_list.append(re_asset_current_sub1)
	re_asset_list.append(re_asset_current_sub2)		
	re_asset_list.append(re_asset_current_sub3)		
	re_asset_list.append(re_asset_non_current)
	re_asset_list.append(re_asset_non_current_sub1)	
	re_asset_list.append(re_asset_non_current_sub2)	
	re_asset_list.append(re_asset_sum)
	re_asset_list.append(re_liability_current)
	re_asset_list.append(re_liability_current_sub1)
	re_asset_list.append(re_liability_current_sub2)		
	re_asset_list.append(re_liability_current_sub3)		
	re_asset_list.append(re_liability_non_current)
	re_asset_list.append(re_liability_non_current_sub1)	
	re_asset_list.append(re_liability_non_current_sub2)	
	re_asset_list.append(re_liability_non_current_sub3)	
	re_asset_list.append(re_liability_non_current_sub4)	
	re_asset_list.append(re_liability_sum)
	re_asset_list.append(re_equity)
	re_asset_list.append(re_equity_sub1)
	re_asset_list.append(re_equity_sub3)
	re_asset_list.append(re_equity_sub2)		
	re_asset_list.append(re_equity_sum)

	balance_sheet_sub_list = {}
	balance_sheet_sub_list["asset_current"]					=	0.0
	balance_sheet_sub_list["asset_current_sub1"]			=	0.0
	balance_sheet_sub_list["asset_current_sub2"]			=	0.0
	balance_sheet_sub_list["asset_current_sub3"]			=	0.0
	balance_sheet_sub_list["asset_non_current"]				=	0.0
	balance_sheet_sub_list["asset_non_current_sub1"]		=	0.0
	balance_sheet_sub_list["asset_non_current_sub2"]		=	0.0
	balance_sheet_sub_list["asset_sum"]						=	0.0
	balance_sheet_sub_list['year']							=	year
	balance_sheet_sub_list["liability_current"]				=	0.0
	balance_sheet_sub_list["liability_current_sub1"]		=	0.0
	balance_sheet_sub_list["liability_current_sub2"]		=	0.0
	balance_sheet_sub_list["liability_current_sub3"]		=	0.0
	balance_sheet_sub_list["liability_non_current"]			=	0.0
	balance_sheet_sub_list["liability_non_current_sub1"]	=	0.0
	balance_sheet_sub_list["liability_non_current_sub2"]	=	0.0
	balance_sheet_sub_list["liability_non_current_sub3"]	=	0.0
	balance_sheet_sub_list["liability_non_current_sub4"]	=	0.0
	balance_sheet_sub_list["liability_sum"]					=	0.0
	balance_sheet_sub_list["equity"]						=	0.0
	balance_sheet_sub_list["equity_sub1"]					=	0.0
	balance_sheet_sub_list["equity_sub3"]					=	0.0
	balance_sheet_sub_list["equity_sub2"]					=	0.0
	balance_sheet_sub_list["equity_sum"]					=	0.0

	balance_sheet_key_list = []
	
	balance_sheet_key_list.append("asset_current")
	balance_sheet_key_list.append("asset_current_sub1")
	balance_sheet_key_list.append("asset_current_sub2")
	balance_sheet_key_list.append("asset_current_sub3")
	balance_sheet_key_list.append("asset_non_current")
	balance_sheet_key_list.append("asset_non_current_sub1")
	balance_sheet_key_list.append("asset_non_current_sub2")
	balance_sheet_key_list.append("asset_sum")
	balance_sheet_key_list.append("liability_current")			
	balance_sheet_key_list.append("liability_current_sub1")		
	balance_sheet_key_list.append("liability_current_sub2")		
	balance_sheet_key_list.append("liability_current_sub3")		
	balance_sheet_key_list.append("liability_non_current")		
	balance_sheet_key_list.append("liability_non_current_sub1")	
	balance_sheet_key_list.append("liability_non_current_sub2")	
	balance_sheet_key_list.append("liability_non_current_sub3")	
	balance_sheet_key_list.append("liability_non_current_sub4")	
	balance_sheet_key_list.append("liability_sum")				
	balance_sheet_key_list.append("equity")						
	balance_sheet_key_list.append("equity_sub1")				
	balance_sheet_key_list.append("equity_sub3")				
	balance_sheet_key_list.append("equity_sub2")				
	balance_sheet_key_list.append("equity_sum")					
	
	trs = balance_sheet_table.findAll("tr")

	# Balance sheet statement
	if (len(trs) != 2):
		for tr in trs:
			#print("trs", len(trs))
			tds = tr.findAll("td")
			#print("tds", len(tds))
			try:
				if (len(tds) != 0):
					#print(tds[0].text.strip())
					value = 0.0
					for i in range(len(re_asset_list)):
						if re_asset_list[i].search(tds[0].text.strip()):
							if len(tds)>4:
								if (tds[1].text.strip() != '') and (tds[1].text.strip() != '-'):
									value = find_value(tds[1].text.strip(), unit)
									break # for i in len(re_asset_list)
								elif (tds[2].text.strip() != '') and (tds[2].text.strip() != '-'):
									value = find_value(tds[2].text.strip(), unit)
									break # for i in len(re_asset_list)
							else:
								if (tds[1].text.strip() != '') and (tds[1].text.strip() != '-'):
									value = find_value(tds[1].text.strip(), unit)
									break # for i in len(re_asset_list)
					if value != 0.0 and balance_sheet_sub_list[balance_sheet_key_list[i]] == 0.0:
						balance_sheet_sub_list[balance_sheet_key_list[i]] = value
			except Exception as e:
				print("NET INCOME PARSING ERROR in Balance sheet")
				print(e)
	# Special case
	## if (len(trs) != 2):
	else:	
		tr = trs[1]
		tds = tr.findAll("td")
		
		index_col = []
		prev = 0
		for a in tds[0].childGenerator():
			if (str(a) == "<br/>"):
				if (prev == 1):
					index_col.append('')	
				prev = 1
			else:
				prev = 0
				index_col.append(str(a).strip())	
		data_col = []
		prev = 0
		for b in tds[1].childGenerator():
			if (str(b) == "<br/>"):
				if (prev == 1):
					data_col.append('')	
				prev = 1
			else:
				data_col.append(str(b))	
				prev = 0
		data_col2 = []
		prev = 0
		for b in tds[2].childGenerator():
			if (str(b) == "<br/>"):
				if (prev == 1):
					data_col2.append('')	
				prev = 1
			else:
				data_col2.append(str(b))	
				prev = 0

		print("##################################################")
		print(index_col)
		print(data_col)
		print(data_col2)
		print(len(index_col))
		print(len(data_col))
		index_cnt = 0

		for (index) in (index_col):
			try:
				value = 0.0
				for i in range(len(re_asset_list)):
					if re_asset_list[i].search(index):
						if len(tds)>4:
							if (data_col[index_cnt].strip() != '') and (data_col[index_cnt].strip() != '-'):
								value = find_value(data_col[index_cnt], unit)
								break
							elif (data_col2[index_cnt].strip() != '') and (data_col2[index_cnt].strip() != '-'):
								value = find_value(data_col2[index_cnt], unit)
								break
						else:
							if (data_col[index_cnt].strip() != '') and (data_col[index_cnt].strip() != '-'):
								value = find_value(data_col[index_cnt], unit)
								break
				if value != 0.0 and balance_sheet_sub_list[balance_sheet_key_list[i]] == 0.0:
					balance_sheet_sub_list[balance_sheet_key_list[i]] = value
			except Exception as e:
				print("PARSING ERROR in BALANCE SHEET")
				print(e)
			index_cnt = index_cnt + 1

	print("balance_sheet \t",balance_sheet_sub_list)
	return balance_sheet_sub_list


# Get information of cashflows statements
def scrape_cashflows(cashflow_table, year, unit):

	error_cashflows_list = []
	re_cashflow_list = []

	# Regular expression
	re_op_cashflow			= re.compile("((영업활동)|(영업활동으로[ \s]*인한)|(영업활동으로부터의))[ \s]*([순]*현금[ \s]*흐름)")
	re_op_cashflow_sub1 	= re.compile("((영업에서)|(영업으로부터))[ \s]*창출된[ \s]*현금(흐름)*")
	re_op_cashflow_sub2 	= re.compile("(연[ \s]*결[ \s]*)*당[ \s]*기[ \s]*순[ \s]*((이[ \s]*익)|(손[ \s]*익))")
	re_op_cashflow_sub3 	= re.compile("감[ \s]*가[ \s]*상[ \s]*각[ \s]*비")
	re_op_cashflow_sub4 	= re.compile("신[ \s]*탁[ \s]*계[ \s]*정[ \s]*대")
	
	re_invest_cashflow		= re.compile("투자[ \s]*활동[ \s]*현금[ \s]*흐름|투[ \s]*자[ \s]*활[ \s]*동[ \s]*으[ \s]*로[ \s]*인[ \s]*한[ \s]*[순]*현[ \s]*금[ \s]*흐[ \s]*름")
	re_invest_cashflow_sub1 = re.compile("유[ \s]*형[ \s]*자[ \s]*산[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub2 = re.compile("무[ \s]*형[ \s]*자[ \s]*산[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub3 = re.compile("토[ \s]*지[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub4 = re.compile("건[ \s]*물[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub5 = re.compile("구[ \s]*축[ \s]*물[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub6 = re.compile("기[ \s]*계[ \s]*장[ \s]*치[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub7 = re.compile("건[ \s]*설[ \s]*중[ \s]*인[ \s]*자[ \s]*산[ \s]*의[ \s]*((증[ \s]*가)|(취[ \s]*득))")
	re_invest_cashflow_sub8 = re.compile("차[ \s]*량[ \s]*운[ \s]*반[ \s]*구[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub9 = re.compile("비[ \s]*품[ \s]*의[ \s]*취[ \s]*득|비[ \s]*품[ \s]*의[ \s]*((증[ \s]*가)|(취[ \s]*득))")
	re_invest_cashflow_sub10= re.compile("공[ \s]*구[ \s]*기[ \s]*구[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub11= re.compile("시[ \s]*험[ \s]*연[ \s]*구[ \s]*설[ \s]*비[ \s]*의[ \s]*취[ \s]*득")
	re_invest_cashflow_sub12= re.compile("렌[ \s]*탈[ \s]*자[ \s]*산[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub13= re.compile("영[ \s]*업[ \s]*권[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub14= re.compile("산[ \s]*업[ \s]*재[ \s]*산[ \s]*권[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub15= re.compile("소[ \s]*프[ \s]*트[ \s]*웨[ \s]*어[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub16= re.compile("기[ \s]*타[ \s]*무[ \s]*형[ \s]*자[ \s]*산[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub17= re.compile("투[ \s]*자[ \s]*부[ \s]*통[ \s]*산[ \s]*의[ \s]*((취[ \s]*득)|(증[ \s]*가))")
	re_invest_cashflow_sub18= re.compile("관[ \s]*계[ \s]*기[ \s]*업[ \s]*투[ \s]*자[ \s]*의[ \s]*취[ \s]*득|관계[ \s]*기업[ \s]*투자[ \s]*주식의[ \s]*취득|지분법[ \s]*적용[ \s]*투자[ \s]*주식의[ \s]*취득")
	
	re_fin_cashflow			= re.compile("재무[ \s]*활동[ \s]*현금[ \s]*흐름|재무활동으로[ \s]*인한[ \s]*현금흐름")
	re_fin_cashflow_sub1	= re.compile("단기차입금의[ \s]*순증가")
	re_fin_cashflow_sub2	= re.compile("배당금[ \s]*지급|현금배당금의[ \s]*지급|배당금의[ \s]*지급|현금배당|보통주[ ]*배당[ ]*지급")
	re_fin_cashflow_sub3	= re.compile("자기주식의[ \s]*취득")
	re_start_cash			= re.compile("기초[ ]*현금[ ]*및[ ]*현금성[ ]*자산|기초의[ \s]*현금[ ]*및[ ]*현금성[ ]*자산|기[ \s]*초[ \s]*의[ \s]*현[ \s]*금|기[ \s]*초[ \s]*현[ \s]*금")
	re_end_cash				= re.compile("기말[ ]*현금[ ]*및[ ]*현금성[ ]*자산|기말의[ \s]*현금[ ]*및[ ]*현금성[ ]*자산|기[ \s]*말[ \s]*의[ \s]*현[ \s]*금|기[ \s]*말[ \s]*현[ \s]*금")

	re_cashflow_list.append(re_op_cashflow)
	re_cashflow_list.append(re_op_cashflow_sub1) 	
	re_cashflow_list.append(re_op_cashflow_sub2) 	
	re_cashflow_list.append(re_op_cashflow_sub3) 	
	re_cashflow_list.append(re_op_cashflow_sub4) 	
	
	re_cashflow_list.append(re_invest_cashflow)		
	re_cashflow_list.append(re_invest_cashflow_sub1) 
	re_cashflow_list.append(re_invest_cashflow_sub2) 
	re_cashflow_list.append(re_invest_cashflow_sub3) 
	re_cashflow_list.append(re_invest_cashflow_sub4) 
	re_cashflow_list.append(re_invest_cashflow_sub5) 
	re_cashflow_list.append(re_invest_cashflow_sub6) 
	re_cashflow_list.append(re_invest_cashflow_sub7) 
	re_cashflow_list.append(re_invest_cashflow_sub8) 
	re_cashflow_list.append(re_invest_cashflow_sub9) 
	re_cashflow_list.append(re_invest_cashflow_sub10)
	re_cashflow_list.append(re_invest_cashflow_sub11)
	re_cashflow_list.append(re_invest_cashflow_sub12)
	re_cashflow_list.append(re_invest_cashflow_sub13)
	re_cashflow_list.append(re_invest_cashflow_sub14)
	re_cashflow_list.append(re_invest_cashflow_sub15)
	re_cashflow_list.append(re_invest_cashflow_sub16)
	re_cashflow_list.append(re_invest_cashflow_sub17)
	re_cashflow_list.append(re_invest_cashflow_sub18)
	
	re_cashflow_list.append(re_fin_cashflow)		
	re_cashflow_list.append(re_fin_cashflow_sub1)	
	re_cashflow_list.append(re_fin_cashflow_sub2)	
	re_cashflow_list.append(re_fin_cashflow_sub3)	
	re_cashflow_list.append(re_start_cash)
	re_cashflow_list.append(re_end_cash)


	# 영업현금흐름
	## 영업에서 창출된 현금흐름
	## 당기순이익
	## 신탁계정대
	# 투자현금흐름
	## 유형자산의 취득
	## 무형자산의 취득
	## 토지의 취득
	## 건물의 취득
	## 구축물의 취득
	## 기계장치의 취득
	## 건설중인자산의증가
	## 차량운반구의 취득
	## 영업권의 취득
	## 산업재산권의 취득
	## 기타의무형자산의취득
	## 투자부동산의 취득
	## 관계기업투자의취득
	# 재무현금흐름
	## 단기차입금의 순증가
	## 배당금 지급
	## 자기주식의 취득
	# 기초 현금 및 현금성자산
	# 기말 현금 및 현금성자산

	cashflow_sub_list = {}
	
	cashflow_sub_list['year']					= year
	cashflow_sub_list["op_cashflow"]			= 0.0
	cashflow_sub_list["op_cashflow_sub1"]		= 0.0
	cashflow_sub_list["op_cashflow_sub2"]		= 0.0
	cashflow_sub_list["op_cashflow_sub3"]		= 0.0
	cashflow_sub_list["op_cashflow_sub4"]		= 0.0
	cashflow_sub_list["invest_cashflow"]		= 0.0
	cashflow_sub_list["invest_cashflow_sub1"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub2"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub3"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub4"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub5"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub6"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub7"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub8"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub9"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub10"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub11"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub12"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub13"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub14"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub15"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub16"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub17"]	= 0.0
	cashflow_sub_list["invest_cashflow_sub18"]	= 0.0
	cashflow_sub_list["fin_cashflow"]			= 0.0
	cashflow_sub_list["fin_cashflow_sub1"]		= 0.0
	cashflow_sub_list["fin_cashflow_sub2"]		= 0.0
	cashflow_sub_list["fin_cashflow_sub3"]		= 0.0
	cashflow_sub_list["start_cash"]				= 0.0
	cashflow_sub_list["end_cash"]				= 0.0

	cashflow_key_list = []

	cashflow_key_list.append("op_cashflow")
	cashflow_key_list.append("op_cashflow_sub1")
	cashflow_key_list.append("op_cashflow_sub2")
	cashflow_key_list.append("op_cashflow_sub3")
	cashflow_key_list.append("op_cashflow_sub4")
	cashflow_key_list.append("invest_cashflow")
	cashflow_key_list.append("invest_cashflow_sub1")
	cashflow_key_list.append("invest_cashflow_sub2")
	cashflow_key_list.append("invest_cashflow_sub3")
	cashflow_key_list.append("invest_cashflow_sub4")
	cashflow_key_list.append("invest_cashflow_sub5")
	cashflow_key_list.append("invest_cashflow_sub6")
	cashflow_key_list.append("invest_cashflow_sub7")
	cashflow_key_list.append("invest_cashflow_sub8")
	cashflow_key_list.append("invest_cashflow_sub9")
	cashflow_key_list.append("invest_cashflow_sub10")
	cashflow_key_list.append("invest_cashflow_sub11")
	cashflow_key_list.append("invest_cashflow_sub12")
	cashflow_key_list.append("invest_cashflow_sub13")
	cashflow_key_list.append("invest_cashflow_sub14")
	cashflow_key_list.append("invest_cashflow_sub15")
	cashflow_key_list.append("invest_cashflow_sub16")
	cashflow_key_list.append("invest_cashflow_sub17")
	cashflow_key_list.append("invest_cashflow_sub18")
	cashflow_key_list.append("fin_cashflow")
	cashflow_key_list.append("fin_cashflow_sub1")
	cashflow_key_list.append("fin_cashflow_sub2")
	cashflow_key_list.append("fin_cashflow_sub3")
	cashflow_key_list.append("start_cash")
	cashflow_key_list.append("end_cash")

	#net_income = 0.0
	#print("len(trs)", len(trs))
	
	trs = cashflow_table.findAll("tr")
			
	# CASHFLOW statement
	if (len(trs) != 2):
		for tr in trs:
			#print("trs", len(trs))
			tds = tr.findAll("td")
			#print("tds", len(tds))
			try:
				if (len(tds) != 0):
					#print(tds[0].text.strip())

					value = 0.0
					for i in range(len(re_cashflow_list)):
						if re_cashflow_list[i].search(tds[0].text.strip()):
							if len(tds)>4:
								if (tds[1].text.strip() != '') and (tds[1].text.strip() != '-'):
									value = find_value(tds[1].text.strip(), unit)
									break # for i in len(re_cashflow_list)
								elif (tds[2].text.strip() != '') and (tds[2].text.strip() != '-'):
									value = find_value(tds[2].text.strip(), unit)
									break # for i in len(re_cashflow_list)
							else:
								if (tds[1].text.strip() != '') and (tds[1].text.strip() != '-'):
									value = find_value(tds[1].text.strip(), unit)
									break # for i in len(re_cashflow_list)
					if value != 0.0 and cashflow_sub_list[cashflow_key_list[i]] == 0.0:
						cashflow_sub_list[cashflow_key_list[i]] = value
					# No matching case
					else:
						error_cashflows_list.append(tds[0].text.strip())
			except Exception as e:
				print("NET INCOME PARSING ERROR in Cashflows")
				cashflow_sub_list["op_cashflow_sub1"] = "PARSING ERROR"
				print(e)
	# Special case
	## if (len(trs) != 2):
	else:	
		tr = trs[1]
		tds = tr.findAll("td")
		
		index_col = []
		prev = 0
		for a in tds[0].childGenerator():
			if (str(a) == "<br/>"):
				if (prev == 1):
					index_col.append('')	
				prev = 1
			else:
				prev = 0
				index_col.append(str(a).strip())	
		data_col = []
		prev = 0
		for b in tds[1].childGenerator():
			if (str(b) == "<br/>"):
				if (prev == 1):
					data_col.append('0')	
				prev = 1
			else:
				data_col.append(str(b))	
				prev = 0
		data_col2 = []
		prev = 0
		for b in tds[2].childGenerator():
			if (str(b) == "<br/>"):
				if (prev == 1):
					data_col2.append('')	
				prev = 1
			else:
				data_col2.append(str(b))	
				prev = 0

		#print(index_col)
		#print(data_col)
		print(len(index_col))
		print(len(data_col))
		index_cnt = 0

		for (index) in (index_col):
			try:
				value = 0.0
				for i in range(len(re_cashflow_list)):
					if re_cashflow_list[i].search(index):
						if len(tds)>4:
							if (data_col[index_cnt].strip() != '') and (data_col[index_cnt].strip() != '-'):
								value = find_value(data_col[index_cnt], unit)
								break
							elif (data_col2[index_cnt].strip() != '') and (data_col2[index_cnt].strip() != '-'):
								value = find_value(data_col2[index_cnt], unit)
								break
						else:
							if (data_col[index_cnt].strip() != '') and (data_col[index_cnt].strip() != '-'):
								value = find_value(data_col[index_cnt], unit)
								break
				if value != 0.0 and cashflow_sub_list[cashflow_key_list[i]] == 0.0:
					cashflow_sub_list[cashflow_key_list[i]] = value
			except Exception as e:
				print("PARSING ERROR")
				cashflow_sub_list["op_cashflow_sub1"] = "PARSING ERROR"
				print(e)
			index_cnt = index_cnt + 1

	print("cashflow_sheet \t",cashflow_sub_list)
	print("error_cashflow_sheet \t",error_cashflows_list)
	return cashflow_sub_list

# Get information of income statements
def scrape_income_statement(income_table, year, unit, mode):

	#매출액
	#매출원가
	#매출총이익
	#판매비와관리비
	#영업이익
	#기타수익
	#기타비용
	#금융수익
	#금융비용
	#법인세비용차감전순이익
	#번인세비용
	#당기순이익
	#기본주당이익

	re_income_list = []
	
	# Regular expression
	re_sales			=	re.compile("^매[ \s]*출[ \s]*액|\.[ \s]*매[ \s]*출[ \s]*액|\(매출액\)")
	re_sales_sub1		= 	re.compile("^매[ \s]*출[ \s]*원[ \s]*가|\.[ \s]*매[ \s]*출[ \s]*원[ \s]*가")
	re_sales_sub2		= 	re.compile("^매[ \s]*출[ \s]*총[ \s]*이[ \s]*익|\.[ \s]*매[ \s]*출[ \s]*총[ \s]*이[ \s]*익")
	re_sales_sub3		= 	re.compile("판[ \s]*매[ \s]*비[ \s]*와[ \s]*관[ \s]*리[ \s]*비")
	re_sales2			=	re.compile("^영[ \s]*업[ \s]*수[ \s]*익|\.[ \s]*영[ \s]*업[ \s]*수[ \s]*익")
	re_sales2_sub1		= 	re.compile("^영[ \s]*업[ \s]*비[ \s]*용|\.[ \s]*영[ \s]*업[ \s]*비[ \s]*용")
	re_op_income		= 	re.compile("^영[ \s]*업[ \s]*이[ \s]*익|\.[ \s]*영[ \s]*업[ \s]*이[ \s]*익")
	re_op_income_sub1	= 	re.compile("기[ \s]*타[ \s]*수[ \s]*익")
	re_op_income_sub2	= 	re.compile("기[ \s]*타[ \s]*비[ \s]*용")
	re_op_income_sub3	= 	re.compile("금[ \s]*융[ \s]*수[ \s]*익")
	re_op_income_sub4	= 	re.compile("금[ \s]*융[ \s]*비[ \s]*용")
	re_op_income_sub6	= 	re.compile("영[ \s]*업[ \s]*외[ \s]*수[ \s]*익")
	re_op_income_sub7	= 	re.compile("영[ \s]*업[ \s]*외[ \s]*비[ \s]*용")
	re_op_income_sub5	= 	re.compile("법[ \s]*인[ \s]*세[ \s]*비[ \s]*용[ \s]*차[ \s]*감[ \s]*전[ \s]*순[ \s]*((이[ \s]*익)|(손[ \s]*실))|법[ \s]*인[ \s]*세[ \s]*차[ \s]*감[ \s]*전[ \s]*계[ \s]*속[ \s]*영[ \s]*업[ \s]*순[ \s]*이[ \s]*익|법인세[ \s]*차감전[ \s]*순이익|법인세차감전계속영업이익|법인세비용차감전이익|법인세비용차감전계속영업[순]*이익|법인세비용차감전당기순이익|법인세비용차감전순이익|법인세비용차감전[ \s]*계속사업이익|법인세비용차감전순손익")
	re_tax				=	re.compile("법[ \s]*인[ \s]*세[ \s]*비[ \s]*용")
	re_net_income		=	re.compile("^순[ \s]*이[ \s]*익|^당[ \s]*기[ \s]*순[ \s]*이[ \s]*익|^연[ ]*결[ ]*[총 ]*당[ ]*기[ ]*순[ ]*이[ ]*익|지배기업의 소유주에게 귀속되는 당기순이익|분기순이익|당\(분\)기순이익|\.[ \s]*당[ \s]*기[ \s]*순[ \s]*이[ \s]*익|당분기연결순이익")
	re_eps				=	re.compile("기[ \s]*본[ \s]*주[ \s]*당[ \s]*((수[ \s]*익)|([순 \s]*이[ \s]*익))")

	re_income_list.append(re_sales)	
	re_income_list.append(re_sales_sub1)		 	
	re_income_list.append(re_sales_sub2)		 	
	re_income_list.append(re_sales_sub3)		 	
	re_income_list.append(re_sales2)	
	re_income_list.append(re_sales2_sub1)		 	
	re_income_list.append(re_op_income)		 	
	re_income_list.append(re_op_income_sub1)	 	
	re_income_list.append(re_op_income_sub2)	 	
	re_income_list.append(re_op_income_sub3)	 	
	re_income_list.append(re_op_income_sub4)	 	
	re_income_list.append(re_op_income_sub5)	 	
	re_income_list.append(re_op_income_sub6)	 	
	re_income_list.append(re_op_income_sub7)	 	
	re_income_list.append(re_tax)
	re_income_list.append(re_net_income)
	re_income_list.append(re_eps)				

	income_statement_sub_list = {}
	income_statement_sub_list["sales"]				=	0.0
	income_statement_sub_list["sales_sub1"]			=	0.0
	income_statement_sub_list["sales_sub2"]			=	0.0
	income_statement_sub_list["sales_sub3"]			=	0.0
	income_statement_sub_list["sales2"]				=	0.0
	income_statement_sub_list["sales2_sub1"]		=	0.0
	income_statement_sub_list["op_income"]		 	=	0.0
	income_statement_sub_list["op_income_sub1"]		=	0.0
	income_statement_sub_list["op_income_sub2"]		=	0.0
	income_statement_sub_list["op_income_sub3"]		=	0.0
	income_statement_sub_list["op_income_sub4"]		=	0.0
	income_statement_sub_list["op_income_sub5"]		=	0.0
	income_statement_sub_list["op_income_sub6"]		=	0.0
	income_statement_sub_list["op_income_sub7"]		=	0.0
	income_statement_sub_list["tax"]				=	0.0
	income_statement_sub_list["net_income"]			=	0.0
	income_statement_sub_list["eps"]				=	0.0
	income_statement_sub_list['year']				=	year

	income_statement_key_list = []
	income_statement_key_list.append("sales")			
	income_statement_key_list.append("sales_sub1")		
	income_statement_key_list.append("sales_sub2")		
	income_statement_key_list.append("sales_sub3")		
	income_statement_key_list.append("sales2")			
	income_statement_key_list.append("sales2_sub1")		
	income_statement_key_list.append("op_income")		
	income_statement_key_list.append("op_income_sub1")	
	income_statement_key_list.append("op_income_sub2")	
	income_statement_key_list.append("op_income_sub3")	
	income_statement_key_list.append("op_income_sub4")	
	income_statement_key_list.append("op_income_sub5")	
	income_statement_key_list.append("op_income_sub6")	
	income_statement_key_list.append("op_income_sub7")	
	income_statement_key_list.append("tax")			
	income_statement_key_list.append("net_income")		
	income_statement_key_list.append("eps")			

	trs = income_table.findAll("tr")

	# Income statement
	if (len(trs) != 2):
		for income_tr in trs:
			tds = income_tr.findAll("td")
			try:
				if (len(tds) != 0):
					#print(tds[0].text.strip())
					value = 0.0
					for i in range(len(re_income_list)):
						if re_income_list[i].search(tds[0].text.strip()):
							if mode == 0:
								if len(tds)>4:
									if (tds[1].text.strip() != '') and (tds[1].text.strip() != '-'):
										value = find_value(tds[1].text.strip(), unit)
										break # for i in len(re_income_list)
									elif (tds[2].text.strip() != '') and (tds[2].text.strip() != '-'):
										value = find_value(tds[2].text.strip(), unit)
										break # for i in len(re_income_list)
								else:
									if (tds[1].text.strip() != '') and (tds[1].text.strip() != '-'):
										value = find_value(tds[1].text.strip(), unit)
										break # for i in len(re_income_list)
							# mode 1
							else:
								if len(tds)>4:
									if (tds[3].text.strip() != '') and (tds[3].text.strip() != '-'):
										value = find_value(tds[2].text.strip(), unit)
										break # for i in len(re_income_list)
								else:
									if (tds[2].text.strip() != '') and (tds[2].text.strip() != '-'):
										value = find_value(tds[1].text.strip(), unit)
										break # for i in len(re_income_list)
					if value != 0.0 and income_statement_sub_list[income_statement_key_list[i]] == 0.0:
						income_statement_sub_list[income_statement_key_list[i]] = value
			except Exception as e:
				print("NET INCOME PARSING ERROR in Income statement")
				print(e)
				net_income = 0.0
	## if (len(trs) != 2):
	else:	
		income_tr = trs[1]
		tds = income_tr.findAll("td")
		
		index_col = []
		prev = 0
		for a in tds[0].childGenerator():
			if (str(a) == "<br/>"):
				if (prev == 1):
					index_col.append('')	
				prev = 1
			else:
				prev = 0
				index_col.append(str(a).strip())	
		data_col = []
		prev = 0
		for b in tds[1].childGenerator():
			if (str(b) == "<br/>"):
				if (prev == 1):
					data_col.append('0')	
				prev = 1
			else:
				data_col.append(str(b))	
				prev = 0
		data_col2 = []
		prev = 0
		for b in tds[2].childGenerator():
			if (str(b) == "<br/>"):
				if (prev == 1):
					data_col2.append('')	
				prev = 1
			else:
				data_col2.append(str(b))	
				prev = 0

		
		print(len(index_col))
		print(len(data_col))
		index_cnt = 0

		for (index) in (index_col):
			try:
				value = 0.0
				for i in range(len(re_income_list)):
					if re_income_list[i].search(index):
						if len(tds)>4:
							if (data_col[index_cnt].strip() != '') and (data_col[index_cnt].strip() != '-'):
								value = find_value(data_col[index_cnt], unit)
								break
							elif (data_col2[index_cnt].strip() != '') and (data_col2[index_cnt].strip() != '-'):
								value = find_value(data_col2[index_cnt], unit)
								break
						else:
							if (data_col[index_cnt].strip() != '') and (data_col[index_cnt].strip() != '-'):
								value = find_value(data_col[index_cnt], unit)
								break
				if value != 0.0 and income_statement_sub_list[income_statement_key_list[i]] == 0.0:
					income_statement_sub_list[income_statement_key_list[i]] = value
			except Exception as e:
				print("PARSING ERROR in INCOME STATEMENT")
				print(e)
			index_cnt = index_cnt + 1

	print("income_sheet \t",income_statement_sub_list)
	return income_statement_sub_list

# Main function
def main():

	# Default
	corp = "민앤지"
	#corp = "LG화학"
	workbook_name = "{}_Dart_financial_statement.xlsx".format(corp)

	try:
		opts, args = getopt.getopt(sys.argv[1:], "c:o:h", ["corp=", "output", "help"])
	except getopt.GetoptError as err:
		print(err)
		sys.exit(2)
	for option, argument in opts:
		if option == "-h" or option == "--help":
			help_msg = """
================================================================================
-c or --corp <name>     :  Corporation name
-o or --output <name>	:  Output file name
-h or --help            :  Show help messages

<Example>
>> python dart_financial_statement.py -c S-Oil
================================================================================
					"""
			print(help_msg)
			sys.exit(2)
		elif option == "--corp" or option == "-c":
			corp = argument
		elif option == "--output" or option == "-o":
			workbook_name = argument + ".xlsx"

	re_income_find = re.compile("법[ \s]*인[ \s]*세[ \s]*비[ \s]*용(\(이익\))*[ \s]*차[ \s]*감[ \s]*전[ \s]*순[ \s]*((이[ \s]*익)|(손[ \s]*실))|법[ \s]*인[ \s]*세[ \s]*차[ \s]*감[ \s]*전[ \s]*계[ \s]*속[ \s]*영[ \s]*업[ \s]*순[ \s]*이[ \s]*익|법인세[ \s]*차감전[ \s]*순이익|법인세차감전계속영업이익|법인세비용차감전이익|법인세비용차감전계속영업[순]*이익|법인세비용차감전당기순이익|법인세(비용차감|손익가감)전순이익|법인세비용차감전[ \s]*계속사업이익|법인세비용차감전순손익")
	re_cashflow_find = re.compile("영업활동[ \s]*현금[ \s]*흐름|영업활동으로[ \s]*인한[ \s]*[순]*현금[ \s]*흐름|영업활동으로부터의[ \s]*현금흐름|영업활동으로 인한 자산부채의 변동")
	re_balance_sheet_find = re.compile("현[ \s]*금[ \s]*및[ \s]*현[ \s]*금[ \s]*((성[ \s]*자[ \s]*산)|(등[ \s]*가[ \s]*물))")

	### PART I - Read Excel file for stock lists
	if os.path.isfile("tickers.pk"):
		tickers = pd.read_pickle("tickers.pk")
		print("load url pickle")
	else:
		tickers_code = STOCK.get_market_ticker_code_list()
		tickers_etf = e3.get_etf_ticker_list()
		tickers = tickers_code['종목'].append(tickers_etf)
		tickers = pd.DataFrame(tickers.values, index=tickers.index, columns=['종목'])
		tickers['시장'] = tickers_code['시장']
		tickers['시장'] = tickers['시장'].replace(np.nan, 'ETF')
		tickers.to_pickle("tickers.pk")
		tickers.to_csv("tickers.csv", encoding='ms949')
	# pdb.set_trace()
	# stock_cat_list = tickers['시장'].tolist()
	# stock_name_list = tickers['종목'].tolist()
	# stock_num_list = tickers.index.tolist()
	#
	#
	# num_stock = 2040
	# input_file = "basic_20171221.xlsx"
	# cur_dir = os.getcwd()
	# workbook_read_name = input_file
	
	# stock_cat_list = []
	# stock_name_list = []
	# stock_num_list = []
	# stock_url_list = []
	#
	# workbook_read = xlrd.open_workbook(os.path.join(cur_dir, workbook_read_name))
	# sheet_list = workbook_read.sheets()
	# sheet1 = sheet_list[0]
	#
	# for i in range(num_stock):
	# 	stock_cat_list.append(sheet1.cell(i+1,0).value)
	# 	stock_name_list.append(sheet1.cell(i+1,1).value)
	# 	stock_num_list.append(sheet1.cell(i+1,2).value)
	# 	# stock_url_list.append(sheet1.cell(i+1,3).value)
	#
	# find_index = stock_name_list.index(corp)
	#
	# stock_code = ""
	corp_info = tickers[tickers['종목']==corp]
	if len(corp_info) != 0:
		stock_code = corp_info.index[0]
		stock_cat = corp_info['시장'].values[0]
	else:
		print("STOCK CODE ERROR")
		raise AssertionError("STOCK CODE ERROR")
	# URL
	#url_templete = "http://dart.fss.or.kr/dsab002/search.ax?reportName=%s&&maxResults=100&&textCrpNm=%s"
	url_templete = "http://dart.fss.or.kr/dsab002/search.ax?reportName=%s&&maxResults=100&&textCrpNm=%s&&startDate=%s&&endDate=%s"
	headers = {'Cookie':'DSAB002_MAXRESULTS=5000;'}
	
	dart_post_list = []
	cashflow_list = []
	balance_sheet_list = []
	income_statement_list = []
	
	year = 2010
	start_day = datetime(2005,1,1)
	#start_day = datetime(2000,1,1)
	#end_day = datetime(2002,11,15)
	end_day = datetime(2019,5,28)
	delta = end_day - start_day

	# 사업보고서 "%EB %B3%B4 %EA%B3 %A0%EC %84%9C"
	report = "%EC%82%AC%EC%97%85%EB%B3%B4%EA%B3%A0%EC%84%9C"
	# 분기보고서
	report2 = "%EB%B6%84%EA%B8%B0%EB%B3%B4%EA%B3%A0%EC%84%9C"
	# 반기보고서
	report3 = "%EB%B0%98%EA%B8%B0%EB%B3%B4%EA%B3%A0%EC%84%9C"
	# print(report2.decode())

	start_day2 = datetime(2005,1,1)
	end_day2 = datetime(2019,5,28)


	# 최신 분기보고서 읽기
	handle = urllib.request.urlopen(url_templete % (report2, urllib.parse.quote(corp), start_day2.strftime('%Y%m%d'), end_day2.strftime('%Y%m%d')))

	data = handle.read()
	soup = BeautifulSoup(data, 'html.parser', from_encoding='utf-8')
	
	table = soup.find('table')
	trs = table.findAll('tr')
	tds = table.findAll('td')
	counts = len(tds)
	
	if counts > 2:
		# Delay operation
		#time.sleep(20)
	
		link_list = []
		date_list = []
		corp_list = []
		market_list = []
		title_list = []
		reporter_list = []

		# recent report
		tr = trs[1]
		time.sleep(2)
		tds = tr.findAll('td')
		link = 'http://dart.fss.or.kr' + tds[2].a['href']
		date = tds[4].text.strip().replace('.', '-')
		corp_name = tds[1].text.strip()
		market = tds[1].img['title']
		title = " ".join(tds[2].text.split())
		reporter = tds[3].text.strip()

		link_list.append(link)
		date_list.append(date)
		corp_list.append(corp_name)
		market_list.append(market)
		title_list.append(title)
		reporter_list.append(reporter)
	
		dart_post_sublist = []

		year = title[-8:-1] ##첫번째 분기값 year 결정하는 곳
		print(year)
		print(corp_name)
		print(title)
		print(date)
		handle = urllib.request.urlopen(link)
		data = handle.read()
		soup2 = BeautifulSoup(data, 'html.parser', from_encoding='utf-8')
		print(link)
		
		head_lines = soup2.find('head').text.split("\n")
		#print(head_lines)
		re_tree_find1 = re.compile("2.[ ]*연결재무제표")
		re_tree_find1_bak = re.compile("4.[ ]*재무제표")
		line_num = 0
		line_find = 0
		for head_line in head_lines:
			#print(head_line)
			if (re_tree_find1.search(head_line)):
				line_find = line_num
				break
			line_num = line_num + 1
		
		line_num = 0
		line_find_bak = 0
		for head_line in head_lines:
			if (re_tree_find1_bak.search(head_line)):
				line_find_bak = line_num
				break
			line_num = line_num + 1


		if(line_find != 0):
		
			line_words = head_lines[line_find+4].split("'")
			#print(line_words)
			rcpNo = line_words[1]
			dcmNo = line_words[3]
			eleId = line_words[5]
			offset = line_words[7]
			length = line_words[9]

			dart = soup2.find_all(string=re.compile('dart.dtd'))
			dart2 = soup2.find_all(string=re.compile('dart2.dtd'))
			dart3 = soup2.find_all(string=re.compile('dart3.xsd'))

			if len(dart3) != 0:
				link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=" + eleId + "&offset=" + offset + "&length=" + length + "&dtd=dart3.xsd"
			elif len(dart2) != 0:
				link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=" + eleId + "&offset=" + offset + "&length=" + length + "&dtd=dart2.dtd"
			elif len(dart) != 0:
				link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=" + eleId + "&offset=" + offset + "&length=" + length + "&dtd=dart.dtd"
			else:
				link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=0&offset=0&length=0&dtd=HTML"  
			
			print(link2)

			#try:
			handle = urllib.request.urlopen(link2)
			print(handle) ## <http.client.HTTPResponse object at 0x0D86EC90>
			data = handle.read()
			soup3 = BeautifulSoup(data, 'html.parser', from_encoding='utf-8')

			tables = soup3.findAll("table")

			# 2. 연결재무제표가 비어 있는 경우
			if (len(tables) == 0):
				line_words = head_lines[line_find_bak+4].split("'")
				#print(line_words)
				rcpNo = line_words[1]
				dcmNo = line_words[3]
				eleId = line_words[5]
				offset = line_words[7]
				length = line_words[9]

				dart = soup2.find_all(string=re.compile('dart.dtd'))
				dart2 = soup2.find_all(string=re.compile('dart2.dtd'))
				dart3 = soup2.find_all(string=re.compile('dart3.xsd'))

				if len(dart3) != 0:
					link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=" + eleId + "&offset=" + offset + "&length=" + length + "&dtd=dart3.xsd"
				elif len(dart2) != 0:
					link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=" + eleId + "&offset=" + offset + "&length=" + length + "&dtd=dart2.dtd"
				elif len(dart) != 0:
					link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=" + eleId + "&offset=" + offset + "&length=" + length + "&dtd=dart.dtd"
				else:
					link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=0&offset=0&length=0&dtd=HTML"  
				
				print(link2)
				
				handle = urllib.request.urlopen(link2)
				print(handle) ## <http.client.HTTPResponse object at 0x0D86EC90>
				data = handle.read()
				soup3 = BeautifulSoup(data, 'html.parser', from_encoding='utf-8')
				tables = soup3.findAll("table")

			cnt = 0
			table_num = 0

			for table in tables:
				if (re_cashflow_find.search(table.text)):
					table_num = cnt
					break
				cnt = cnt + 1
			
			print("table_num", table_num, "Tables", len(tables)) ## table_num 7 Tables 8
			cashflow_table = soup3.findAll("table")[table_num]
			
			cnt = 0
			table_income_num = 0
			for table in tables:
				if (re_income_find.search(table.text)):
					table_income_num = cnt
					break
				cnt = cnt + 1
			income_table = soup3.findAll("table")[table_income_num]
			#print("table_income_num", table_income_num, "Tables", len(tables))
			
			cnt = 0
			table_balance_num = 0
			for table in tables:
				if (re_balance_sheet_find.search(table.text)):
					table_balance_num = cnt
					break
				cnt = cnt + 1
			balance_table = soup3.findAll("table")[table_balance_num]
			print("table_balance_num", table_balance_num, "Tables", len(tables)) ## table_balance_num 1 Tables 8
			
			unit = 100.0
			unit_find = 0
			re_unit1 = re.compile('단위[ \s]*:[ \s]*원')
			re_unit2 = re.compile('단위[ \s]*:[ \s]*백만원')
			re_unit3 = re.compile('단위[ \s]*:[ \s]*천원')

			# 원
			if len(soup3.findAll("table")[table_num-1](string=re_unit1)) != 0:
				unit = 100000000.0
				unit_find = 1
				#print("Unit ###1")
			# 백만원
			elif len(soup3.findAll("table")[table_num-1](string=re_unit2)) != 0:
				unit = 100.0
				unit_find = 1
				#print("Unit ###2")
			elif len(soup3.findAll("table")[table_num-1](string=re_unit3)) != 0:
				unit = 100000.0
				unit_find = 1
				#print("Unit ###3")

			if unit_find == 0:
				print ("UNIT NOT FOUND")
				if len(soup3.findAll(string=re_unit1)) != 0:
					print("Unit ###1")
					unit = 100000000.0
				elif len(soup3.findAll(string=re_unit2)) != 0:
					print("Unit ###2")
					unit = 100.0
				elif len(soup3.findAll(string=re_unit3)) != 0:
					print("Unit ###3")
					unit = 100000.0
			
			cashflow_sub_list = scrape_cashflows(cashflow_table, year, unit)
			income_statement_sub_list = scrape_income_statement(income_table, year, unit, 1)
			balance_sheet_sub_list = scrape_balance_sheet(balance_table, year, unit)
			
			cashflow_sub_list['net_income'] = income_statement_sub_list['net_income']

		## if(line_find != 0):
		else:
			print("FINDING LINE NUMBER ERROR")
			cashflow_sub_list = {}
			
			cashflow_sub_list['year']				= 2010
			cashflow_sub_list['op_cashflow']		= 0.0
			cashflow_sub_list['op_cashflow_sub1']	= "FINDING LINE NUMBER ERROR"
			cashflow_sub_list['op_cashflow_sub2']	= 0.0

			cashflow_sub_list['invest_cashflow']		= 0.0
			cashflow_sub_list['invest_cashflow_sub1']	= 0.0
			cashflow_sub_list['invest_cashflow_sub2'] 	= 0.0
			cashflow_sub_list['invest_cashflow_sub3'] 	= 0.0
			cashflow_sub_list['invest_cashflow_sub4'] 	= 0.0
			cashflow_sub_list['invest_cashflow_sub5'] 	= 0.0
			cashflow_sub_list['invest_cashflow_sub6'] 	= 0.0
			cashflow_sub_list['invest_cashflow_sub7'] 	= 0.0
			cashflow_sub_list['invest_cashflow_sub8'] 	= 0.0
			cashflow_sub_list['invest_cashflow_sub9'] 	= 0.0
			cashflow_sub_list['invest_cashflow_sub10']	= 0.0
			cashflow_sub_list['invest_cashflow_sub11'] 	= 0.0
			cashflow_sub_list['invest_cashflow_sub12'] 	= 0.0
			cashflow_sub_list['invest_cashflow_sub13'] 	= 0.0
			cashflow_sub_list['invest_cashflow_sub14'] 	= 0.0
			cashflow_sub_list['invest_cashflow_sub15'] 	= 0.0
			cashflow_sub_list['invest_cashflow_sub16'] 	= 0.0
			cashflow_sub_list['invest_cashflow_sub17'] 	= 0.0
			cashflow_sub_list['invest_cashflow_sub18'] 	= 0.0
			
			cashflow_sub_list['fin_cashflow']		= 0.0
			cashflow_sub_list['fin_cashflow_sub1']	= 0.0
			cashflow_sub_list['fin_cashflow_sub2'] 	= 0.0
			cashflow_sub_list['fin_cashflow_sub3'] 	= 0.0

			cashflow_sub_list['start_cash']		= 0.0
			cashflow_sub_list['end_cash']		= 0.0
			cashflow_sub_list['net_income']		= 0.0
			
			#print(cashflow_sub_list)

			balance_sheet_sub_list = {}
			balance_sheet_sub_list['year']						=	2010
			balance_sheet_sub_list["asset_current"]				=	0.0
			balance_sheet_sub_list["asset_current_sub1"]		=	"FINDING LINE NUMBER ERROR"
			balance_sheet_sub_list["asset_current_sub2"]		=	0.0
			balance_sheet_sub_list["asset_current_sub3"]		=	0.0
			balance_sheet_sub_list["asset_non_current"]			=	0.0
			balance_sheet_sub_list["asset_non_current_sub1"]	=	0.0
			balance_sheet_sub_list["asset_non_current_sub2"]	=	0.0
			balance_sheet_sub_list["asset_sum"]					=	0.0
			balance_sheet_sub_list["liability_current"]				=	0.0
			balance_sheet_sub_list["liability_current_sub1"]		=	0.0
			balance_sheet_sub_list["liability_current_sub2"]		=	0.0
			balance_sheet_sub_list["liability_current_sub3"]		=	0.0
			balance_sheet_sub_list["liability_non_current"]			=	0.0
			balance_sheet_sub_list["liability_non_current_sub1"]	=	0.0
			balance_sheet_sub_list["liability_non_current_sub2"]	=	0.0
			balance_sheet_sub_list["liability_non_current_sub3"]	=	0.0
			balance_sheet_sub_list["liability_non_current_sub4"]	=	0.0
			balance_sheet_sub_list["liability_sum"]					=	0.0
			balance_sheet_sub_list["equity"]						=	0.0
			balance_sheet_sub_list["equity_sub1"]					=	0.0
			balance_sheet_sub_list["equity_sub3"]					=	0.0
			balance_sheet_sub_list["equity_sub2"]					=	0.0
			balance_sheet_sub_list["equity_sum"]					=	0.0
					
			income_statement_sub_list = {}
			income_statement_sub_list['year']				=	2010
			income_statement_sub_list["sales"]				=	0.0
			income_statement_sub_list["sales_sub1"]			=	"FINDING LINE NUMBER ERROR"
			income_statement_sub_list["sales_sub2"]			=	0.0
			income_statement_sub_list["sales_sub3"]			=	0.0
			income_statement_sub_list["sales2"]				=	0.0
			income_statement_sub_list["sales2_sub1"]		=	0.0
			income_statement_sub_list["op_income"]		 	=	0.0
			income_statement_sub_list["op_income_sub1"]		=	0.0
			income_statement_sub_list["op_income_sub2"]		=	0.0
			income_statement_sub_list["op_income_sub3"]		=	0.0
			income_statement_sub_list["op_income_sub4"]		=	0.0
			income_statement_sub_list["op_income_sub5"]		=	0.0
			income_statement_sub_list["op_income_sub6"]		=	0.0
			income_statement_sub_list["op_income_sub7"]		=	0.0
			income_statement_sub_list["tax"]				=	0.0
			income_statement_sub_list["net_income"]			=	0.0
			income_statement_sub_list["eps"]				=	0.0

		dart_post_sublist.append(date)
		dart_post_sublist.append(corp_name)
		dart_post_sublist.append(market)
		dart_post_sublist.append(title)
		dart_post_sublist.append(link)
			
		dart_post_list.append(dart_post_sublist)
		cashflow_list.append(cashflow_sub_list)
		balance_sheet_list.append(balance_sheet_sub_list)
		income_statement_list.append(income_statement_sub_list)

	#handle = urllib.request.urlopen(url_templete % (report, urllib.parse.quote(corp)))
	#print("URL" + url_templete % (report, corp))
	## 선택
	handle = urllib.request.urlopen(url_templete % (report, urllib.parse.quote(corp), start_day.strftime('%Y%m%d'), end_day.strftime('%Y%m%d')))
	print("URL" + url_templete % (report, corp, start_day.strftime('%Y%m%d'), end_day.strftime('%Y%m%d')))

	data = handle.read()
	soup = BeautifulSoup(data, 'html.parser', from_encoding='utf-8')
	
	table = soup.find('table')
	trs = table.findAll('tr')
	tds = table.findAll('td')
	counts = len(tds)
	#print(counts)

	#if counts > 0:
	if counts > 2:
		# Delay operation
		time.sleep(20)
	
		link_list = []
		date_list = []
		corp_list = []
		market_list = []
		title_list = []
		reporter_list = []
		tr_cnt = 0

		for tr in trs[1:]:
			tr_cnt = tr_cnt + 1
			time.sleep(2)
			tds = tr.findAll('td')
			link = 'http://dart.fss.or.kr' + tds[2].a['href']
			date = tds[4].text.strip().replace('.', '-')
			corp_name = tds[1].text.strip()
			market = tds[1].img['title']

			old_title = title
			print("old", old_title)
			title = " ".join(tds[2].text.split())
			print("new",title)

			if old_title.find('정정')>-1:
				print("정정")
				continue
			reporter = tds[3].text.strip()

			re_pass = re.compile("해외증권거래소등에신고한사업보고서등의국내신고")
			if (not re_pass.search(title)):
				link_list.append(link)
				date_list.append(date)
				corp_list.append(corp_name)
				market_list.append(market)
				title_list.append(title)
				reporter_list.append(reporter)

				dart_post_sublist = []

				# year = int(date[0:4])
				year = title[-8:-1]
				print(year)
				print(corp_name)
				print(title)
				print(date)
				handle = urllib.request.urlopen(link)
				#print(link)
				data = handle.read()
				soup2 = BeautifulSoup(data, 'html.parser', from_encoding='utf-8')
				#print(soup2)
				
				#print(type(soup2.find('head').text))
				head_lines = soup2.find('head').text.split("\n")
				#print(head_words)

				# From 2015 ~ now
				#if (year>2014):
				#	re_tree_find = re.compile("2. 연결재무제표")
				## From 2010 to 2014
				#elif (year>2009):
				#	re_tree_find = re.compile("재무제표 등")
				## From 2008 to 2009
				#elif (year>2007):
				#	re_tree_find = re.compile("1. 연결재무제표에 관한 사항")
				## From 2002 to 2007
				#elif (year>2001):
				#	re_tree_find = re.compile("4. 재무제표")
				#else:
				#	re_tree_find = re.compile("3. 재무제표")

				re_tree_find1 = re.compile("2. 연결재무제표")
				re_tree_find2 = re.compile("재무제표 등")
				re_tree_find3 = re.compile("1. 연결재무제표에 관한 사항")
				re_tree_find4 = re.compile("4. 재무제표")
				re_tree_find5 = re.compile("3. 재무제표")
				
				re_tree_find1_bak = re.compile("4.[ ]*재무제표")
				
				line_num = 0
				line_find = 0
				for head_line in head_lines:
					if (re_tree_find1.search(head_line)):
						line_find = line_num
						break
						#print(head_line)
					elif (re_tree_find2.search(head_line)):
						line_find = line_num
						break
					elif (re_tree_find3.search(head_line)):
						line_find = line_num
						break
					elif (re_tree_find4.search(head_line)):
						line_find = line_num
						break
					elif (re_tree_find5.search(head_line)):
						line_find = line_num
						break
					line_num = line_num + 1

				line_num = 0
				line_find_bak = 0
				for head_line in head_lines:
					if (re_tree_find1_bak.search(head_line)):
						line_find_bak = line_num
						break
					line_num = line_num + 1


				if(line_find != 0):
		
					#print(head_lines[line_find])
					#print(head_lines[line_find+1])
					#print(head_lines[line_find+2])
					#print(head_lines[line_find+3])
					#print(head_lines[line_find+4])

					line_words = head_lines[line_find+4].split("'")
					#print(line_words)
					rcpNo = line_words[1]
					dcmNo = line_words[3]
					eleId = line_words[5]
					offset = line_words[7]
					length = line_words[9]

					#test = soup2.find('a', {'href' : '#download'})['onclick']
					#words = test.split("'")
					#rcpNo = words[1]
					#dcmNo = words[3]
					
					dart = soup2.find_all(string=re.compile('dart.dtd'))
					dart2 = soup2.find_all(string=re.compile('dart2.dtd'))
					dart3 = soup2.find_all(string=re.compile('dart3.xsd'))

					if len(dart3) != 0:
						link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=" + eleId + "&offset=" + offset + "&length=" + length + "&dtd=dart3.xsd"
					elif len(dart2) != 0:
						link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=" + eleId + "&offset=" + offset + "&length=" + length + "&dtd=dart2.dtd"
					elif len(dart) != 0:
						link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=" + eleId + "&offset=" + offset + "&length=" + length + "&dtd=dart.dtd"
					else:
						link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=0&offset=0&length=0&dtd=HTML"  
					
					print(link2)

					#try:
					handle = urllib.request.urlopen(link2)
					#print(handle)
					data = handle.read()
					soup3 = BeautifulSoup(data, 'html.parser', from_encoding='utf-8')
					#print(soup3)

					tables = soup3.findAll("table")
			
					# 2. 연결재무제표가 비어 있는 경우
					if (len(tables) == 0):
						line_words = head_lines[line_find_bak+4].split("'")
						#print(line_words)
						rcpNo = line_words[1]
						dcmNo = line_words[3]
						eleId = line_words[5]
						offset = line_words[7]
						length = line_words[9]

						dart = soup2.find_all(string=re.compile('dart.dtd'))
						dart2 = soup2.find_all(string=re.compile('dart2.dtd'))
						dart3 = soup2.find_all(string=re.compile('dart3.xsd'))

						if len(dart3) != 0:
							link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=" + eleId + "&offset=" + offset + "&length=" + length + "&dtd=dart3.xsd"
						elif len(dart2) != 0:
							link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=" + eleId + "&offset=" + offset + "&length=" + length + "&dtd=dart2.dtd"
						elif len(dart) != 0:
							link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=" + eleId + "&offset=" + offset + "&length=" + length + "&dtd=dart.dtd"
						else:
							link2 = "http://dart.fss.or.kr/report/viewer.do?rcpNo=" + rcpNo + "&dcmNo=" + dcmNo + "&eleId=0&offset=0&length=0&dtd=HTML"  
						
						print(link2)
						
						handle = urllib.request.urlopen(link2)
						print(handle)
						data = handle.read()
						soup3 = BeautifulSoup(data, 'html.parser', from_encoding='utf-8')
						tables = soup3.findAll("table")


					cnt = 0
					table_num = 0

					for table in tables:
						if (re_cashflow_find.search(table.text)):
							table_num = cnt
							break
						cnt = cnt + 1
					
					print("table_num", table_num, "Tables", len(tables))
					cashflow_table = soup3.findAll("table")[table_num]
					
					trs = cashflow_table.findAll("tr")
					
					cnt = 0
					table_income_num = 0
					for table in tables:
						if (re_income_find.search(table.text)):
							table_income_num = cnt
							break
						cnt = cnt + 1
					income_table = soup3.findAll("table")[table_income_num]
					print("table_income_num", table_income_num, "Tables", len(tables))
					
					cnt = 0
					table_balance_num = 0
					for table in tables:
						if (re_balance_sheet_find.search(table.text)):
							table_balance_num = cnt
							break
						cnt = cnt + 1
					balance_table = soup3.findAll("table")[table_balance_num]
					print("table_balance_num", table_balance_num, "Tables", len(tables))
			
					unit = 100.0
					unit_find = 0
					re_unit1 = re.compile('단위[ \s]*:[ \s]*원')
					re_unit2 = re.compile('단위[ \s]*:[ \s]*백만원')
					re_unit3 = re.compile('단위[ \s]*:[ \s]*천원')

					# 원
					if len(soup3.findAll("table")[table_num-1](string=re_unit1)) != 0:
						unit = 100000000.0
						unit_find = 1
						#print("Unit ###1")
					# 백만원
					elif len(soup3.findAll("table")[table_num-1](string=re_unit2)) != 0:
						unit = 100.0
						unit_find = 1
						#print("Unit ###2")
					elif len(soup3.findAll("table")[table_num-1](string=re_unit3)) != 0:
						unit = 100000.0
						unit_find = 1
						#print("Unit ###3")

					if unit_find == 0:
						print ("UNIT NOT FOUND")
						if len(soup3.findAll(string=re_unit1)) != 0:
							print("Unit ###1")
							unit = 100000000.0
						elif len(soup3.findAll(string=re_unit2)) != 0:
							print("Unit ###2")
							unit = 100.0
						elif len(soup3.findAll(string=re_unit3)) != 0:
							print("Unit ###3")
							unit = 100000.0
			
					## 원
					#if len(soup3.findAll("table")[table_num-1](string=re.compile('단위[ ]*:[ ]*원'))) != 0:
					#	unit = 100000000.0
					## 백만원
					#elif len(soup3.findAll("table")[table_num-1](string=re.compile('단위[ ]*:[ ]*백만원'))) != 0:
					#	unit = 100.0
					#elif len(soup3.findAll("table")[table_num-1](string=re.compile('단위[ ]*:[ ]*천원'))) != 0:
					#	unit = 100000.0
				
					# Scrape data
					cashflow_sub_list = scrape_cashflows(cashflow_table, title[-8:-1], unit) ##title[-8:-1]->year-1
					income_statement_sub_list = scrape_income_statement(income_table, title[-8:-1], unit, 0)
					balance_sheet_sub_list = scrape_balance_sheet(balance_table, title[-8:-1], unit)
					# print("cashflow_sheet \t",cashflow_sub_list)
					
					cashflow_sub_list['net_income'] = income_statement_sub_list['net_income']

				## if(line_find != 0):
				else:
					print("FINDING LINE NUMBER ERROR")
					cashflow_sub_list = {}
					
					cashflow_sub_list['year']				= title[-8:-1]
					cashflow_sub_list['op_cashflow']		= 0.0
					cashflow_sub_list['op_cashflow_sub1']	= "FINDING LINE NUMBER ERROR"
					cashflow_sub_list['op_cashflow_sub2']	= 0.0

					cashflow_sub_list['invest_cashflow']		= 0.0
					cashflow_sub_list['invest_cashflow_sub1']	= 0.0
					cashflow_sub_list['invest_cashflow_sub2'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub3'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub4'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub5'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub6'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub7'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub8'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub9'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub10']	= 0.0
					cashflow_sub_list['invest_cashflow_sub11'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub12'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub13'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub14'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub15'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub16'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub17'] 	= 0.0
					cashflow_sub_list['invest_cashflow_sub18'] 	= 0.0
					
					cashflow_sub_list['fin_cashflow']		= 0.0
					cashflow_sub_list['fin_cashflow_sub1']	= 0.0
					cashflow_sub_list['fin_cashflow_sub2'] 	= 0.0
					cashflow_sub_list['fin_cashflow_sub3'] 	= 0.0

					cashflow_sub_list['start_cash']		= 0.0
					cashflow_sub_list['end_cash']		= 0.0
					cashflow_sub_list['net_income']		= 0.0
			
					balance_sheet_sub_list = {}
					balance_sheet_sub_list['year']						=	title[-8:-1]
					balance_sheet_sub_list["asset_current"]				=	0.0
					balance_sheet_sub_list["asset_current_sub1"]		=	"FINDING LINE NUMBER ERROR"
					balance_sheet_sub_list["asset_current_sub2"]		=	0.0
					balance_sheet_sub_list["asset_current_sub3"]		=	0.0
					balance_sheet_sub_list["asset_non_current"]			=	0.0
					balance_sheet_sub_list["asset_non_current_sub1"]	=	0.0
					balance_sheet_sub_list["asset_non_current_sub2"]	=	0.0
					balance_sheet_sub_list["asset_sum"]					=	0.0
					balance_sheet_sub_list["liability_current"]				=	0.0
					balance_sheet_sub_list["liability_current_sub1"]		=	0.0
					balance_sheet_sub_list["liability_current_sub2"]		=	0.0
					balance_sheet_sub_list["liability_current_sub3"]		=	0.0
					balance_sheet_sub_list["liability_non_current"]			=	0.0
					balance_sheet_sub_list["liability_non_current_sub1"]	=	0.0
					balance_sheet_sub_list["liability_non_current_sub2"]	=	0.0
					balance_sheet_sub_list["liability_non_current_sub3"]	=	0.0
					balance_sheet_sub_list["liability_non_current_sub4"]	=	0.0
					balance_sheet_sub_list["liability_sum"]					=	0.0
					balance_sheet_sub_list["equity"]						=	0.0
					balance_sheet_sub_list["equity_sub1"]					=	0.0
					balance_sheet_sub_list["equity_sub3"]					=	0.0
					balance_sheet_sub_list["equity_sub2"]					=	0.0
					balance_sheet_sub_list["equity_sum"]					=	0.0

					income_statement_sub_list = {}
					income_statement_sub_list["year"]				=	title[-8:-1]
					income_statement_sub_list["sales"]				=	0.0
					income_statement_sub_list["sales_sub1"]			=	"FINDING LINE NUMBER ERROR"
					income_statement_sub_list["sales_sub2"]			=	0.0
					income_statement_sub_list["sales_sub3"]			=	0.0
					income_statement_sub_list["sales2"]				=	0.0
					income_statement_sub_list["sales2_sub1"]		=	0.0
					income_statement_sub_list["op_income"]		 	=	0.0
					income_statement_sub_list["op_income_sub1"]		=	0.0
					income_statement_sub_list["op_income_sub2"]		=	0.0
					income_statement_sub_list["op_income_sub3"]		=	0.0
					income_statement_sub_list["op_income_sub4"]		=	0.0
					income_statement_sub_list["op_income_sub5"]		=	0.0
					income_statement_sub_list["op_income_sub6"]		=	0.0
					income_statement_sub_list["op_income_sub7"]		=	0.0
					income_statement_sub_list["tax"]				=	0.0
					income_statement_sub_list["net_income"]			=	0.0
					income_statement_sub_list["eps"]				=	0.0

				dart_post_sublist.append(date)
				dart_post_sublist.append(corp_name)
				dart_post_sublist.append(market)
				dart_post_sublist.append(title)
				dart_post_sublist.append(link)
				
				dart_post_list.append(dart_post_sublist)
				cashflow_list.append(cashflow_sub_list)
				balance_sheet_list.append(balance_sheet_sub_list)
				income_statement_list.append(income_statement_sub_list)

	write_excel_file(workbook_name, dart_post_list, cashflow_list, balance_sheet_list, income_statement_list, corp, stock_code, stock_cat)

# Main
if __name__ == "__main__":
	main()


############## rebalancing
"""
19.05.13
종목과 비중 리스트 만들어서
지정된 시기(주단위)부터 누적 수익률을 plotly로 그래프 그리기

지금 알고리즘 구성
0. 종목 검색
1. 네이버에서 해당 기간의 페이지에서 모든 가격과 날짜를 다 불러옴
2. 투자본금에 비례해서 매월 첫날에 매매
3. 누적 주식수, 누적 투자금, 누적 평가금 계산, 누적 손익률
4. 종목 별 누적 주식수, 누적 투자금, 누적 평가금으로 총 투자금, 총 평가금, 총 누적 손익률을 구함

19.05.25
리밸런싱 가능하려면
1. 날짜별 가격 loop
2. 첫달은 지정한 비중대로 종목 매매 -> 종목별 종목수, 평가금, 투자금
3. 두번째 달 부터 총 평가금 대비 종목 평가금으로 비중 계산
4. 현재가 * 주식수 / 평가금이 지정 비중보다 크거나 작으면 몇 주를 매매할지 계산
3. 종목별 비중 오차가 큰 종목 순으로 정렬 -> 0.1 이상 차이가 나게 되면 비중 리밸런싱
done


앞으로할 것
0. 인덱스 값 받아오기 : done
1. 종목과 비중을 dict으로 받아오기 : done
2. 포트폴리오 저장명 지정 : done
3. 현재는 최초 투자 후 변동없이 가지고 가는 형태인데, 만약 자금이 지속적으로 적립된다면 어떻게 되는지에 대한 것도 구현 : done
4. 비중 대신 초기 자금? done
5. 비교 종목을 input 써서 검색할 수 있게 가능?
6. 기간이아니라 시작 날짜와 끝 날짜로 데이터 크롤링 : done
7. 미국 주식도 크롤링 가능하게
8. 리밸런싱 가능하게: done
9. pykrx 이거 정리해서 폴더로 만들기
"""
import pandas as pd
import numpy as np
import os
from urllib.request import urlopen
import bs4
import datetime
import plotly.offline as offline
import plotly.graph_objs as go
import pdb
from tqdm import tqdm
from pykrx import stock as STOCK
from pykrx import e3
from pykrx import bond
import yfinance as yf
import pandas_datareader

class Portfolio():
  def __init__(self,code={'삼성전자':1.0},start_date='',end_date='',
               init_amount = 0, monthly_amount = 0, new=True, save='my_port',rebalancing=True):
    """
    종목명을 잘 몰라 에러 뜰 때 확인
    '삼화' 라는 단어가 포함된 종목을 모두 검색
    self.code_df["Indexes"] = self.code_df.name.str.find(input("검색할 종목명을 입력하세요: \n"))
    print(self.code_df.loc[self.code_df.Indexes > -1])

    """
    self.save = save
    if new:
      self.code_list = [*code.keys()]
      self.num_stocks = len(code.keys())
      self.share_weight = [cdt[0] for cdt in code.values()]
      self.cat = [cdt[1] for cdt in code.values()]

      self.start_date = self.date_format(start_date).strftime("%Y%m%d")
      if end_date:
        self.end_date = self.date_format(end_date).strftime("%Y%m%d")
      else:
        self.end_date = datetime.date.today().strftime("%Y%m%d")
      ##
      if os.path.isfile("tickers.pk"):
        self.tickers = pd.read_pickle("tickers.pk")
        print("load url pickle")
      else:
        tickers_code = STOCK.get_market_ticker_code_list()
        tickers_etf = e3.get_etf_ticker_list()
        tickers = tickers_code['종목'].append(tickers_etf)
        self.tickers = pd.DataFrame(tickers.values, index=tickers.index, columns=['종목'])
        self.tickers['시장'] = tickers_code['시장']
        self.tickers['시장'] = self.tickers['시장'].replace(np.nan, 'ETF')
        self.tickers.to_pickle("tickers.pk")
        self.tickers.to_csv("tickers.csv", encoding='ms949')

      self.init_amount = init_amount
      self.monthly_amount = monthly_amount
      self.rebalancing = rebalancing
      self.amount_ratio = self.monthly_amount / self.init_amount
      self.add_stock()
      # self.df = self.df.sort_index(ascending=False)
      self.add_index()

      self.save_df(save)
    else:
      self.df = self.load_df(save)
  def get_url(self,item_name):
    # code = self.code_df.query("name=='{}'".format(item_name))['code'].to_string(index=False)
    code = self.tickers[self.tickers == '{}'.format(item_name)].index[0]
    url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=code)
    # print("요청 URL = {}".format(url))
    return url

  def get_index_url(self,code):
    url = 'https://finance.naver.com/sise/sise_index_day.nhn?code={code}'.format(code=code)

    return url

  def date_format(self,d):
    d = str(d).replace('-', '.')
    yyyy = int(d.split('.')[0])
    mm = int(d.split('.')[1])
    dd = int(d.split('.')[2])
    this_date = datetime.date(yyyy, mm, dd)
    return this_date

  # 네이버에서 일자별 인덱스를 추출하는 함수 정의
  def historical_index_naver(self,index_cd, start_date='', end_date='', historical_prices=dict(), page_n=1, last_page=0):
    if start_date:  # start_date가 있으면
      start_date = self.date_format(start_date)  # date 포맷으로 변환
    else:  # 없으면
      start_date = datetime.date.today()  # 오늘 날짜를 지정
    if end_date:
      end_date = self.date_format(end_date)
    else:
      end_date = datetime.date.today()
    naver_index = self.get_index_url(index_cd) + '&page={page_n}'.format(page_n=str(page_n))
    # naver_index = 'http://finance.naver.com/sise/sise_index_day.nhn?code=' + index_cd + '&page=' + str(page_n)
    source = urlopen(naver_index).read()  # 지정한 페이지에서 코드 읽기
    source = bs4.BeautifulSoup(source, 'lxml')  # 뷰티풀 스프로 태그별로 코드 분류
    dates = source.find_all('td', class_='date')  # <td class="date">태그에서 날짜 수집
    prices = source.find_all('td', class_='number_1')  # <td class="number_1">태그에서 지수 수집

    for n in range(len(dates)):
      if dates[n].text.split('.')[0].isdigit():
        # 날짜 처리
        this_date = dates[n].text
        this_date = self.date_format(this_date)
        if this_date <= end_date and this_date >= start_date:
          # start_date와 end_date 사이에서 데이터 저장
          # 종가 처리
          this_close = prices[n * (len(prices)//len(dates))].text  # prices 중 종가지수인 0,4,8,...번째 데이터 추출
          this_close = this_close.replace(',', '')
          this_close = float(this_close)
          # 딕셔너리에 저장
          historical_prices[this_date] = this_close

        elif this_date < start_date:
          # start_date 이전이면 함수 종료
          return historical_prices
    # 페이지 네비게이션
    if last_page == 0:
      last_page = source.find('td', class_='pgRR').find('a')['href']
      # 마지막페이지 주소 추출
      last_page = last_page.split('&')[1]  # & 뒤의 page=506 부분 추출
      last_page = last_page.split('=')[1]  # = 뒤의 페이지번호만 추출
      last_page = int(last_page)  # 숫자형 변수로 변환

    # 다음 페이지 호출
    if page_n < last_page:
      page_n = page_n + 1
      self.historical_index_naver(index_cd, start_date, end_date,historical_prices, page_n, last_page)
    return historical_prices

  def historical_stock_naver(self,stock_cd, start_date='', end_date='', historical_st_prices=dict(), page_n=1, last_page=0):

    if start_date:  # start_date가 있으면
      start_date = self.date_format(start_date)  # date 포맷으로 변환
    else:  # 없으면
      start_date = datetime.date.today()  # 오늘 날짜를 지정
    if end_date:
      end_date = self.date_format(end_date)
    else:
      end_date = datetime.date.today()

    naver_stock = self.get_url(stock_cd)+'&page={page_n}'.format(page_n=str(page_n))
    pr_source = urlopen(naver_stock).read()  # 지정한 페이지에서 코드 읽기
    pr_source = bs4.BeautifulSoup(pr_source, 'lxml')  # 뷰티풀 스프로 태그별로 코드 분류

    dates = pr_source.find_all("td", align="center")  # <td align="center">태그에서 날짜 수집
    prices = pr_source.find_all('td', class_="num")  # <td class="num">태그에서 지수 수집

    for n in range(len(dates)):
      if dates[n].text.split('.')[0].isdigit():
        # 날짜 처리
        this_date = dates[n].text
        this_date = self.date_format(this_date)

        if this_date <= end_date and this_date >= start_date:
          # start_date와 end_date 사이에서 데이터 저장
          # 종가 처리

          this_close = prices[n * (len(prices)//len(dates))].text  # prices 중 종가지수인 0,4,8,...번째 데이터 추출
          this_close = this_close.replace(',', '')
          this_close = float(this_close)

          # 딕셔너리에 저장
          historical_st_prices[this_date] = this_close

        elif this_date < start_date:
          # start_date 이전이면 함수 종료
          return historical_st_prices

    # 페이지 네비게이션
    if last_page == 0:
      last_page = pr_source.find('td', class_='pgRR').find('a')['href']
      # 마지막페이지 주소 추출
      last_page = last_page.split('&')[1]  # & 뒤의 page=506 부분 추출
      last_page = last_page.split('=')[1]  # = 뒤의 페이지번호만 추출
      last_page = int(last_page)  # 숫자형 변수로 변환


    # 다음 페이지 호출
    if page_n < last_page:
      page_n = page_n + 1
      if (this_date - end_date).days> 40:
        page_n = (this_date - end_date).days*2//3 // len(dates)
      self.historical_stock_naver(stock_cd, start_date, end_date, historical_st_prices, page_n, last_page)
    return historical_st_prices

  def add_stock(self):

    i=0
    columns_name = []

    income_name = []
    monthly_prices = dict()
    ###
    no_internet = False
    if no_internet :
      self.prev_df = self.load_df(self.save)
      for stockItem in tqdm(self.code_list):
        if self.cat[i] == 'ko':
          prices = self.prev_df[stockItem]
        elif self.cat[i] == 'etf':
          print(stockItem, 'etf')
          prices = self.prev_df[stockItem]
        if i == 0:
          self.df = pd.DataFrame(prices.values, index=prices.index, columns=[stockItem])
          # 첫달부터 정렬
          self.df = self.df.sort_index(ascending=True)
          months = [m[4:6] for m in self.df.index]

          # 첫달 인덱스 추가
          month_index = [0]
          for m in range(len(months) - 1):
            if months[m] != months[m + 1]:
              # 달이 바뀌는 마지막 날 인덱스
              month_index.append(m)

        else:
          self.df[stockItem] = prices.values
        self.df[stockItem + '_주식수'] = 0
        self.df[stockItem + '_평균단가'] = 0
        self.df[stockItem + '_투자금액'] = 0
        self.df[stockItem + '_평가액'] = 0
        # 첫 달의 가격들
        monthly_prices[stockItem] = self.df[stockItem].iloc[month_index].values
        i += 1
    else:
      for stockItem in tqdm(self.code_list):
        if self.cat[i] == 'ko':
          prices = STOCK.get_market_c_by_date(self.start_date, self.end_date, self.tickers['종목'][self.tickers['종목']==stockItem].index[0],'d')
        elif self.cat[i] =='etf':
          print(stockItem,'etf')
          prices = e3.get_etf_ohlcv_by_date(self.start_date, self.end_date, stockItem, 'd')['종가']
        if i ==0:
          self.df = pd.DataFrame(prices.values, index=prices.index,columns=[stockItem])
          # 첫달부터 정렬
          self.df = self.df.sort_index(ascending=True)
          months = [m[4:6] for m in self.df.index]

          # 첫달 인덱스 추가
          month_index = [0]
          for m in range(len(months) - 1):
            if months[m] != months[m + 1]:
              # 달이 바뀌는 마지막 날 인덱스
              month_index.append(m)

        else:
          self.df[stockItem] = prices.values
        self.df[stockItem + '_주식수'] = 0
        self.df[stockItem + '_평균단가'] = 0
        self.df[stockItem + '_투자금액'] = 0
        self.df[stockItem + '_평가액'] = 0
        # 첫 달의 가격들
        monthly_prices[stockItem] = self.df[stockItem].iloc[month_index].values
        i+=1


    # 구간 설정을 위해서 오늘 날짜 인덱스를 제일마지막에 추가
    month_index.append(len(self.df.index)-1)

    # 종목별 첫달 주식수
    stocks = dict()
    # 종목별 첫달 평가액
    # 종목별 청달 투자액
    avg_prices = dict()
    # stocks[code].append(stock)
    # monthly_values[code].append(monthly_payment)
    revenue = 0

    total_captial = []
    for idx in range(len(month_index) - 1):

      if idx == 0:
        # 초기 투자 주식수
        tot_cap = 0
        net_amount = self.init_amount
        for i, cd in enumerate(self.code_list):
          stck = (self.share_weight[i] * net_amount) // monthly_prices[cd][idx]  # (비중x 적립금)// 종목별 첫달 가격 = 주식수
          monthly_payment = stck * monthly_prices[cd][idx]  # 주식수 * 종목별 첫달 가격
          tot_cap += monthly_payment

          stocks[cd] = [stck] # 초기 주식수
          avg_prices[cd] = [monthly_prices[cd][idx]] # 초기 평균단가
        total_captial.append(tot_cap)
        net_amount -= tot_cap
        print("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t초기 자본으로 남은돈 {}".format(net_amount))
      else:
        # 이번달 종목별 평가금
        this_month_value = list()

        for code in self.code_list:
          this_month_value.append(stocks[code][idx-1] * monthly_prices[code][idx]) # 주식수 * 현재가
          # print("{}의 {}주 평가금액: {}".format(code,stocks[code][idx-1],stocks[code][idx-1] * monthly_prices[code][idx]))
        # 이번달 종목 비중
        month_ratio = this_month_value[:]/sum(this_month_value) # 평가금/ 총액
        month_ratio_err = self.share_weight - month_ratio
        # print("\n \t\t\t 매집전 {}th번째 비중 {}/{}\n".format(idx,month_ratio,sum(month_ratio)))
        ################################################### 리밸런싱 #####################################################

        net_amount += self.monthly_amount
        if self.rebalancing:
          print("\n############rebalancing##############")
          tot_cap = 0
          tot_amnt = 0
          for sorted, ratio_idx in enumerate(np.argsort(month_ratio_err)):
            # 현재 주식수 - 총액 * 종목별 비중 / 현재가 = 팔아야할 주식 수 (-면 사야함)

            num_sell_or_buy = int(stocks[self.code_list[ratio_idx]][idx - 1]
                                  - (sum(this_month_value) * self.share_weight[ratio_idx] / monthly_prices[self.code_list[ratio_idx]][idx]))

            num_buy = 0
            num_sell = 0
            if num_sell_or_buy > 0:
              num_sell = num_sell_or_buy
              revenue += num_sell * monthly_prices[self.code_list[ratio_idx]][idx]*0.967 # 수익금 = 주식수 * 현재가 * 수수료(3.3%)

            else:

              if revenue > 0:  # 월 적릭금이 기준가보다 많으면 리밸런싱 추매
                # 매수해야할 갯수
                num_buy = -num_sell_or_buy
                # 여윳돈으로 살 수 있는 최대 수
                max_buy = int(revenue) // monthly_prices[self.code_list[ratio_idx]][idx]
                if max_buy >= num_buy:  # 사야할 갯수가 맥스 구매보다 작으면 그대로
                  num_buy = num_buy
                else:  # 사야할 갯수가 너무 많은데 돈없으면 최대 살 수 있을 만큼만
                  num_buy = max_buy
                # 매달 자본금에서 산만큼 빼기
                revenue -= num_buy * monthly_prices[self.code_list[ratio_idx]][idx]


            net_num = int(num_buy)-int(num_sell)
            # 이번 달에 예산으로 산 주식 수
            tot_cap += self.share_weight[ratio_idx] * self.monthly_amount // monthly_prices[self.code_list[ratio_idx]][idx] * monthly_prices[self.code_list[ratio_idx]][idx]
            stock = int((self.share_weight[ratio_idx] * net_amount) // monthly_prices[self.code_list[ratio_idx]][idx])

            # net_cap = stock * monthly_prices[self.code_list[ratio_idx]][idx]
            # 총 증감한 주식수
            net_stock = stock + net_num # 주식수 변동량
            # print("stock: {} net_num : {}".format(stock, net_num))
            total_stock = stocks[self.code_list[ratio_idx]][idx - 1] + net_stock # 전체 주식수
            prev_captial = avg_prices[self.code_list[ratio_idx]][idx - 1] * stocks[self.code_list[ratio_idx]][idx - 1]
            if net_stock > 0: # 평균단가 = (이전 평단 * 이전 주식수 + 새로산 주식* 현재가)/ 전체주식수
              print("{}. {} {}주 사고, revenue is {}".format(sorted, self.code_list[ratio_idx], net_stock, revenue))
              monthly_payment = net_stock * monthly_prices[self.code_list[ratio_idx]][idx]  # 새로산 주식수 * 현재가
              avg_prc =  (prev_captial+monthly_payment)/total_stock
            else: # 투자금액은? 이전 평단 * 현재 주식수
              print("{}. {} {}주 팔고, revenue is {}".format(sorted, self.code_list[ratio_idx], -net_stock, revenue))
              monthly_payment = net_stock * monthly_prices[self.code_list[ratio_idx]][idx]  # 예산으로으로 산 주식수 * 현재가
              avg_prc = avg_prices[self.code_list[ratio_idx]][idx - 1]
            tot_amnt += (stock * monthly_prices[self.code_list[ratio_idx]][idx])
            stocks[self.code_list[ratio_idx]].append(total_stock)  # 주식수
            avg_prices[self.code_list[ratio_idx]].append(avg_prc)  # 평균단가
          total_captial.append(total_captial[-1]+tot_cap)
          print("\t\t\t\t\t\t\t\t{} 에서 {} 만큼 쓰고 {} 만큼 수익 --> 토탈 {}"
                .format(int(net_amount),int(tot_amnt),int(revenue),int(net_amount-tot_cap+revenue)))

          net_amount = net_amount - tot_amnt + revenue
          if net_amount<0:
            assert net_amount>0, "minus Net_Amount"
          revenue = 0
          print("###################################\n")
        else:
          tot_cap = 0

          for i, code in enumerate(self.code_list):

            stock = int((self.share_weight[i] * net_amount) // monthly_prices[code][idx])
            tot_cap += stock * monthly_prices[code][idx]

            # 총 증감한 주식수
            net_stock = stock
            total_stock = stocks[code][idx - 1] + net_stock  # 전체 주식수
            prev_captial = avg_prices[code][idx - 1] * stocks[code][idx - 1]
            monthly_payment = net_stock * monthly_prices[code][idx]  # 새로산 주식수 * 현재가
            avg_prc = (prev_captial + monthly_payment) / total_stock
            # 이번달 투자액
            stocks[code].append(total_stock)  # 주식수
            avg_prices[code].append(avg_prc)  # 평균단가
          total_captial.append(total_captial[-1]+tot_cap)
          net_amount -= tot_cap



      print("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t남은 수익금 : {}".format(revenue))
      # 이번달 종목별 평가금
      check = list()
      for cdd in self.code_list:
        check.append(stocks[cdd][idx] * monthly_prices[cdd][idx]) # 주식수 * 현재가
        print("{}의 {}주 평가금액: {}".format(cdd,stocks[cdd][idx], stocks[cdd][idx] * monthly_prices[cdd][idx]))
      # 이번달 종목 비중
      month_ratio = check[:]/sum(check) # 평가금/ 총액
      print("매집후 {}th 비중 {}/{}\n\n".format(idx,month_ratio,sum(month_ratio)))

    # 구간 설정을 위해서 첫번째 달 인덱스를 -1 으로하고 제일 마지막 인덱스 추가
    month_index[0] = month_index[0] -1
    self.df['총투자액'] = 0
    for fst, stockItem in enumerate(self.code_list):
      self.df[stockItem + '_주식수'] = 0
      self.df[stockItem + '_평균단가'] = 0
      self.df[stockItem + '_투자금액'] = 0
      for s in range(len(month_index) - 1):
        if fst == 0:
          self.df['총투자액'].iloc[np.arange(month_index[s] + 1, month_index[s + 1] + 1, 1)] = total_captial[s]

        self.df[stockItem + '_주식수'].iloc[np.arange(month_index[s] + 1, month_index[s + 1] + 1, 1)] = stocks[stockItem][s]
        self.df[stockItem + '_평균단가'].iloc[np.arange(month_index[s] + 1, month_index[s + 1] + 1, 1)] = \
          avg_prices[stockItem][s]
        self.df[stockItem + '_투자금액'].iloc[np.arange(month_index[s] + 1, month_index[s + 1] + 1, 1)] = \
          avg_prices[stockItem][s]*stocks[stockItem][s]
      self.df[stockItem + '_평가액'] = self.df[stockItem] * self.df[stockItem + '_주식수']
      # 누적 손익률로 표시
      self.df[stockItem + '_누적손익'] = round(
        (self.df[stockItem + '_평가액'] - self.df[stockItem + '_투자금액']) / self.df[stockItem + '_투자금액'] * 100, 3)
      # 종목별 누적손익 column 이름 저장
      columns_name.append(stockItem + '_누적손익')
      income_name.append(stockItem + '_평가액')

    # 비중을 곱해서 포트폴리오 전체 수익률 계산
    self.df['총평가액'] = self.df[income_name].sum(axis=1)
    self.df['total'] = round(
      (self.df['총평가액'] - self.df['총투자액']) / self.df['총투자액'] * 100, 3)

  def add_index(self):
    i = 0
    for stockItem in ['코스피', '코스닥']:
      # indexes = self.historical_index_naver(index_cd=stockItem, start_date=self.start_date, end_date=self.end_date)

      if stockItem == '코스피':
        indexes = STOCK.get_index_kospi_ohlcv_by_date(self.start_date, self.end_date,stockItem)['종가']
      else:
        indexes = STOCK.get_index_kosdaq_ohlcv_by_date(self.start_date, self.end_date, stockItem)['종가']


      months = months = [m[4:6] for m in self.df.index]
      month_index = []
      # 첫달 인덱스 추가
      month_index = [0]
      for m in range(len(months) - 1):
        if months[m] != months[m + 1]:
          # 달이 바뀌는 마지막 날 인덱스
          month_index.append(m)

      self.df[stockItem] = indexes.values
      self.df[stockItem + '_주식수'] = 0
      self.df[stockItem + '_평균단가'] = 0

      # 첫 달의 가격들
      monthly_prices = self.df[stockItem].iloc[month_index].values
      # 매달 투자하는 주식수
      stock =  self.amount_ratio / monthly_prices[1:]
      # int 형으로 변환
      stock = stock.tolist()
      # 초기 투자 주식수
      stock.insert(0,1 / monthly_prices[0])

      # 투입한 금액 총합
      monthly_payment = stock * monthly_prices
      monthly_payment = monthly_payment.tolist()
      # 주식수와 투입 자본 누적하기
      # stock = stock
      # monthly_payment = monthly_payment
      for l in range(len(stock) - 1):
        stock[l + 1] = stock[l] + stock[l + 1]
        monthly_payment[l + 1] = monthly_payment[l] + monthly_payment[l + 1]

      # stock = stock[::-1]
      # monthly_payment = monthly_payment[::-1]

      # 구간 설정을 위해서 첫번째 달 인덱스에 -1 추가
      month_index[0] = month_index[0] -1
      # 구간 설정을 위해서 오늘 날짜 인덱스를 제일마지막에 추가
      month_index.append(len(self.df.index) - 1)
      for s in range(len(month_index) - 1):
        self.df[stockItem + '_주식수'].iloc[np.arange(month_index[s]+1, month_index[s + 1] + 1, 1)] = stock[s]
        self.df[stockItem + '_평균단가'].iloc[np.arange(month_index[s]+1, month_index[s + 1] + 1, 1)] = monthly_payment[s]

      self.df[stockItem + '_평가액'] = self.df[stockItem] * self.df[stockItem + '_주식수']
      # 누적 손익률로 표시
      self.df[stockItem + '_누적손익'] = round(
        (self.df[stockItem + '_평가액'] - self.df[stockItem + '_평균단가']) / self.df[stockItem + '_평균단가'] * 100, 3)

      i += 1

  def save_df(self,save):
    self.df.to_pickle(str(save) + '.pickle')
    self.df.to_csv(str(save) + '.csv',encoding='ms949')
  def load_df(self,save):
    return pd.read_pickle(str(save) + '.pickle')

init_amount = 20000000
monthly_amount = 2000000


code = dict(
  고영=[0.1,'ko'],
  삼화콘덴서=[0.1,'ko'],
  티씨케이=[0.1,'ko'],
  리노공업=[0.1,'ko'],
  포스코케미칼=[0.1,'ko'],
  # 민앤지=[0.1,'ko'],
  피에스텍=[0.1,'ko'],
  # 비즈니스온=[0.1,'ko'],
  한국경제TV=[0.1,'ko'],
  해마로푸드서비스=[0.1,'ko'],
)
code["TIGER 미국나스닥100"] = [0.2,'etf']
# code["KODEX 국채선물10년"] = [0.4,'etf']

rebalancing = True
port_name = 'my_port_rebalance_only_stock_{}'.format(str(rebalancing))
my_port = Portfolio(code=code ,start_date='2016-01-02',end_date='2019-05-28',
                    init_amount = init_amount, monthly_amount = monthly_amount, new=True, save=port_name, rebalancing=rebalancing)
offline.init_notebook_mode(connected=True)
data = []

datetime_index = [dt[0:4]+'-'+dt[4:6]+'-'+dt[6:] for dt in my_port.df.index]

for key in code.keys():
  data.append(go.Scatter(x=datetime_index, y=my_port.df[key+'_누적손익'], name='{}'.format(key), line=dict(width=1)))


trace1 = go.Scatter( x=datetime_index, y=my_port.df.코스피_누적손익, name='KOSPI', line=dict(width=5))
trace2 = go.Scatter( x=datetime_index, y=my_port.df.코스닥_누적손익, name='KOSDAQ', line=dict(width=5))
trace3 = go.Scatter( x=datetime_index, y=my_port.df.total, name='포트폴리오', line=dict(width=5))
data += [trace1,trace2,trace3]

layout = dict(
              title="포트폴리오 백테스트_{}".format(port_name),
              xaxis=dict
                (
                    rangeselector=dict(
                      buttons=list([
                        # dict(count=1,
                        #      label='1m',
                        #      step='month',
                        #      stepmode='backward'),
                        # dict(count=3,
                        #      label='3m',
                        #      step='month',
                        #      stepmode='backward'),
                        # dict(count=6,
                        #      label='6m',
                        #      step='month',
                        #      stepmode='backward'),
                        dict(step='all')
                      ])
                    ),
                rangeslider=dict(),
                type='date'
                )
              )

fig = go.Figure(data=data, layout=layout)
# offline.iplot(fig)
offline.plot(fig)






