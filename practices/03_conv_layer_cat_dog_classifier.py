"""
Tensorflow Lecture Note 03
PNU VISLAB
created by Junmyung Jimmy Choi
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
import tensorflow as tf
import numpy as np

def get_files(data_path,datafolder):
  ################
  # your code here#
  ################
  return image_list, label_list

def get_batch(image, label, image_W, image_H, batch_size, capacity):
  image = tf.cast(image, tf.string)
  label = tf.cast(label, tf.int32)

  # make an input queue
  input_queue = tf.train.slice_input_producer([image, label])
  label = input_queue[1]
  image_contents = tf.read_file(input_queue[0])
  image = tf.image.decode_jpeg(image_contents, channels=3)

  # data argumentation
  image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)


  image = tf.image.per_image_standardization(image)

  image_batch, label_batch = tf.train.batch([image, label],
                                            batch_size=batch_size,
                                            num_threads=64,
                                            capacity=capacity)
  label_batch = tf.one_hot(label_batch, depth=2)
  image_batch = tf.cast(image_batch, tf.float32)

  return image_batch, label_batch

def inference(images, batch_size, n_classes):
  '''Build the model
  Args:
      images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
  Returns:
      output tensor with the computed logits, float, [batch_size, n_classes]
  '''
  ################
  # your code here#
  ################
  softmax_linear = None

  return softmax_linear

def losses(logits, labels):
  '''Compute loss from logits and labels
  Args:
      logits: logits tensor, float, [batch_size, n_classes]
      labels: label tensor, tf.int32, [batch_size]

  Returns:
      loss tensor of float type
  '''
  with tf.variable_scope('loss') as scope:
    ################
    # your code here#
    ################
  return loss

# %%
def trainning(loss, learning_rate):

  with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

# %%
def evaluation(logits, labels):

  with tf.variable_scope('accuracy') as scope:
    labels = tf.argmax(labels, 1)
    correct = tf.nn.in_top_k(logits, labels, 1)
    correct = tf.cast(correct, tf.float16)
    accuracy = tf.reduce_mean(correct)
    tf.summary.scalar(scope.name+'/accuracy', accuracy)
  return accuracy

def _run():
  train, train_label = get_files(data_path=data_path,datafolder=datafolder)

  train_batch, train_label_batch = get_batch(train,
                                             train_label,
                                             img_W,
                                             img_H,
                                             batch_size,
                                             batch_size*2)
  train_logits = inference(train_batch, batch_size, n_classes)
  train_loss = losses(train_logits, train_label_batch)
  train_op = trainning(train_loss, learning_rate)
  train__acc = evaluation(train_logits, train_label_batch)

  summary_op = tf.summary.merge_all()
  sess = tf.Session()
  train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
  saver = tf.train.Saver()

  sess.run(tf.global_variables_initializer())
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  try:
    for step in np.arange(10000):
      if coord.should_stop():
        break
      _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])

      if step%50==0:
        print('Step %d, train loss = %.2f, train accuracy = %.2f%%'%(step, tra_loss, tra_acc*100.0))
        summary_str = sess.run(summary_op)
        train_writer.add_summary(summary_str, step)

      if step%2000==0 or (step+1)==10000:
        checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

  except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
  finally:
    coord.request_stop()

  coord.join(threads)
  sess.close()

if __name__=='__main__':
  learning_rate = 0.1
  batch_size = 16
  n_classes = 2
  data_path = '/mnt/DATA/DATASET/catdog/train'
  datafolder = tf.gfile.ListDirectory(data_path)
  img_W = 256
  img_H = 256
  # you need to change the directories to yours.
  logs_train_dir = './graphs/convnet_catdog'
  _run()
  # _evaluate_one_image()