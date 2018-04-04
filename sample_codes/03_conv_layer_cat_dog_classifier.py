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
  cats = []
  label_cats = []
  dogs = []
  label_dogs = []
  for file in datafolder:
    name = file.split('.')
    if name[0]=='cat':
      cats.append(os.path.join(data_path, file))
      label_cats.append(0)
    else:
      dogs.append(os.path.join(data_path, file))
      label_dogs.append(1)

  image_list = np.hstack((cats, dogs))
  label_list = np.hstack((label_cats, label_dogs))

  shuffle = list(zip(image_list, label_list))
  random.shuffle(shuffle)
  image_list, label_list = zip(*shuffle)
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

  with tf.variable_scope('conv1') as scope:
    weights = tf.get_variable('weights',
                              shape=[3, 3, 3, 16],
                              dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    biases = tf.get_variable('biases',
                             shape=[16],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.1))
    conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)

  # pool1
  with tf.variable_scope('pooling1_lrn') as scope:
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pooling1')


  # conv2
  with tf.variable_scope('conv2') as scope:
    weights = tf.get_variable('weights',
                              shape=[3, 3, 16, 16],
                              dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    biases = tf.get_variable('biases',
                             shape=[16],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.1))
    conv = tf.nn.conv2d(pool1, weights, strides=[1, 1, 1, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name='conv2')

  # pool2 a
  with tf.variable_scope('pooling2_lrn') as scope:
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                           padding='SAME', name='pooling2')

  # local3
  with tf.variable_scope('local3') as scope:
    reshape = tf.reshape(pool2, shape=[batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = tf.get_variable('weights',
                              shape=[dim, 128],
                              dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
    biases = tf.get_variable('biases',
                             shape=[128],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights)+biases, name=scope.name)

    # local4
  with tf.variable_scope('local4') as scope:
    weights = tf.get_variable('weights',
                              shape=[128, 128],
                              dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
    biases = tf.get_variable('biases',
                             shape=[128],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights)+biases, name='local4')

  # softmax
  with tf.variable_scope('softmax_linear') as scope:
    weights = tf.get_variable('softmax_linear',
                              shape=[128, n_classes],
                              dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
    biases = tf.get_variable('biases',
                             shape=[n_classes],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.1))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

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
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2 \
      (logits=logits, labels=labels, name='xentropy_per_example')
    loss = tf.reduce_mean(cross_entropy, name='loss')
    tf.summary.scalar(scope.name+'/loss', loss)
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


  # %% Evaluate one image
  # when training, comment the following codes.


from PIL import Image
import matplotlib.pyplot as plt

def get_one_image(train):
   '''Randomly pick one image from training data
   Return: ndarray
   '''
   n = len(train)
   ind = np.random.randint(0, n)
   img_dir = train[ind]

   image = Image.open(img_dir)
   plt.imshow(image)
   image = image.resize([img_W, img_H])
   image = np.array(image)
   return image

def _evaluate_one_image():
   data_path = '/mnt/DATA/DATASET/catdog/train'
   datafolder = tf.gfile.ListDirectory(data_path)
   train, train_label = get_files(data_path=data_path, datafolder=datafolder)

   image_array = get_one_image(train)

   with tf.Graph().as_default():
       BATCH_SIZE = 1
       N_CLASSES = 2

       image = tf.cast(image_array, tf.float32)
       image = tf.image.per_image_standardization(image)
       image = tf.reshape(image, [1, img_W, img_H, 3])
       logit = inference(image, BATCH_SIZE, N_CLASSES)

       logit = tf.nn.softmax(logit)

       x = tf.placeholder(tf.float32, shape=[img_W, img_H, 3])

       # you need to change the directories to yours.
       logs_train_dir = './graphs/convnet_catdog'

       saver = tf.train.Saver()

       with tf.Session() as sess:

           print("Reading checkpoints...")
           ckpt = tf.train.get_checkpoint_state(logs_train_dir)
           if ckpt and ckpt.model_checkpoint_path:
               global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
               saver.restore(sess, ckpt.model_checkpoint_path)
               print('Loading success, global_step is %s' % global_step)
           else:
               print('No checkpoint file found')

           prediction = sess.run(logit, feed_dict={x: image_array})
           max_index = np.argmax(prediction)
           if max_index==0:
               print('This is a cat with possibility %.6f' %prediction[:, 0])
           else:
               print('This is a dog with possibility %.6f' %prediction[:, 1])


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