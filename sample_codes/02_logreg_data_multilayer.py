""" Solution for simple logistic regression model for MNIST
with tf.data module
MNIST dataset: yann.lecun.com/exdb/mnist/
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time
import utils

# Step 1: Read in data
mnist_folder = 'data/mnist'
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# Step 2: Create datasets and iterator
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000) # if you want to shuffle your data
train_data = train_data.batch(128)

test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.batch(128)

iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                           train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)	# initializer for train_data
test_init = iterator.make_initializer(test_data)	# initializer for train_data

# Step 3 + Step 4: build model with wrapper
mlp1 = tf.layers.dense(img,
                      units=32,
                      activation=tf.nn.relu,
                      use_bias=True,
                      kernel_initializer=tf.initializers.variance_scaling,
                      bias_initializer=tf.initializers.zeros,
                      trainable=True,
                      name='fc1',
                      reuse=None)
mlp2 = tf.layers.dense(mlp1,
                      units=64,
                      activation=None,
                      use_bias=True,
                      kernel_initializer=tf.initializers.variance_scaling,
                      bias_initializer=tf.initializers.zeros,
                      trainable=True,
                      name='fc2',
                      reuse=None)
mlp3 = tf.layers.dense(mlp1,
                      units=64,
                      activation=None,
                      use_bias=True,
                      kernel_initializer=tf.initializers.variance_scaling,
                      bias_initializer=tf.initializers.zeros,
                      trainable=True,
                      name='fc3',
                      reuse=None)
# Step 3 + Step 4: build model with wrapper
logits = tf.layers.dense(img,
                      units=10,
                      activation=None,
                      use_bias=True,
                      kernel_initializer=tf.initializers.variance_scaling(),
                      bias_initializer=tf.initializers.zeros(),
                      trainable=True,
                      name='logits',
                      reuse=None)

# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label, name='entropy')
loss = tf.reduce_mean(entropy, name='loss') # computes the mean over all the examples in the batch

# Step 6: define training op
# using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg_mlp', tf.get_default_graph())
with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    # train the model n_epochs times
    for i in range(30):
        sess.run(train_init)	# drawing samples from train_data
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # test the model
    sess.run(test_init)			# drawing samples from test_data
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print('Accuracy {0}'.format(total_correct_preds/10000))
writer.close()
