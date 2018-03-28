""" Solution for simple logistic regression model for MNIST
with placeholder
MNIST dataset: yann.lecun.com/exdb/mnist/
Created by Chip Huyen (huyenn@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

import utils


# Step 1: Read in data
mnist = input_data.read_data_sets('data/mnist', one_hot=True)
X_batch, Y_batch = mnist.train.next_batch(128)

# Step 2: create placeholders for features and labels
X = tf.placeholder(tf.float32, [128, 784], name='image')
Y = tf.placeholder(tf.int32, [128, 10], name='label')

# Step 3: create weights and bias
w = tf.get_variable(name='weights', shape=(784, 10), initializer=tf.random_normal_initializer())
b = tf.get_variable(name='bias', shape=(1, 10), initializer=tf.zeros_initializer())

# Step 4: build model
logits = tf.matmul(X, w) + b 

# Step 5: define loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
loss = tf.reduce_mean(entropy) # computes the mean over all the examples in the batch

# Step 6: define training op
# using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg_placeholder', tf.get_default_graph())
with tf.Session() as sess:
	start_time = time.time()
	sess.run(tf.global_variables_initializer())	
	n_batches = int(mnist.train.num_examples/128)
	
	# train the model n_epochs times
	for i in range(30):
		total_loss = 0

		for j in range(n_batches):
			X_batch, Y_batch = mnist.train.next_batch(128)
			_, loss_batch = sess.run([optimizer, loss], {X: X_batch, Y:Y_batch}) 
			total_loss += loss_batch
		print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
	print('Total time: {0} seconds'.format(time.time() - start_time))

	# test the model
	n_batches = int(mnist.test.num_examples/128)
	total_correct_preds = 0

	for i in range(n_batches):
		X_batch, Y_batch = mnist.test.next_batch(128)
		accuracy_batch = sess.run(accuracy, {X: X_batch, Y:Y_batch})
		total_correct_preds += accuracy_batch	

	print('Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples))
writer.close()
