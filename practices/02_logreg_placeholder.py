"""
Tensorflow lecture note 02
PNU VISLAB
modified by Junmyung Jimmy Choi

Created by Chip Huyen (chiphuyen@cs.stanford.edu)
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
X = None
Y = None

# Step 3: create weights and bias
w = None
b = None

# Step 4: build model
logits = None

# Step 5: define loss function
entropy = None
loss = None # computes the mean over all the examples in the batch

# Step 6: define training op
# using gradient descent with learning rate of 0.01 to minimize loss
optimizer = None

# Step 7: calculate accuracy with test set
preds = None
correct_preds = None
accuracy = None

writer = tf.summary.FileWriter('./graphs/logreg_placeholder', tf.get_default_graph())
with tf.Session() as sess:
	start_time = time.time()
	sess.run(None)
	n_batches = None

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
