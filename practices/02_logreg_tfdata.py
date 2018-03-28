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
import time
import utils

# Step 1: Read in data
mnist_folder = 'data/mnist'
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# Step 2: Create datasets and iterator
train_data = None
train_data = None # if you want to shuffle your data
train_data = None

test_data = None
test_data = None

iterator = None
img, label = None

train_init = None	# initializer for train_data
test_init = None	# initializer for train_data

# Step 3: create weights and bias
w = None
b = None

# Step 4: build model
logits = None

# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
entropy = None
loss = None # computes the mean over all the examples in the batch

# Step 6: define training op
# using gradient descent with learning rate of 0.01 to minimize loss
optimizer = None

# Step 7: calculate accuracy with test set
preds = None
correct_preds = None
accuracy = None

writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())
with tf.Session() as sess:
    start_time = time.time()
    sess.run(None)

    # train the model n_epochs times
    for i in range(30):
        sess.run(None)	# drawing samples from train_data
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run(None)
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # test the model
    sess.run(None)			# drawing samples from test_data
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print('Accuracy {0}'.format(total_correct_preds/10000))
writer.close()
