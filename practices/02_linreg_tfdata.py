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
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import utils

DATA_FILE = 'data/birth_life_2010.txt'

# Step 1: read in the data
data, n_samples = utils.read_birth_life_data(DATA_FILE)

# Step 2: create Dataset and iterator
dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))
iterator = None
X, Y = None

# Step 3: create weight and bias, initialized to 0
w = None
b = None

# Step 4: build model to predict Y
Y_predicted = None

# Step 5: use the square error as the loss function
loss = None

# Step 6: using gradient descent with learning rate of 0.001 to minimize loss
optimizer = None
start = time.time()
with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(None)
    writer = tf.summary.FileWriter('./graphs/linear_reg', sess.graph)
    # Step 8: train the model for 100 epochs
    for i in range(100):
        sess.run(None) # initialize the iterator
        total_loss = 0
        try:
            while True:
                _, l = sess.run(None)
                total_loss += l
        except tf.errors.OutOfRangeError:
            pass
            
        print('Epoch {0}: {1}'.format(i, total_loss/n_samples))
    # close the writer when you're done using it
    writer.close() 
    
    # Step 9: output the values of w and b
    w_out, b_out = None
    print('w: %f, b: %f' %(w_out, b_out))
print('Took: %f seconds' %(time.time() - start))

# # plot the results
# plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
# plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data with squared error')
# # plt.plot(data[:,0], data[:,0] * (-5.883589) + 85.124306, 'g', label='Predicted data with Huber loss')
# plt.legend()
# plt.show()