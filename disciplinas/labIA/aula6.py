import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf # pip install tensorflow
import numpy as np      # pip install numpy
import cv2              # pip install opencv-python
import os

# parameters
HEIGHT = 64
WIDTH = 64
CHANNELS = 3


# load dataset
X_train = 
y_train = 
X_val = 
y_val = 

# create model as a directed acyclic graph
graph = tf.Graph()
with graph.as_default():
	# INPUT LAYER - https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/placeholder
	# Function: tf.compat.v1.placeholder
	# Format: TYPE, SHAPE
	X = tf.compat.v1.placeholder(tf.float32, shape=(None, HEIGHT*WIDTH*CHANNELS))
	y = tf.compat.v1.placeholder(tf.int64, shape=(None,))
	lr = tf.compat.v1.placeholder(tf.float32)

	# FULLY CONNECTED LAYER - https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/layers/dense
	# Function: tf.layers.dense (DEPRECATED)
	# Format: INPUT, SIZE
	#     INPUT: 2-dimensional array with size (number of images)x(size of an image)
	#     SIZE: integer, number of output nodes
	# Number of parameters: (number of images)x(SIZE) + SIZE
	fc1 = tf.layers.dense(X, 64, activation=tf.nn.relu, name='fc1')
	fc2 = tf.layers.dense(fc1, 32, activation=tf.nn.relu, name='fc2')
	fc3 = tf.layers.dense(fc2, 16, activation=tf.nn.relu, name='fc3')
	fc4 = tf.layers.dense(fc3, 4, name='fc4')

	# one-hot label encoding - https://www.tensorflow.org/api_docs/python/tf/one_hot
	y_one_hot = tf.one_hot(y, 4)

	# https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/sparse_softmax_cross_entropy_with_logits
	loss = tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=fc4)

	# gradient descent step - https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/train/GradientDescentOptimizer
	train_op = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

	correct = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(fc4, axis=1), y), dtype=tf.float32))

	print(X.shape, y.shape, fc1.shape, y_one_hot.shape)
	print(tf.trainable_variables())

def accuracy(session, Xi, yi):
	batch_size=32
	cont=0
	for i in range(0, len(Xi), batch_size):
		X_batch = Xi[i:i+batch_size]
		y_batch = yi[i:i+batch_size]
		ret = session.run([correct], feed_dict = {X : X_batch, y : y_batch})
		cont += ret[0]
	return 100.0*cont/len(Xi)

# train model
with tf.compat.v1.Session(graph = graph) as session:
	# weight initialization
	session.run(tf.compat.v1.global_variables_initializer())

	learning_rate=0.0001
	batch_size=32
	for i in range(10000):
		# get a random batch of images
		idx = np.random.permutation(len(X_train))[:batch_size]
		X_batch = np.take(X_train, idx, axis=0)
		y_batch = np.take(y_train, idx, axis=0)
		ret = session.run([train_op], feed_dict = {X : X_batch, y : y_batch, lr : learning_rate})

		if i%100 == 99:
			print("Iteration #%d" % (i))
			print("TRAIN: ACC=%.5f" % (accuracy(session, X_train, y_train)))
			print("VAL: ACC=%.5f\n" % (accuracy(session, X_val, y_val)))

