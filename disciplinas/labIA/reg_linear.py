import numpy as np
import cv2
import os

def gradient_descent_step(b0, w0, X_batch, y_batch, learning_rate):
	b_grad = 0
	w_grad = np.zeros(len(w0))

	N = len(X_batch)
	u_aux = np.zeros(len(X_batch))
	for i in range(N):
		u_aux[i] = np.sum(w0*X_batch[i])+b-y_batch[i]

	for i in range(N):
		b_grad += (2.0/N)*u_aux[i]
		for j in range(len(w0)):
			w_grad[j] += (2.0/N)*X_batch[i][j]*u_aux[i]

	# update parameters
	b1 = b0 - (learning_rate * b_grad)
	w1 = w0 - (learning_rate * w_grad)

	return b1, w1

def MSE(b, w, X_batch, y_batch):
	error = 0
	N = len(X_batch)
	for i in range(N):
		y_ = np.sum(X_batch[i]*w)+b
		error += (y_batch[i]-y_)**2
	error /= N
	return error

def accuracy(b, w, X_batch, y_batch):
	correct = 0
	N = len(X_batch)
	for i in range(N):
		y = np.sum(X_batch[i]*w)+b
		if y > 2.5:
			label = 3
		elif y > 1.5:
			label = 2
		elif y > 0.5:
			label = 1
		else:
			label = 0
		if label == y_batch[i]:
			correct += 1
	return 100.*correct/N

# list images
path = 'base/treino/'
classes = sorted(os.listdir(path))
image_names = [[c+'/'+img,i] for i,c in enumerate(classes) for img in sorted(os.listdir(path+'/'+c))]
num_images = len(image_names)

#load images
X_data = np.empty([num_images, 64, 64, 3], dtype=np.uint8)
y_data = np.empty([num_images], dtype=np.int32)

for i,info in enumerate(image_names):
	img = cv2.imread(path+'/'+info[0], cv2.IMREAD_COLOR)
	X_data[i] = img
	y_data[i] = info[1]

# preprocess data
X_data = X_data.reshape((-1, 64*64*3))/255.

# split in training and validation
idx = np.random.permutation(len(X_data))
X_train = np.take(X_data, idx[:len(X_data)/2], axis=0)
y_train = np.take(y_data, idx[:len(X_data)/2], axis=0)
X_val = np.take(X_data, idx[len(X_data)/2:], axis=0)
y_val = np.take(y_data, idx[len(X_data)/2:], axis=0)

# parameter initialization
w = np.zeros(64*64*3).astype(dtype=np.float64)
b = np.zeros(1).astype(dtype=np.float64)

# training loop
learning_rate=0.0001
batch_size=32

print "RANDOM INITIALIZATION"
print "TRAIN: MSE=%.5f ACC=%.5f" % (MSE(b, w, X_train, y_train), accuracy(b, w, X_train, y_train))
print "VAL: MSE=%.5f ACC=%.5f\n" % (MSE(b, w, X_val, y_val), accuracy(b, w, X_val, y_val))

for i in range(1000):
	idx = np.random.permutation(len(X_train))[:batch_size]
	X_batch = np.take(X_train, idx, axis=0)
	y_batch = np.take(y_train, idx, axis=0)

	b, w = gradient_descent_step(b, w, X_batch, y_batch, learning_rate)

	if i%100 == 99:
		print "Iteration #%d" % (i)
		print "TRAIN: MSE=%.5f ACC=%.5f" % (MSE(b, w, X_train, y_train), accuracy(b, w, X_train, y_train))
		print "VAL: MSE=%.5f ACC=%.5f\n" % (MSE(b, w, X_val, y_val), accuracy(b, w, X_val, y_val))


















