import numpy as np
from pandas import read_csv
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle
import random

#classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]

classes = range(12)

def load_data_x(x_filename):
	''' Load images file. '''
	x = read_csv(x_filename, sep=',', header=None).values

	return preprocessing.normalize(x, norm='l2')

def load_data_y(y_filename):
	''' Load label file. '''
	y = read_csv(y_filename, header=None).values

	return encode(y)

def encode(y):
	''' 1 of k encoding of labels. '''
	encoding_dict = dict(zip(classes, range(0, len(classes))))
	encoded_y = np.zeros((y.shape[0], len(classes)))

	for i in range(encoded_y.shape[0]):
		encoded_y[i, encoding_dict[y[i, 0]]] += 1

	return encoded_y

def decode(p):
	''' Decode 1 of k encoding of labels. '''
	decoded_p = np.zeros((p.shape[0], 1))
	for i in range(p.shape[0]):
		decoded_p[i, 0] = decode_instance(p[i, :])
	return decoded_p

def decode_instance(i):
	return classes[np.argmax(i)]

def sigmoid(z):
	''' Sigmoid function. '''
	return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
	''' Derivative of sigmoid function. '''
	return sigmoid(z) * (1.0 - sigmoid(z))

def softmax(z):
	''' Softmax function. '''
	return np.exp(z) / np.sum(np.exp(z))

def write_prediction_to_file(p, output_filename):
	''' Write prediction to file in csv format. '''
	output_file = open(output_filename, "w")
	output_file.write("Id,Label\n")

	for i, l in enumerate(p[:, 0]):
		output_file.write("%d,%d\n" % (i, l))

	output_file.close()

def num_correct(p, a):
	''' Number of instances predicted correctly. '''
	return np.sum(p == a)

def accuracy(p, a):
	''' Calculate accuracy of prediction. '''
	return num_correct(p, a) / float(p.shape[0])

def cross_entropy_error(y, t):
	''' Calculate cross entropy error. '''
	return - np.sum(np.multiply(t, np.log(y)) + np.multiply((1 - t), np.log(1 - y)))

def save_neural_net(filename, weights, biases):
	''' Save neural net to file. '''
	with open(filename, 'w') as f:
		pickle.dump(weights, f)
		pickle.dump(biases, f)

def load_neural_net(filename):
	''' Load neural net from file. '''
	with open(filename) as f:
		weights = pickle.load(f)
		biases = pickle.load(f)
	return weights, biases

def feedforward(x, weights, biases):
	''' Neural net feedforward. '''

	activation = x
	activations = [activation]
	all_z = []

	for i, (w, b) in enumerate(zip(weights, biases)):
		z = np.dot(w, activation) + b
		all_z.append(z)
		if (i < len(weights) - 1):
			# activation function for hidden layers is sigmoid
			activation = sigmoid(z)
		else:
			# activation function for output layer is softmax
			activation = softmax(z)
		activations.append(activation)

	return activations, all_z

def mini_batch_gd(train_x, train_y, weights, biases, epochs, mini_batch_size, learning_rate, cross_validate):
	''' Neural net nini-batch gradient descent. '''

	if cross_validate:
		train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.3)

	train_data = zip(train_x, train_y)

	for i in xrange(epochs):
		# divide training data into random mini-batches
		random.shuffle(train_data)
		mini_batches = [train_data[k : k + mini_batch_size] for k in xrange(0, len(train_data), mini_batch_size)]

		# gradient descent on each mini-batch
		for mb in mini_batches:
			gradient_w = [np.zeros(w.shape) for w in weights]
			gradient_b = [np.zeros(b.shape) for b in biases]

			for (x, y) in mb:
				x = np.reshape(x, (x.shape[0], 1))
				y = np.reshape(y, (y.shape[0], 1))
				err_w, err_b = backpropagation(x, y, weights, biases)
				gradient_w = [gw + ew for gw, ew in zip(gradient_w, err_w)]
				gradient_b = [gb + eb for gb, eb in zip(gradient_b, err_b)]

			# update weights and biases
			weights = [w - gw * (learning_rate / len(mb)) for w, gw in zip(weights, gradient_w)]
			biases = [b - gb * (learning_rate / len(mb)) for b, gb in zip(biases, gradient_b)]

		if cross_validate:
			#results_t = predict(train_x, weights, biases)
			results_t = 0
			results_v = predict(test_x, weights, biases)
			#print results.flatten()
			#print decode(test_y).flatten()
			print ("epoch=%d, testing_acc=%d/%d, validation_acc=%d/%d" % (i, np.sum(results_t == decode(train_y)), train_x.shape[0], np.sum(results_v == decode(test_y)), test_x.shape[0]))
		else:
			print ("epoch=%d" % i)

	return weights, biases

def predict_instance(x, weights, biases):
	activations, zs = feedforward(x, weights, biases)
	return decode_instance(activations[-1])

def predict(test_x, weights, biases):
	results = np.zeros((test_x.shape[0], 1))
	for i, x in enumerate(test_x):
		x = np.reshape(x, (x.shape[0], 1))
		results[i, 0] = predict_instance(x, weights, biases)
	return results

def backpropagation(x, y, weights, biases):
	''' Neural net backpropagation. '''

	gradient_w = [np.zeros(w.shape) for w in weights]
	gradient_b = [np.zeros(b.shape) for b in biases]

	# forward pass
	activations, zs = feedforward(x, weights, biases)

	# backward pass
	output_err = activations[-1] - y

	gradient_w[-1] = np.dot(output_err, activations[-2].transpose())
	gradient_b[-1] = output_err

	err = output_err

	for i in xrange(2, len(weights) + 1):
		err = np.dot(weights[-i + 1].transpose(), err) * sigmoid_derivative(zs[-i])
		gradient_w[-i] = np.dot(err, activations[-i - 1].transpose())
		gradient_b[-i] = err

	return gradient_w, gradient_b

def nn(train_x, train_y, num_nodes, epochs, learning_rate, mini_batch_size, cross_validate):
	''' Neural net. '''

	# initialize weights and biases
	weights = []
	biases = []
	for i in xrange(len(num_nodes) - 1):
		weights.append(np.random.randn(num_nodes[i + 1], num_nodes[i]))
		biases.append(np.random.randn(num_nodes[i + 1], 1))

	weights, biases = mini_batch_gd(train_x, train_y, weights, biases, epochs, mini_batch_size, learning_rate, cross_validate)
	
	return weights, biases

def train():
	# training data
	train_x_filename = "train_x.csv"
	train_y_filename = "train_y.csv"

	# sample training data
	train_x_filename = "mini_master_data_final.csv"
	train_y_filename = "mini_master_label_final.csv"

	# load and preprocess data
	train_x = load_data_x(train_x_filename)
	train_y = load_data_y(train_y_filename)

	# hyperparameters
	num_nodes = [train_x.shape[1], 100, len(classes)] # number of nodes in each layer ([input, hidden, output])
	learning_rate = 0.5
	epochs = 500
	mini_batch_size = 32

	cross_validate = True # compute cross validation error while training
	
	weights, biases = nn(train_x, train_y, num_nodes, epochs, learning_rate, mini_batch_size, cross_validate)
	
	# save neural net to file
	save_filename = "nn.pkl"
	save_neural_net(save_filename, weights, biases)

def test():
	# testing data
	test_x_filename = "test_x.csv"
	test_x_filename = "miniseg.csv"

	# preprocess
	test_x = load_data_x(test_x_filename)

	# load neural net from file
	load_filename = "nn.pkl"
	weights, biases = load_neural_net(load_filename)

	# make prediction
	p = predict(test_x, weights, biases)

	write_prediction_to_file(p, "test_y.csv")


if __name__ == "__main__":
	#train();
	test();