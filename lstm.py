# Load Packages

import NAS
import logging
import numpy as np
import os
import pickle
import random
import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import Activation
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.backend import clear_session

# notset > debug > info > warning > error > critical
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Main")

# global base_output_dim
base_output_dim = 0


def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


# CIFAR-10 Data Setup ##############################


def __cifar10__():
	global base_output_dim
	base_output_dim = 10


def load_cifar10(waste_of_space_cifar10=__cifar10__()):
	global base_output_dim
	base_output_dim = 10
	output_dim = 10
	input_shape = (3, 1024)

	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	x_train = x_train.reshape((len(x_train), 3, 1024))
	x_test = x_test.reshape((len(x_test), 3, 1024))
	'''
	  Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test).
	  x_train, x_test: uint8 arrays of RGB image data with shape (num_samples, 3, 32, 32) 
	    if tf.keras.backend.image_data_format() is 'channels_first', 
	    or (num_samples, 32, 32, 3) if the data format is 'channels_last'.
	  y_train, y_test: uint8 arrays of category labels (integers in range 0-9) each with shape (num_samples, 1).
	'''
	# dirpath="/Users/zacharris/Datasets/cifar10/cifar10_batches"
	# train_filenames = ["data_batch_" + str(i) for i in range(1, 6)]
	# test_filename="test_batch"


def load_cifar10(_=__cifar10__()):
	"""
		Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test).
		x_train, x_test: uint8 arrays of RGB image data with shape (num_samples, 3, 32, 32)
		if tf.keras.backend.image_data_format() is 'channels_first',
		or (num_samples, 32, 32, 3) if the data format is 'channels_last'.
		y_train, y_test: uint8 arrays of category labels (integers in range 0-9) each with shape (num_samples, 1).
	"""
	global base_output_dim
	base_output_dim = 1
	input_shape = (3, 1024)

	(x_t, y_t), (x_tst, y_tst) = cifar10.load_data()
	x_t = x_t.reshape((len(x_t), 3, 1024))
	x_tst = x_tst.reshape((len(x_tst), 3, 1024))


	# dirpath = "/Users/zacharris/Datasets/cifar10/cifar10_batches"
	# train_filenames = ["data_batch_" + str(i) for i in range(1, 6)]
	# test_filename = "test_batch"
	
	# train_batches = [unpickle(os.path.join(dirpath, i)) for i in train_filenames]
	# test_batch = unpickle(os.path.join(dirpath, test_filename))


	# start = 0
	# end = 10
	# data_len=end-start
	# dtype='int8'
	# x_train = np.concatenate( [np.asarray(train_batches[i][b'data'][start:end], dtype=dtype).reshape((data_len, 1, input_shape[1])) for i in range(5)])
	# y_train = np.concatenate( [np.asarray(train_batches[i][b'labels'][start:end], dtype=dtype).reshape((data_len, 1)) for i in range(5)])
	# x_test = np.asarray(test_batch[b'data'][start:end], dtype=dtype).reshape((data_len, 1, input_shape[1]))
	# y_test = np.asarray(test_batch[b'labels'][start:end], dtype=dtype).reshape((data_len, 1))
	# # quit(0)

	# logger.info(x_train.shape)
	# logger.debug("global0: " + str(base_output_dim))

	
	# start = 0
	# end = 2
	# num_batches = 2
	# data_len = end - start
	# dtype = 'int8'
	# x_t = np.concatenate([np.asarray(train_batches[i][b'data'][start:end],
	#                                      dtype=dtype).reshape((data_len, input_shape[0], input_shape[1])) for i in range(num_batches)])
	# y_t = np.concatenate([np.asarray(train_batches[i][b'labels'][start:end],
	#                                      dtype=dtype).reshape((data_len, 1)) for i in range(num_batches)])
	# x_tst = np.asarray(test_batch[b'data'][start:end], dtype=dtype).reshape((data_len, input_shape[0], input_shape[1]))
	# y_tst = np.asarray(test_batch[b'labels'][start:end], dtype=dtype).reshape((data_len, 1))
	
	# logger.info(x_t.shape)

	return x_t, y_t, x_tst, y_tst, input_shape



# quit(0)


def random_init_values(activation=None, initializer=None, constraint=None, dropout=None, output_dim=None):
	global base_output_dim

	activation = random.choice({0: "softmax", 1: "softplus", 2: "relu", 3: "tanh", 4: "sigmoid", 5: "hard_sigmoid",
								6: "linear"}) if activation is None else activation
	initializer = random.choice({0: "zero", 1: "uniform", 2: "lecun_uniform", 3: "glorot_normal", 4: "glorot_uniform",
								 5: "normal", 6: "he_normal", 7: "he_uniform"}) if initializer is None else initializer
	constraint = random.choice(
		{0: "maxnorm", 1: "nonneg", 2: "unitnorm", 3: None}) if constraint == 0 else constraint
	dropout = random.choice(
		{0: 0.0, 1: 0.1, 2: 0.15, 3: 0.2, 4: 0.25, 5: 0.3, 6: 0.4, 7: 0.5}) if dropout is None else dropout
	if output_dim is None:
		logger.debug("global2: " + str(base_output_dim))
	output_dim = int(random.random() * 10.0 * base_output_dim) + base_output_dim if output_dim is None else output_dim
	return activation, initializer, constraint, dropout, output_dim


def make_uni_LSTM(output_dim, input_shape, init_values=None, return_sequences=False):
	init_values = random_init_values() if init_values is None else init_values
	return LSTM(output_dim, activation=init_values[0], kernel_initializer=init_values[1],
				kernel_constraint=init_values[2], return_sequences=return_sequences, input_shape=input_shape)


def make_bi_LSTM(output_dim, input_shape, init_values=None, return_sequences=False):
	init_values = random_init_values() if init_values is None else init_values
	return Bidirectional(
		LSTM(output_dim, activation=init_values[0], kernel_initializer=init_values[1], kernel_constraint=init_values[2],
			 return_sequences=return_sequences), input_shape=input_shape)


def make_cascaded_LSTM(output_dim, input_shape, init_values=None, return_sequences=False):
	init_values = random_init_values() if init_values is None else init_values
	return (Bidirectional(
		LSTM(output_dim, activation=init_values[0], kernel_initializer=init_values[1], kernel_constraint=init_values[2],
			 return_sequences=True), input_shape=input_shape),
			LSTM(output_dim, activation=init_values[0], input_shape=input_shape, kernel_initializer=init_values[1],
				 kernel_constraint=init_values[2], return_sequences=return_sequences))


def make_Dense(output_dim, init_values=None):
	init_values = random_init_values() if init_values is None else init_values
	return Dense(output_dim, activation=init_values[0], kernel_initializer=init_values[1],
				 kernel_constraint=init_values[2])


# def make_random_model(output_dim, input_shape, model, random_init_vals,
# 					  type=random.choice(["uni", "bi", "cascaded"])):
# 	# number of layer_types
# 	randrange = int(random.random() * 2) + 1

# 	for i in range(randrange):
# 		output_shape = random.randint(output_dim, output_dim * 10)
# 		not_final_run = (i != randrange - 1)
# 		if type == "uni":
# 			model.add(make_uni_LSTM(output_shape, input_shape, init_values=random_init_vals,
# 									return_sequences=not_final_run))  # True = many to many
# 		elif type == "bi":
# 			model.add(
# 				make_bi_LSTM(output_shape, input_shape, init_values=random_init_vals, return_sequences=not_final_run))
# 		elif type == "cascaded":
# 			(a, b) = make_cascaded_LSTM(output_shape, input_shape, init_values=random_init_vals,
# 										return_sequences=not_final_run)
# 			model.add(a)
# 			model.add(b)
# 		input_shape = (input_shape[0], output_shape, 3)
# 	model.add(make_Dense(input_shape[1], ("sigmoid", "normal", None)))
# 	model.add(make_Dense(output_dim, ("sigmoid", "normal", None)))


# model.add(make_Dense(10, kernel_initializer="normal",activation="linear"))


def remake_pop(population):
	# import NAS

	# for model_i in range(len(population['models'])):
	# 	ind_last = len(population['layer_types'][model_i])-1
	# 	population['input_shapes'][model_i][ind_last] = (None, base_output_dim)
	# 	population['layer_specs'][model_i][ind_last] = random_init_values("sigmoid", "normal", None, output_dim=base_output_dim)
	return NAS.make_pop(input_shapes=population['input_shapes'], layer_types=population['layer_types'], layer_specs=population['layer_specs'],
						pop_size=population['pop_size'], m_type=population['m_type'])

def init_pop(output_dim, input_shape, m_type=random.choice(["uni", "bi", "cascaded"]), pop_size=10):
	# import NAS
	return NAS.make_pop(output_dim=output_dim, input_shapes=input_shape, pop_size=pop_size, m_type=m_type)


def train_test_single_gen(X, y, X_T, y_T, population, epochs, batch_size, validation_split, verbose):
	accuracy = [0 for _ in range(len(population))]
	logger.debug("Fitting models:")
	for model_i in range(len(population)):
		population[model_i].fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
								verbose=verbose)
		accuracy[model_i] = test(X_T, y_T, population[model_i])[1]
	logger.debug("Models tested.")
	return accuracy  # Currently testing on same data as trained


# scores = test(X, y, population[0][0])
# print(scores)
# print('Accuracy: {}'.format(scores[1]))

# population[0].fit(X,y,epochs=epochs,batch_size=batch_size,validation_split=validation_split,verbose=verbose);
# score = test(X, y, population[0])
# return [score]



def train(X, y, X_T, y_T, population, h_params, epochs=tf.constant(500), batch_size=tf.constant(5),
		  validation_split=tf.constant(0.05), verbose=tf.constant(0), input_shape=(3, 1024)):
	# import NAS
	# population = {0: population, 1: layer_types, 2: layer_specs, 3: pop_binary_specifications, 4: m_type, 5: pop_size, 6: input_shapes}
	# h_params = {'generations': 1, 'pop_size': 10, 'crossover_rate': 0.9, 'mutation_rate': 0.3, 'elitism_rate': 0.1}

	for _ in range(h_params['generations']):
		logger.debug("Testing:")
		fitness = train_test_single_gen(X, y, pop_data[0], epochs, batch_size, validation_split, verbose) # Tests on same data as trained
		logger.debug(fitness)
		# # crossover - requires: pop_data, hyperparams, fitness		- returns: pop_data
		logger.debug("Crossover:")
		pop_data = NAS.crossover(pop_data, hyperparams, fitness)
		# mutation  - requires: pop_data, hyperparams 				- returns: pop_data
		pop_data[0][0].summary()
		logger.debug("Mutation:")
		pop_data = NAS.mutation(pop_data, hyperparams)
		# Rebuild the population (of models) with the new specifications
		logger.debug("Remake:")
		pop_data = remake_pop(pop_data)
		pop_data[0][0].summary()
	return pop_data


		# fitness = train_test_single_gen(X, y, X_T, y_T, population['models'], epochs, batch_size, validation_split, verbose)
		fitness = train_test_single_gen(X, y, X_T, y_T, population['models'], epochs, batch_size, validation_split, verbose)
		logger.info(fitness)
		population, num_elites = NAS.crossover(population, h_params, fitness, input_shape=input_shape)
		# population = remake_pop(population) # Only uncomment when testing crossover methods
		population = NAS.mutation(population, h_params, num_elites, base_output_dim)
		clear_session()
		population = remake_pop(population)
		population['models'][0].summary()
	return population


def test(X, y, model, batch_size=tf.constant(5), verbose=tf.constant(1)):
	return model.evaluate(X, y, verbose=verbose, batch_size=batch_size)


# model.compile(loss='mse',optimizer ='adam',metrics=['accuracy'])
# model.fit(X,y,epochs=2000,batch_size=5,validation_split=0.05,verbose=0);


if __name__ == "__main__":
	# Get data

	(x_train, y_train, x_test, y_test, output_dim, input_shape) = load_cifar10()

	(x_train, y_train, x_test, y_test, inp_shape) = load_cifar10()

	logger.debug("global: " + str(base_output_dim))
	logger.info("Data loaded...")
	# quit(0)

	# hyperparams = {'generations': 1, 'pop_size': 10, 'crossover_rate': 0.9, 'mutation_rate': 0.3, 'elitism_rate': 0.1}
	# - crossover rate is useless because what purpose is there to randomly change between init_values? none. it's random and does not carry over information.
	# hyperparams = {'generations': 2, 'pop_size': 2, 'mutation_rate': 0.1, 'elitism_rate': 0.1, 'structure_rate': 0.1}
	# h_params = {'generations': 1, 'pop_size': 10, 'crossover_rate': 0.9, 'mutation_rate': 0.3, 'elitism_rate': 0.1} 
	# - crossover rate is useless because what purpose is there to randomly change between init_values? none.
	# it's random and does not carry over information.
	# h_params = {'generations': 2, 'pop_size': 2, 'mutation_rate': 0.1, 'elitism_rate': 0.1, 'structure_rate': 0.1}

	# to test structure mutations, and their crossover
	# h_params = {'generations': 3, 'pop_size': 3, 'mutation_rate': 1.0, 'elitism_rate': 0.1, 'structure_rate': 1.0}

	# to test layer mutations, and their crossover
	# hyperparameters = {'generations': 3, 'pop_size': 3, 'mutation_rate': 1.0, 'elitism_rate': 0.1, 'structure_rate': 0.0}

	# Actually test algorithm
	# h_params = {'generations': 300, 'pop_size': 150, 'mutation_rate': 0.3, 'elitism_rate': 0.1, 'structure_rate': 0.1}

	# Build a random model/pop
	# Here is the LSTM-ready array with a shape of (100 samples, 5 time steps, 1 feature)
	# make_random_model(output_dim, input_shape, model, random_init_values(), type="uni")
	# make_random_model(output_dim, input_shape, model, random_init_values(), random_init_values(), type="cascaded")
	# model.summary()

	hyperparameters = None

	# binary specifications are useless if I don't perturb by bits
	# population = {0: population, 1: layer_types, 2: layer_specs, 3: pop_binary_specifications, 4: m_type, 5: pop_size, 6: input_shapes}
	if hyperparameters == None:
		hyperparameters = {'generations': 100, 'pop_size': 30, 'mutation_rate': 0.3, 'mutation_percentage': 0.05,'elitism_rate': 0.1, 'structure_rate': 0.1}
		# hyperparameters = {'generations': 5, 'pop_size': 3, 'mutation_rate': 1.0, 'mutation_percentage': 0.05, 'elitism_rate': 0.1, 'structure_rate': 1.0}
		# hyperparameters = {'generations': 5, 'pop_size': 3, 'mutation_rate': 1.0, 'mutation_percentage': 2.50, 'elitism_rate': 0.1, 'structure_rate': 0.0}
	population = init_pop(base_output_dim, inp_shape, m_type="uni", pop_size=hyperparameters['pop_size'])
	population['models'][0].summary()

	population = train(X=x_train, y=y_train, X_T=x_test, y_T=y_test, population=population, h_params=hyperparameters,
	                            epochs=tf.constant(500, dtype=tf.int64), input_shape=inp_shape, batch_size=1)

	X = x_train
	y = y_train
	pop_data = train(X, y, pop_data, hyperparams) # , epochs=tf.constant(10, dtype=tf.int64))
	population['models'][0].summary()

# quit(0)

# scores = test(x_test, y_test)
# scores = test(X, y, population[0][0])
# print(scores)
# print('Accuracy: {}'.format(scores[1]))
# hist = model.fit(X, y) # hist['loss']

# import matplotlib.pyplot as plt
# predict=model.predict(X)
# plt.plot(y, abs(predict-y), 'C2')
# plt.ylim(ymax = 10, ymin = -1)
# plt.show()
