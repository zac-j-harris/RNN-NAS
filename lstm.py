# Load Packages

import NAS
import logging
import numpy as np
import os
import pickle
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.backend import clear_session
# import gym
# notset > debug > info > warning > error > critical
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Main")

# global base_output_dim
base_output_dim = 0

SERVER=True


def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


# CIFAR-10 Data Setup ##############################


def __cifar10__():
	global base_output_dim
	base_output_dim = 10


def load_cifar10(type="file", _=__cifar10__()):
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
	# input_shape = (1, 3072)

	if type!="file":

		(x_t, y_t), (x_tst, y_tst) = cifar10.load_data()
		x_t = x_t.reshape((len(x_t), input_shape[0], input_shape[1]))
		x_tst = x_tst.reshape((len(x_tst), input_shape[0], input_shape[1]))

	else:
		dirpath = "/Users/zacharris/Datasets/cifar10/cifar10_batches"
		train_filenames = ["data_batch_" + str(i) for i in range(1, 6)]
		test_filename = "test_batch"
		
		train_batches = [unpickle(os.path.join(dirpath, i)) for i in train_filenames]
		test_batch = unpickle(os.path.join(dirpath, test_filename))
		
		# start = 0
		end = 5 # out of 10,000
		num_batches = 1 # out of 5
		data_len = end

		dtype = 'float32'
		x_t = np.concatenate([np.asarray(train_batches[i][b'data'][:end],
											 dtype=dtype).reshape((data_len, input_shape[0], input_shape[1])) for i in range(num_batches)])
		y_t = np.concatenate([np.asarray(train_batches[i][b'labels'][:end],
											 dtype=dtype).reshape((data_len, 1)) for i in range(num_batches)])
		x_tst = np.asarray(test_batch[b'data'][:end], 
											 dtype=dtype).reshape((data_len, input_shape[0], input_shape[1]))
		y_tst = np.asarray(test_batch[b'labels'][:end], 
											 dtype=dtype).reshape((data_len, 1))
		
		logger.debug(x_t.shape)

		# input_shape = (32, 32, 3)
		# input_shape = (None, 3, 1024)

	return x_t, y_t, x_tst, y_tst, input_shape


def get_data(filename, dirpath, x=False, dtype='float32'):
	with open(os.path.join(dirpath, filename), 'r') as f:
		all_data = f.read()
	all_data = all_data.split("\n") # shape = (7353,)
	if x:
		for i in range(len(all_data) - 1):
			ins = all_data[i].replace("  ", " ")
			all_data[i] = ins.split(" ")[1:]

	all_data = all_data[:len(all_data)-1]

	if not x:
		# out = all_data
		# print(out)
		for i in range(len(all_data)):
			val = int(all_data[i]) - 1
			all_data[i] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
			all_data[i][val] = 1.0

	out = np.asarray(all_data, dtype=dtype)
	if x:
		shape = (out.shape[0], 1, out.shape[1])
	else:
		shape = out.shape
	out = np.reshape(out, shape)
	return out

	# items = list(filter(lambda item: item != element, items))


def load_uci_har():
	global base_output_dim
	base_output_dim = 6
	input_shape = (1, 561)
	# input_shape = (None, 1, 561)

	if SERVER: #
		dirpath = "/home/zharris1/Documents/Github/RNN-NAS/Data/UCI_HAR_Dataset"
	else:
		dirpath = "./Data/UCI_HAR_Dataset"
	x_train_filename 	= "train/X_train.txt"
	x_test_filename 	= "test/X_test.txt"
	y_train_filename 	= "train/y_train.txt"
	y_test_filename 	= "test/y_test.txt"

	# length = 5
	# if not SERVER: #
	# 	x_t = get_data(x_train_filename, dirpath, x=True)[:length]
	# 	y_t = get_data(y_train_filename, dirpath)[:length]
	# else:

	x_t = get_data(x_train_filename, dirpath, x=True)
	y_t = get_data(y_train_filename, dirpath)
	x_tst = get_data(x_test_filename, dirpath, x=True)
	y_tst = get_data(y_test_filename, dirpath)
	
	logger.debug(x_t.shape)

	return x_t, y_t, x_tst, y_tst, input_shape



def remake_pop(population, strategy):
	if strategy is not None:
		with strategy.scope():
			for model in population:
				model.reinit()
	else:
		for model in population:
			model.reinit()

def init_pop(output_dim, input_shape, strategy, m_type=random.choice(["uni", "bi", "cascaded"]), pop_size=10):
	# import NAS
	return NAS.make_pop(output_dim=output_dim, input_shapes=input_shape, pop_size=pop_size, m_type=m_type)


def run_single_gen(X, y, X_T, y_T, population, epochs, batch_size, validation_split, verbose):
	accuracy = [0 for _ in range(len(population))]
	logger.debug("Fitting models:")
	for model_i in range(len(population)):
		# Can't use validation split with distribution strats
		# population[model_i].model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
		# 						verbose=verbose)
		population[model_i].model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
		accuracy[model_i] = test(X_T, y_T, model=population[model_i].model, batch_size=batch_size)[1]
	logger.debug("Models tested.")
	return accuracy  # Currently testing on same data as trained


def save_models(population, generation):
	for model_i in range(len(population)):
		model = population[model_i].get_model()
		model.save("./models/gen_" + str(generation) + "_model_" + str(model_i) + ".h5")


def train(X, y, X_T, y_T, population, h_params, epochs=tf.constant(500), batch_size=tf.constant(7),
		  validation_split=0.05, verbose=0, input_shape=(3, 1024), strategy=None):
	# import NAS
	# h_params = {'generations': 1, 'pop_size': 10, 'crossover_rate': 0.9, 'mutation_rate': 0.3, 'elitism_rate': 0.1}
	elites = [0]
	for gen in range(h_params['generations']):
		logger.debug("Testing:")
		
		fitness = run_single_gen(X, y, X_T, y_T, population, epochs, batch_size, validation_split, verbose)
		
		logger.debug(fitness)

		save_models(population, gen)
		clear_session()
		population, elites = NAS.crossover(population, h_params, fitness, input_shape=input_shape)
		
		population = NAS.mutation(population, h_params, elites)
		clear_session()
		for model in population:
			model.layer_specs[len(model.layer_specs)-1][4] = 6
		remake_pop(population, strategy)

		population[elites[0]].model.build(input_shape=(None, 1, 561))
		population[elites[0]].get_model().summary()
		for i in elites:
			print(population[i].get_layer_specs())
	return population, elites


def test(X, y, model, batch_size=tf.constant(7), verbose=1):
	return model.evaluate(x=X, y=y, verbose=verbose, batch_size=batch_size)


if __name__ == "__main__":
	# Get data
	# (x_train, y_train, x_test, y_test, inp_shape) = load_cifar10()		# shape = n, 1, 3072 or n, 3, 1024
	(x_train, y_train, x_test, y_test, inp_shape) = load_uci_har()	# shape = n, 1, 561


	if SERVER: # changed to test
		mirrored_strategy = tf.distribute.MirroredStrategy()
		# mirrored_strategy = None
	else:
		mirrored_strategy = None
		# mirrored_strategy = tf.distribute.MirroredStrategy(num_gpus=4)

	# quit(0)
	logger.debug("global: " + str(base_output_dim))


	train_gym = False
	test_gym = False


	# hyperparameters = {'generations': 300, 'pop_size': 50, 'mutation_rate': 0.3, 'mutation_percentage': 0.05,'elitism_rate': 0.1, 'structure_rate': 0.1}
	# hyperparameters = {'generations': 30, 'pop_size': 3, 'mutation_rate': 0.8, 'mutation_percentage': 0.20,'elitism_rate': 0.1, 'structure_rate': 0.1}
	# hyperparameters = {'generations': 300, 'pop_size': 50, 'mutation_rate': 0.80, 'mutation_percentage': 0.20, 'elitism_rate': 0.1, 'structure_rate': 0.80}

	# As described in paper (crossover rate and mutation percentage differ)
	if SERVER: #
		# hyperparameters = {'generations': 30, 'pop_size': 20, 'mutation_rate': 0.30, 'mutation_percentage': 0.05,'elitism_rate': 0.1, 'structure_rate': 0.1}
		# epochs = 100
		hyperparameters = {'generations': 30, 'pop_size': 20, 'mutation_rate': 0.30, 'mutation_percentage': 0.15, 'elitism_rate': 0.1, 'structure_rate': 0.25}
		epochs = 32
		# hyperparameters = {'generations': 100, 'pop_size': 25, 'mutation_rate': 0.30, 'mutation_percentage': 0.3,'elitism_rate': 0.1, 'structure_rate': 0.33}
		# epochs = 250
		# Epochs - 32, optimizer - Adam
	else:
		hyperparameters = {'generations': 10, 'pop_size': 3, 'mutation_rate': 1.0, 'mutation_percentage': 0.5,
		                   'elitism_rate': 0.1, 'structure_rate': 1.33}
		epochs = 20

	# if train_gym:
	# 	train_with_gym(hyperparameters)
	# elif test_gym:
	# 	test_model_gym()
	# else:

	if mirrored_strategy is not None:
		with mirrored_strategy.scope():
			population = init_pop(base_output_dim, inp_shape, mirrored_strategy, m_type="uni", pop_size=hyperparameters['pop_size'])
	else:
		population = init_pop(base_output_dim, inp_shape, mirrored_strategy, m_type="uni", pop_size=hyperparameters['pop_size'])
	# print(inp_shape)
	# population[0].get_summary(inp_shape)
	# quit()

	population, elites = train(X=x_train, y=y_train, X_T=x_test, y_T=y_test, population=population, h_params=hyperparameters,
								epochs=epochs, input_shape=inp_shape, batch_size=7, strategy=mirrored_strategy)

	population[elites[0]].get_model().build(input_shape=(None, 1, 561))
	population[elites[0]].get_model().summary()
	print(population[elites[0]].get_layer_specs())
	print(population[elites[0]].get_layer_types())


	# Secondary Test:
	second_test = True
	if second_test:
		clear_session()
		print(('#' * 40 + '\n\n') * 10)
		hyperparameters = {'generations': 100, 'pop_size': 30, 'mutation_rate': 0.30, 'mutation_percentage': 0.3,
		                   'elitism_rate': 0.1, 'structure_rate': 0.25}
		epochs = 64
		population = init_pop(base_output_dim, inp_shape, mirrored_strategy, m_type="uni",
		                      pop_size=hyperparameters['pop_size'])
		population, elites = train(X=x_train, y=y_train, X_T=x_test, y_T=y_test, population=population,
		                   h_params=hyperparameters, epochs=epochs, input_shape=inp_shape, batch_size=7,
		                   strategy=mirrored_strategy)
		population[elites[0]].get_model().build(input_shape=(None, 1, 561))
		population[elites[0]].get_model().summary()
		print(population[elites[0]].get_layer_specs())
		print(population[elites[0]].get_layer_types())


