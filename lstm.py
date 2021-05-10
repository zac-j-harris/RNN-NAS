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
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Reshape
from keras.models import Sequential
from keras.backend import clear_session
import gym
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

	if type!="file":

		(x_t, y_t), (x_tst, y_tst) = cifar10.load_data()
		x_t = x_t.reshape((len(x_t), 3, 1024))
		x_tst = x_tst.reshape((len(x_tst), 3, 1024))

	else:
		dirpath = "/Users/zacharris/Datasets/cifar10/cifar10_batches"
		train_filenames = ["data_batch_" + str(i) for i in range(1, 6)]
		test_filename = "test_batch"
		
		train_batches = [unpickle(os.path.join(dirpath, i)) for i in train_filenames]
		test_batch = unpickle(os.path.join(dirpath, test_filename))
		
		start = 0
		end = 1 # out of 10,000
		num_batches = 2 # out of 5
		data_len = end - start
		dtype = 'float32'
		x_t = np.concatenate([np.asarray(train_batches[i][b'data'][start:end],
											 dtype=dtype).reshape((data_len, input_shape[0], input_shape[1])) for i in range(num_batches)])
		y_t = np.concatenate([np.asarray(train_batches[i][b'labels'][start:end],
											 dtype=dtype).reshape((data_len, 1)) for i in range(num_batches)])
		x_tst = np.asarray(test_batch[b'data'][start:end], 
											 dtype=dtype).reshape((data_len, input_shape[0], input_shape[1]))
		y_tst = np.asarray(test_batch[b'labels'][start:end], 
											 dtype=dtype).reshape((data_len, 1))
		
		logger.info(x_t.shape)

		# input_shape = (32, 32, 3)

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
	return [activation, initializer, constraint, dropout, output_dim]


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


# def make_2d_cnn(output_dim, input_shape, init_values=None, return_sequences=False):
# 	init_values = random_init_values() if init_values is None else init_values
# 	new_input_shape = [int(input_shape[0] ** 0.5), int(input_shape[0] ** 0.5), input_shape[1]] if len(input_shape) == 2 else input_shape
# 	return Reshape(target_shape=new_input_shape, input_shape=input_shape), Conv2D(filters=new_input_shape[2], kernel_size=1, activation=init_values[0], 
# 		kernel_initializer=init_values[1], kernel_constraint=init_values[2]), Reshape(target_shape=(new_input_shape[1] ** 2, new_input_shape[2]))


def make_Dense(output_dim, input_shape, init_values=None):
	init_values = random_init_values() if init_values is None else init_values
	return Dense(output_dim, input_shape=input_shape, activation=init_values[0], kernel_initializer=init_values[1],
				 kernel_constraint=init_values[2])




# def remake_pop(population):
# 	# import NAS

# 	# for model_i in range(len(population['models'])):
# 	# 	ind_last = len(population['layer_types'][model_i])-1
# 	# 	population['input_shapes'][model_i][ind_last] = (None, base_output_dim)
# 	# 	population['layer_specs'][model_i][ind_last] = random_init_values("sigmoid", "normal", None, output_dim=base_output_dim)
# 	# return NAS.make_pop(input_shapes=population.get_input_shapes(), layer_types=population.get_layer_types(), layer_specs=population.get_layer_specs(),
# 	# 					pop_size=len(population), m_type=population.get_m_type())
# 	return NAS.make_pop(population)

def init_pop(output_dim, input_shape, m_type=random.choice(["uni", "bi", "cascaded"]), pop_size=10):
	# import NAS
	return NAS.make_pop(output_dim=output_dim, input_shapes=input_shape, pop_size=pop_size, m_type=m_type)


def run_single_gen(X, y, X_T, y_T, population, epochs, batch_size, validation_split, verbose):
	accuracy = [0 for _ in range(len(population))]
	logger.debug("Fitting models:")
	for model_i in range(len(population)):
		population[model_i].model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
								verbose=verbose)
		accuracy[model_i] = test(X_T, y_T, population[model_i].model)[1]
	logger.debug("Models tested.")
	return accuracy  # Currently testing on same data as trained


# scores = test(X, y, population[0][0])
# print(scores)
# print('Accuracy: {}'.format(scores[1]))

# population[0].fit(X,y,epochs=epochs,batch_size=batch_size,validation_split=validation_split,verbose=verbose);
# score = test(X, y, population[0])
# return [score]

def save_models(population, generation):
	for model_i in range(len(population)):
		model = population[model_i].get_model()
		model.save("./models/gen_" + str(generation) + "/model_" + str(model_i) + ".h5")


def train(X, y, X_T, y_T, population, h_params, epochs=tf.constant(500), batch_size=tf.constant(5),
		  validation_split=tf.constant(0.05), verbose=tf.constant(0), input_shape=(3, 1024)):
	# import NAS
	# h_params = {'generations': 1, 'pop_size': 10, 'crossover_rate': 0.9, 'mutation_rate': 0.3, 'elitism_rate': 0.1}

	for gen in range(h_params['generations']):
		logger.debug("Testing:")
		# fitness = train_test_single_gen(X, y, X_T, y_T, population['models'], epochs, batch_size, validation_split, verbose)
		fitness = run_single_gen(X, y, X_T, y_T, population, epochs, batch_size, validation_split, verbose)
		logger.info(fitness)
		save_models(population, gen)
		population, num_elites = NAS.crossover(population, h_params, fitness, input_shape=input_shape)
		# population = remake_pop(population) # Only uncomment when testing crossover methods
		population = NAS.mutation(population, h_params, num_elites)
		clear_session()
		# population = remake_pop(population)
		population[0].model.build()
		population[0].get_model().summary()
	return population


def test(X, y, model, batch_size=tf.constant(5), verbose=tf.constant(1)):
	return model.evaluate(x=X, y=y, verbose=verbose, batch_size=batch_size)


# model.compile(loss='mse',optimizer ='adam',metrics=['accuracy'])
# model.fit(X,y,epochs=2000,batch_size=5,validation_split=0.05,verbose=0);


def train_with_gym(h_params, steps=30):
	# environment = gym.make('CartPole-v1')
	environment = gym.make('CartPole-v1')
	state_size = environment.observation_space.shape[0]
	# environment.reset()

	base_output_dim = 2
	inp_shape = (1, 1, state_size)
	input_shape = (1, state_size)


	population = init_pop(base_output_dim, input_shape, m_type="uni", pop_size=hyperparameters['pop_size'])

	for gen in range(h_params['generations']):
		logger.debug("Testing:")

		outs = [[] for _ in range(len(population))]

		fitness = [0 for _ in range(len(population))]

		for i in range(len(population)):
			model = population[i]
			state = environment.reset()
			prev_state = None
			prev_action = None
			# state = environment.reset()
			next_state = np.reshape(state, inp_shape)
			done = False
			
			# Get training data
			for _ in range(steps):
				state = next_state
				# environment.render()
				if prev_state is not None:
					model.model.train_on_batch(prev_state, prev_action)

				orig_action = model.model.predict(state)[0]
				action = np.argmax(orig_action)
				# action = np.reshape([0.0, 0.0], (1,2))
				# action[0][action_ind] = 1.0
				next_state, reward, done, _ = environment.step(action)
				next_state = np.reshape(next_state, inp_shape)

				outs[i].append((state, action, reward, next_state, done))
				prev_state = state
				orig_action[0][action] = reward + np.amax(model.model.predict(state)[0])
				prev_action = orig_action

			steps = len(outs[0])
			# Train the models
			for state, action, reward, next_state, done in outs[i][:int(0.8 * steps)]:
				target = reward
				if not done:
					target = reward + np.amax(model.model.predict(next_state)[0])
				target_f = model.model.predict(state)[0]
				# print(action)
				# print(target_f)
				target_f[0][action] = target
				model.model.fit(x=state, y=target_f, epochs=1, verbose=0)

			# Test the models
			for state, action, reward, next_state, done in outs[i][int(0.8 * steps):]:
				target = reward
				if not done:
					target = reward + np.amax(model.model.predict(next_state)[0])
				target_f = model.model.predict(state)[0]
				target_f[0][action] = target
				acc = model.model.evaluate(x=state, y=target_f, verbose=0)[1]
				fitness[i] += acc

		logger.info(fitness)
		save_models(population, gen)
		population, num_elites = NAS.crossover(population, h_params, fitness, input_shape=input_shape)

		population = NAS.mutation(population, h_params, num_elites)
		clear_session()
	
	environment.monitor.close()
	population[0].model.build()
	population[0].get_model().summary()


if __name__ == "__main__":
	# Get data
	# (x_train, y_train, x_test, y_test, inp_shape) = load_cifar10()

	# quit(0)
	logger.debug("global: " + str(base_output_dim))


	gym_test = True

	hyperparameters = None # future implementation of reading h_params from IO

	if hyperparameters == None:
		hyperparameters = {'generations': 100, 'pop_size': 30, 'mutation_rate': 0.3, 'mutation_percentage': 0.05,'elitism_rate': 0.1, 'structure_rate': 0.1}
		# hyperparameters = {'generations': 5, 'pop_size': 2, 'mutation_rate': 1.0, 'mutation_percentage': 0.05, 'elitism_rate': 0.1, 'structure_rate': 1.0}
		# hyperparameters = {'generations': 5, 'pop_size': 3, 'mutation_rate': 1.0, 'mutation_percentage': 2.50, 'elitism_rate': 0.1, 'structure_rate': 0.0}


	if gym_test:
		train_with_gym(hyperparameters)
	else:

		population = init_pop(base_output_dim, inp_shape, m_type="uni", pop_size=hyperparameters['pop_size'])
		population[0].get_model().summary()

		population = train(X=x_train, y=y_train, X_T=x_test, y_T=y_test, population=population, h_params=hyperparameters,
									epochs=tf.constant(100, dtype=tf.int64), input_shape=inp_shape, batch_size=1)

		population[0].get_model().summary()

