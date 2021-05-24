# Load Packages

import NAS
import logging
import numpy as np
import os
import pickle
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from keras.datasets import cifar10
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
	shape = (out.shape[0], 1, out.shape[1])
	out = np.reshape(out, shape)
	return out

	# items = list(filter(lambda item: item != element, items))


def load_uci_har():
	global base_output_dim
	base_output_dim = 6
	input_shape = (1, 561)
	# input_shape = (None, 1, 561)

	dirpath = "./Data/UCI_HAR_Dataset"
	x_train_filename 	= "train/X_train.txt"
	x_test_filename 	= "test/X_test.txt"
	y_train_filename 	= "train/y_train.txt"
	y_test_filename 	= "test/y_test.txt"

	length = 5
	
	x_t = get_data(x_train_filename, dirpath, x=True)[:length]
	x_tst = get_data(x_test_filename, dirpath, x=True)
	y_t = get_data(y_train_filename, dirpath)[:length]
	y_tst = get_data(y_test_filename, dirpath)
	
	logger.debug(x_t.shape)

	return x_t, y_t, x_tst, y_tst, input_shape



def remake_pop(population, strategy):
	with strategy.scope():
		for model in population:
			model.reinit()


def init_pop(output_dim, input_shape, strategy, m_type=random.choice(["uni", "bi", "cascaded"]), pop_size=10):
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


def save_models(population, generation):
	for model_i in range(len(population)):
		model = population[model_i].get_model()
		model.save("./models/gen_" + str(generation) + "/model_" + str(model_i) + ".h5")


def train(X, y, X_T, y_T, population, h_params, epochs=tf.constant(500), batch_size=tf.constant(5),
		  validation_split=tf.constant(0.05), verbose=tf.constant(0), input_shape=(3, 1024), strategy=None):
	# import NAS
	# h_params = {'generations': 1, 'pop_size': 10, 'crossover_rate': 0.9, 'mutation_rate': 0.3, 'elitism_rate': 0.1}

	for gen in range(h_params['generations']):
		logger.debug("Testing:")



		
		fitness = run_single_gen(X, y, X_T, y_T, population, epochs, batch_size, validation_split, verbose)
		
		logger.debug(fitness)
		
		save_models(population, gen)
		population, num_elites = NAS.crossover(population, h_params, fitness, input_shape=input_shape)
		
		population = NAS.mutation(population, h_params, num_elites)
		clear_session()
		remake_pop(population, strategy)

		population[0].model.build()
		population[0].get_model().summary()
	return population


def test(X, y, model, batch_size=tf.constant(5), verbose=tf.constant(1)):
	return model.evaluate(x=X, y=y, verbose=verbose, batch_size=batch_size)


def test_model_gym(gen=1, model=3, steps=30, batches=1000):
	# model.save("./models/gen_" + str(generation) + "/model_" + str(model_i) + ".h5")
	# model = load_model("./models/gen_" + str(gen) + "/model_" + str(model) + ".h5")
	model = Sequential()
	model.add(LSTM(10, input_shape=(1, 4)))
	model.add(Activation('tanh'))
	model.add(Dense(2))
	model.add(Activation('sigmoid'))
	# model.compile(loss='mse', optimizer='adam')
	model.compile(loss='mse', optimizer='SGD')

	# model.stateful = True
	env = gym.make('CartPole-v1')
	state_size = env.observation_space.shape[0]
	env.reset()
	# base_output_dim = 2
	input_shape = (1, 1, state_size) 

	for _ in range(batches):
		state = env.reset()
		next_state = np.reshape(state, input_shape)
		prev_state = None
		prev_action = None
		done = False
		while not done:
		# for _ in range(steps):
			state = next_state
			env.render()
			if prev_state is not None:
				prev_norm = np.linalg.norm(prev_state)
				if prev_norm < 0.5:
					prev_action = np.array([prev_action])[0]
				else:
					prev_action = np.array([np.abs(prev_action - 1)])[0]
				if prev_norm > 1.5:
					# model.train_on_batch(np.append(prev_state, state, axis=1), np.reshape(prev_action, (1, 2)))
					model.train_on_batch(prev_state, np.reshape(prev_action, (1, 2)))
				orig_action = model.predict(state)[0]
				# orig_action = model.predict(np.append(prev_state, state, axis=1))[0]
			else:
				orig_action = np.asarray([0.0, 0.0])
				# print(orig_action.shape)
				orig_action[random.choice([0, 1])] = 1.0
			# print(orig_action)
			action = np.argmax(orig_action)

			# # print(action)

			# action = np.reshape([0.0, 0.0], (1,2))
			# action[0][action_ind] = 1.0
			# print(action)
			next_state, reward, done, _ = env.step(action)
			# quit(0)
			next_state = np.reshape(next_state, input_shape)

			# if prev_state is not None:
				# outs[model_i].append((state, action, reward, next_state, done))
			prev_state = state
			orig_action[action] = reward + np.amax(orig_action)
			prev_action = orig_action



# model.compile(loss='mse',optimizer ='adam',metrics=['accuracy'])
# model.fit(X,y,epochs=2000,batch_size=5,validation_split=0.05,verbose=0);


def train_with_gym(h_params, steps=30, batches=10):
	env = gym.make('CartPole-v1')
	state_size = env.observation_space.shape[0]
	env.reset()

	base_output_dim = 2
	input_shape = (None, 1, state_size)


	# population = init_pop(base_output_dim, input_shape=(2, state_size), m_type="uni", pop_size=hyperparameters['pop_size'])
	population = init_pop(base_output_dim, input_shape=(1, state_size), m_type="uni", pop_size=hyperparameters['pop_size'])

	for gen in range(h_params['generations']):
		logger.debug("Testing:")

		outs = [[] for _ in range(len(population))]

		fitness = [0 for _ in range(len(population))]

		for _ in range(batches):

			for model_i in range(len(population)):
				model = population[model_i]
				state = env.reset()
				next_state = np.reshape(state, input_shape)
				prev_state = None
				prev_action = None
				done = False
				# Perhaps make fitness how long each model lasted? Cap of some time limit like 100. Random perturbations? Training on data with current and next 
				
				# Get training data
				while not done:
				# for _ in range(steps):
					state = next_state
					# env.render()

					if prev_state is not None:
						prev_norm = np.linalg.norm(prev_state)
						if prev_norm < 0.5:
							prev_action = np.array([prev_action])[0]
						else:
							prev_action = np.array([np.abs(prev_action - 1)])[0]
						if prev_norm > 1.5:
							# model.model.train_on_batch(np.append(prev_state, state, axis=1), np.reshape(prev_action, (1, 2)))
							model.model.train_on_batch(prev_state, np.reshape(prev_action, (1, 2)))
						# np.reshape(target_f, (1, 2))
						# orig_action = model.model.predict(np.append(prev_state, state, axis=1))[0]
						orig_action = model.model.predict(state)[0]
					else:
						orig_action = np.asarray([0.0, 0.0])
						# print(orig_action.shape)
						orig_action[random.choice([0, 1])] = 1.0
					# print(orig_action)
					# print(orig_action)
					action = np.argmax(orig_action)
					# action = np.reshape([0.0, 0.0], (1,2))
					# action[0][action_ind] = 1.0
					# print(action)
					next_state, reward, done, _ = env.step(action)
					# quit(0)
					next_state = np.reshape(next_state, input_shape)

					if prev_state is not None:
						outs[model_i].append((state, action, reward, next_state, done))
					prev_state = state
					orig_action[action] = reward + np.amax(orig_action)
					prev_action = orig_action

				# steps = len(outs[0])
				# # Train the models
				# for state, action, reward, next_state, done in outs[model_i][:int(0.8 * steps)]:
				# 	target = reward
				# 	target_f = model.model.predict(np.append(state, next_state, axis=1))[0]
				# 	if not done:
				# 		target = reward + np.amax(target_f)
				# 	# target_f = model.model.predict(state)[0]
				# 	# print(action)
				# 	# print(target_f)
				# 	target_f[action] = target
				# 	x = np.append(state, next_state, axis=1)
				# 	# print(x.shape)
				# 	# quit(0)
				# 	model.model.fit(x=x, y=np.reshape(target_f, (1, 2)), epochs=1, verbose=0)

				# 
				for state, action, reward, next_state, done in outs[model_i][int(0.8 * steps):]:
					# target = reward
					# target_f = model.model.predict(np.append(state, next_state, axis=1))[0]
					# if not done:
					# 	target = reward + np.amax(target_f)
					# target_f[action] = target
					# acc = model.model.evaluate(x=np.append(state, next_state, axis=1), y=np.reshape(target_f, (1, 2)), verbose=0)[1]
					# fitness[model_i] += acc
					# 
					fitness[model_i] += reward



		logger.debug(fitness)
		save_models(population, gen)
		population, num_elites = NAS.crossover(population, h_params, fitness, input_shape=input_shape)
		# logger.debug(num_elites)

		population = NAS.mutation(population, h_params, num_elites)
		clear_session()
		remake_pop(population)
	
	env.close()
	# population[0].model.build()
	# population[0].get_model().summary()


if __name__ == "__main__":
	# Get data
	(x_train, y_train, x_test, y_test, inp_shape) = load_cifar10()		# shape = n, 1, 3072 or n, 3, 1024
	# (x_train, y_train, x_test, y_test, inp_shape) = load_uci_har()	# shape = n, 1, 561

	# print(x_train.shape)
	# quit()

	mirrored_strategy = tf.distribute.MirroredStrategy()

	# quit(0)
	logger.debug("global: " + str(base_output_dim))


	train_gym = False
	test_gym = False

	hyperparameters = None # future implementation of reading h_params from IO

	if hyperparameters == None:
		# hyperparameters = {'generations': 300, 'pop_size': 50, 'mutation_rate': 0.3, 'mutation_percentage': 0.05,'elitism_rate': 0.1, 'structure_rate': 0.1}
		hyperparameters = {'generations': 30, 'pop_size': 3, 'mutation_rate': 0.3, 'mutation_percentage': 0.05,'elitism_rate': 0.1, 'structure_rate': 0.1}
		# hyperparameters = {'generations': 3, 'pop_size': 3, 'mutation_rate': 1.0, 'mutation_percentage': 0.05, 'elitism_rate': 0.1, 'structure_rate': 1.0}
		# hyperparameters = {'generations': 5, 'pop_size': 3, 'mutation_rate': 1.0, 'mutation_percentage': 2.50, 'elitism_rate': 0.1, 'structure_rate': 0.0}


	if train_gym:
		train_with_gym(hyperparameters)
	elif test_gym:
		test_model_gym()
	else:
		with mirrored_strategy.scope():
			population = init_pop(base_output_dim, inp_shape, mirrored_strategy, m_type="uni", pop_size=hyperparameters['pop_size'])
		# print(inp_shape)
		population[0].get_summary(inp_shape)
		# quit()

		population = train(X=x_train, y=y_train, X_T=x_test, y_T=y_test, population=population, h_params=hyperparameters,
									epochs=tf.constant(100, dtype=tf.int64), input_shape=inp_shape, batch_size=256, strategy=mirrored_strategy)

		population[0].get_model().summary()


