#Load Packages
import numpy as np
from keras.models import Sequential
# from tensorflow import int8
import tensorflow as tf
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Activation
from keras.datasets import cifar10
import random, pickle, os, logging
import NAS


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



##### CIFAR-10 Data Setup ##############################

def __cifar10__():
	global base_output_dim
	base_output_dim = 10

def cifar10(waste_of_space_cifar10=__cifar10__()):
	global base_output_dim
	base_output_dim = 10
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
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

	# train_batches = [unpickle(os.path.join(dirpath, i)) for i in train_filenames]
	# # logger.info(len(train_batches))
	# test_batch = unpickle(os.path.join(dirpath, test_filename))
	# output_dim = 10

	# start = 0
	# end = 10
	# data_len=end-start
	# input_shape = (1, 3072)
	# dtype='int8'
	# x_train = np.concatenate( [np.asarray(train_batches[i][b'data'][start:end], dtype=dtype).reshape((data_len, 1, input_shape[1])) for i in range(5)])
	# y_train = np.concatenate( [np.asarray(train_batches[i][b'labels'][start:end], dtype=dtype).reshape((data_len, 1)) for i in range(5)])
	# x_test = np.asarray(test_batch[b'data'][start:end], dtype=dtype).reshape((data_len, 1, input_shape[1]))
	# y_test = np.asarray(test_batch[b'labels'][start:end], dtype=dtype).reshape((data_len, 1))
	# # quit(0)

	# logger.info(x_train.shape)
	# logger.debug("global0: " + str(base_output_dim))

	return (x_train, y_train, x_test, y_test, output_dim, input_shape)



##### Random Data Setup ##############################
# input_shape = (5,1)

# #Generate 2 sets of X variables
# #LSTMs have unique 3-dimensional input requirements 
# seq_length=5
# X =[[i+j for j in range(seq_length)] for i in range(100)]
# X_simple =[[i for i in range(4,104)]]
# X =np.array(X)
# X_simple=np.array(X_simple)


# # Generate lagged Y-variable
# y =[[ i+(i-1)*.5+(i-2)*.2+(i-3)*.1 for i in range(4,104)]]
# y =np.array(y)
# X_simple=X_simple.reshape((100,1))
# X=X.reshape((100,5,1))
# y=y.reshape((100,1))


# quit(0)



def random_init_values(activation=None, initializer=None, constraint=None, dropout=None, output_dim=None):
	global base_output_dim
	# quit(0)
	activation = random.choice({0: "softmax", 1: "softplus", 2: "relu", 3: "tanh", 4: "sigmoid", 5: "hard_sigmoid", 6: "linear"}) if activation == None else activation
	initializer = random.choice({0: "zero", 1: "uniform", 2: "lecun_uniform", 3: "glorot_normal", 4: "glorot_uniform", 
								 5: "normal", 6: "he_normal", 7: "he_uniform"}) if initializer == None else initializer
	constraint = random.choice({0: "maxnorm", 1: "nonneg", 2: "unitnorm", 3: None}) if constraint == None else constraint
	dropout = random.choice({0: 0.0, 1: 0.1, 2: 0.15, 3: 0.2, 4: 0.25, 5: 0.3, 6: 0.4, 7: 0.5}) if dropout == None else dropout
	if output_dim == None:
		logger.debug("global2: " + str(base_output_dim))
	output_dim = int(random.random() * 10.0 * base_output_dim) + base_output_dim if output_dim == None else output_dim
	return (activation, initializer, constraint, dropout, output_dim)


def make_uni_LSTM(output_dim, input_shape, init_values=None, return_sequences=False, dtype=np.int8):
	init_values = random_init_values() if init_values == None else init_values
	# return LSTM(output_dim, activation=init_values[0], kernel_initializer=init_values[1], kernel_constraint=init_values[2], return_sequences=return_sequences, dtype=dtype, input_shape=input_shape)
	return LSTM(output_dim, activation=init_values[0], kernel_initializer=init_values[1], kernel_constraint=init_values[2], return_sequences=return_sequences, input_shape=input_shape)

def make_bi_LSTM(output_dim, input_shape, init_values=None, return_sequences=False, dtype=np.int8):
	# return Bidirectional(LSTM(output_dim, activation=init_values[0], kernel_initializer=init_values[1], kernel_constraint=init_values[2], return_sequences=return_sequences), dtype=dtype, input_shape=input_shape)
	init_values = random_init_values() if init_values == None else init_values
	return Bidirectional(LSTM(output_dim, activation=init_values[0], kernel_initializer=init_values[1], kernel_constraint=init_values[2], return_sequences=return_sequences), input_shape=input_shape)

# def make_cascaded_LSTM(output_dim, input_shape, init_values1=random_init_values(), init_values2=random_init_values(), return_sequences=False):
def make_cascaded_LSTM(output_dim, input_shape, init_values=None, return_sequences=False, dtype=np.int8):
	# return (Bidirectional(LSTM(output_dim, activation=init_values[0], kernel_initializer=init_values[1], kernel_constraint=init_values[2], return_sequences=True), dtype=dtype, input_shape=input_shape),
	# 		LSTM(output_dim, activation=init_values[0], input_shape=input_shape, dtype=dtype, kernel_initializer=init_values[1], kernel_constraint=init_values[2], return_sequences=return_sequences))
	init_values = random_init_values() if init_values == None else init_values
	return (Bidirectional(LSTM(output_dim, activation=init_values[0], kernel_initializer=init_values[1], kernel_constraint=init_values[2], return_sequences=True), input_shape=input_shape),
			LSTM(output_dim, activation=init_values[0], input_shape=input_shape, kernel_initializer=init_values[1], kernel_constraint=init_values[2], return_sequences=return_sequences))

def make_Dense(output_dim, init_values=None):
	init_values = random_init_values() if init_values == None else init_values
	return Dense(output_dim, activation=init_values[0], kernel_initializer=init_values[1], kernel_constraint=init_values[2])


def make_random_model(output_dim, input_shape, model, random_init_values, random_init_values2=None, type=random.choice(["uni", "bi", "cascaded"])):
	# randrange = int(random.random() * 10)
	# if type == "cascaded" and random_init_values2 == None:
	# 	random_init_values2 = random_init_values()

	# number of layers
	randrange = int(random.random() * 2) + 1

	for i in range(randrange):
		output_shape = random.randint(output_dim, output_dim * 10)
		not_final_run = (i != randrange - 1)
		# print("i = " + str(i) + ", randrange = " + str(randrange) + ", not_final_run = " + str(not_final_run))
		if type=="uni":
			model.add(make_uni_LSTM(output_shape, input_shape, init_values=random_init_values, return_sequences=not_final_run)) #True = many to many
		elif type == "bi":
			model.add(make_bi_LSTM(output_shape, input_shape, init_values=random_init_values, return_sequences=not_final_run))
		elif type == "cascaded":
			(a, b) = make_cascaded_LSTM(output_shape, input_shape, init_values=random_init_values, return_sequences=not_final_run)
			model.add(a)
			model.add(b)
		input_shape = (input_shape[0], output_shape, 1)
	model.add(make_Dense(input_shape[1], ("sigmoid", "normal", None)))
	model.add(make_Dense(output_dim, ("sigmoid", "normal", None)))
	# model.add(make_Dense(10, kernel_initializer="normal",activation="linear"))



def remake_pop(pop_data):
	# import NAS
	# return NAS.make_pop(output_dims=pop_data[5], input_shapes=pop_data[6], layers=pop_data[1], model_specifications=pop_data[2], pop_binary_specifications=pop_data[3], pop_size=pop_data[7], m_type=pop_data[4])
	return NAS.make_pop(input_shapes=pop_data[6], layers=pop_data[1], model_specifications=pop_data[2], pop_binary_specifications=pop_data[3], pop_size=pop_data[5], m_type=pop_data[4])


def train_test_single_gen(X, y, population, epochs, batch_size, validation_split, verbose):
	accuracy = [0 for _ in range(len(population))]
	logger.debug("Fitting models:")
	for model_i in range(len(population)):
		population[model_i].fit(X,y,epochs=epochs,batch_size=batch_size,validation_split=validation_split,verbose=verbose);
		accuracy[model_i] = test(X, y, population[model_i] )[1]
	logger.debug("Testing models:")
	return accuracy # Currently testing on same data as trained


# def train_test_single_gen(X, y, population, epochs, batch_size, validation_split, verbose):
# 	logger.debug("Fitting models:")
# 	for model in population:
# 		model.fit(X,y,epochs=epochs,batch_size=batch_size,validation_split=validation_split,verbose=verbose);
# 	logger.debug("Testing models:")
# 	# return [test(X, y, population[i])[1] for i in range(len(population))] # Tests on same data as trained
# 	return [test(X, y, population[model_i]  )[1] for model_i in range(len(population))] # Tests on same data as trained
	# return [random.random() * 2 + 9 for i in range(len(population))]

	# scores = test(X, y, pop_data[0][0])
	# print(scores)
	# print('Accuracy: {}'.format(scores[1]))

	# population[0].fit(X,y,epochs=epochs,batch_size=batch_size,validation_split=validation_split,verbose=verbose);
	# score = test(X, y, population[0])
	# return [score]



def train(X, y, pop_data, hyperparams, epochs=tf.constant(500), batch_size=tf.constant(5), validation_split=tf.constant(0.05), verbose=tf.constant(0)):
	# import NAS
	# pop_data = {0: population, 1: layers, 2: model_specifications, 3: pop_binary_specifications, 4: m_type, 5: pop_size, 6: input_shapes}
	# hyperparams = {'generations': 1, 'pop_size': 10, 'crossover_rate': 0.9, 'mutation_rate': 0.3, 'elitism_rate': 0.1}

	for _ in range(hyperparams['generations']):
		logger.debug("Testing:")
		fitness = train_test_single_gen(X, y, pop_data[0], epochs, batch_size, validation_split, verbose) # Tests on same data as trained
		logger.info(fitness)
		# # crossover - requires: pop_data, hyperparams, fitness		- returns: pop_data
		logger.debug("Crossover:")
		pop_data = NAS.crossover(pop_data, hyperparams, fitness)
		# mutation  - requires: pop_data, hyperparams 				- returns: pop_data
		pop_data[0][0].summary()
		logger.info("Mutation:")
		pop_data = NAS.mutation(pop_data, hyperparams)
		# Rebuild the population (of models) with the new specifications
		logger.debug("Remake:")
		pop_data = remake_pop(pop_data)
		pop_data[0][0].summary()
	return pop_data



def test(X, y, model, batch_size=tf.constant(5), verbose=tf.constant(1)):
	return model.evaluate(X,y,verbose=verbose,batch_size=batch_size)




def init_pop(output_dim, input_shape, m_type=random.choice(["uni", "bi", "cascaded"]), pop_size=10):
	# import NAS
	return NAS.make_pop(output_dim=output_dim, input_shapes=input_shape, pop_size=pop_size, m_type=m_type)

# model.compile(loss='mse',optimizer ='adam',metrics=['accuracy'])
# model.fit(X,y,epochs=2000,batch_size=5,validation_split=0.05,verbose=0);


if __name__ == "__main__":
	# Get data
	(x_train, y_train, x_test, y_test, output_dim, input_shape) = cifar10()
	logger.debug("global: " + str(base_output_dim))

	# quit(0)
	# hyperparams = {'generations': 1, 'pop_size': 10, 'crossover_rate': 0.9, 'mutation_rate': 0.3, 'elitism_rate': 0.1} 
	# - crossover rate is useless because what purpose is there to randomly change between init_values? none. it's random and does not carry over information.
	# hyperparams = {'generations': 2, 'pop_size': 2, 'mutation_rate': 0.1, 'elitism_rate': 0.1, 'structure_rate': 0.1}

	# to test structure mutations, and their crossover
	# hyperparams = {'generations': 3, 'pop_size': 3, 'mutation_rate': 1.0, 'elitism_rate': 0.1, 'structure_rate': 1.0}

	 # to test layer mutations, and their crossover
	# hyperparams = {'generations': 300, 'pop_size': 3, 'mutation_rate': 1.0, 'elitism_rate': 0.1, 'structure_rate': 0.0}

	# Actually test algorithm
	hyperparams = {'generations': 300, 'pop_size': 150, 'mutation_rate': 0.3, 'elitism_rate': 0.1, 'structure_rate': 0.1}

	# Build the model/pop
	# Here is the LSTM-ready array with a shape of (100 samples, 5 time steps, 1 feature)
	# make_random_model(output_dim, input_shape, model, random_init_values(), type="uni")
	# make_random_model(output_dim, input_shape, model, random_init_values(), random_init_values(), type="cascaded")
	# model.summary()

	# binary specifications are useless if I don't perturb by bit
	# pop_data = {0: population, 1: layers, 2: model_specifications, 3: pop_binary_specifications, 4: m_type, 5: pop_size, 6: input_shapes}
	pop_data = init_pop(output_dim, input_shape, m_type="uni", pop_size=hyperparams['pop_size'])
	pop_data[0][0].summary()



	X = x_train
	y = y_train
	pop_data = train(X, y, pop_data, hyperparams)# , epochs=tf.constant(10, dtype=tf.int64))

	pop_data[0][0].summary()

	# quit(0)

	# scores = test(x_test, y_test)
	# scores = test(X, y, pop_data[0][0])
	# print(scores)
	# print('Accuracy: {}'.format(scores[1]))
	# hist = model.fit(X, y) # hist['loss']


	# import matplotlib.pyplot as plt
	# predict=model.predict(X)
	# plt.plot(y, abs(predict-y), 'C2')
	# plt.ylim(ymax = 10, ymin = -1)
	# plt.show()

