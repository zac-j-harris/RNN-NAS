#
# @author - zac-j-harris
#

import random
from tensorflow.keras.models import Sequential
from math import ceil
# from lstm import mutate_init_values
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Reshape, Flatten
from tensorflow.keras.models import Sequential
# import tensorflow as tf
import random, logging

logger = logging.getLogger("Model")

class Model():

	m_type_dict = {'uni': 0,'bi': 1, 'cascaded': 2, 'conv': 3, 'dense': 4, 'flat': 5}



	def __init__(self, output_dim, model=None, layer_types=None, layer_specs=None, model_type=None, input_shapes=None):
		self.model = Sequential() if model is None else model
		self.model_type = model_type
		self.base_output_dim = output_dim

		if layer_specs == None:
			first_inp = (None, input_shapes[0], input_shapes[1])
			self.input_shapes = [first_inp, input_shapes] # , input_shapes]
			self.layer_types = [self.m_type_dict[self.model_type], self.m_type_dict['dense']] # , self.m_type_dict['flat']] #, self.m_type_dict['dense']]
			self.layer_specs = [self.random_init_values(output_dim=self.base_output_dim), 
			self.random_init_values("sigmoid", "normal", None, output_dim=self.base_output_dim)] # , self.base_output_dim] #, 
			# self.random_init_values("sigmoid", "normal", None, output_dim=self.base_output_dim)]
		else:
			self.input_shapes, self.layer_types, self.layer_specs = input_shapes, layer_types, layer_specs


		for layer_i in range(len(self.layer_types)):
			# self.layer_types = layer_types
			layer = self.layer_types[layer_i]

			# not_final_layer = (layer_i != len(self.layer_types) - 1)
			

			if layer == self.m_type_dict['uni']:
				a, b = self.make_uni_LSTM(self.layer_specs[layer_i][4], self.input_shapes[layer_i], init_values=self.layer_specs[layer_i]) #, return_sequences=not_final_layer))
				# a = self.make_uni_LSTM(self.layer_specs[layer_i][4], self.input_shapes[layer_i],
				#                        init_values=self.layer_specs[layer_i])  # , return_sequences=not_final_layer))
				self.model.add(a)
				self.model.add(b)

			elif layer == self.m_type_dict['bi']:
				a, b = self.make_bi_LSTM(self.layer_specs[layer_i][4], self.input_shapes[layer_i], init_values=self.layer_specs[layer_i])
				self.model.add(a)
				self.model.add(b)
				# self.model.add(self.make_bi_LSTM(self.layer_specs[layer_i][4], self.input_shapes[layer_i], init_values=self.layer_specs[layer_i]))

			elif layer == self.m_type_dict['cascaded']:
				a, b, c, d = self.make_cascaded_LSTM(self.layer_specs[layer_i][4], self.input_shapes[layer_i], init_values=self.layer_specs[layer_i])
				self.model.add(a)
				self.model.add(b)
				self.model.add(c)
				self.model.add(d)


			elif layer == self.m_type_dict['dense']:
				self.model.add(self.make_Dense(self.layer_specs[layer_i][4], self.input_shapes[layer_i], init_values=self.layer_specs[layer_i]))
				# pop_spec does have a value, because it's never not created

			elif layer == self.m_type_dict['flat']:
				# self.model.add(Flatten())
				self.model.add(self.make_Flatten(self.input_shapes[layer_i], self.base_output_dim))
				# a, b = self.make_flatten(self.layer_specs[layer_i], self.input_shapes[layer_i])
				# self.model.add(a)
				# self.model.add(b)

		# self.get_summary(input_shapes)

		self.model.compile(loss="mean_absolute_error", optimizer='SGD', metrics=['accuracy'])

	def reinit(self):
		self.__init__(self.base_output_dim, layer_types=self.layer_types, layer_specs=self.layer_specs, model_type=self.model_type, input_shapes=self.input_shapes)


	def crossover(self, parent1):

		# Store old values
		old_input_shape, old_layer_specs, old_layer_types = self.input_shapes, self.layer_specs, self.layer_types

		# Set new values to default to parent1
		self.input_shapes, self.layer_specs, self.layer_types = parent1.get_input_shapes(), parent1.get_layer_specs(), parent1.get_layer_types()

		for layer_ind in range(len(self.layer_types)):
			if len(old_layer_types) > layer_ind:
				if parent1.get_layer_types()[layer_ind] == old_layer_types[layer_ind]:
					# Take from fit parent for non-matching genes, otherwise random
					if random.random() < 0.5:
						# Here we take random
						self.layer_types[layer_ind], self.layer_specs[layer_ind], self.input_shapes[layer_ind] = old_layer_types[layer_ind], old_layer_specs[layer_ind], old_input_shape[layer_ind]
				else:
					break
			else:
				break


	def get_summary(self, input_shape):
		self.model.build(input_shape=input_shape)
		self.model.summary()


	def mutate(self, h_params):
		# for layer_i in range(1, len(self.layer_types)-2):
		for layer_i in range(0, len(self.layer_types)-1):
			if random.random() < h_params['mutation_rate']:
				self.mutate_helper(layer_i, h_params, self.base_output_dim)


	def add_layer(self, layer_i, layer_type):
		"""
			Updates population with new layer, but does not make new model
		"""
		# if layer_i == 0:
		# 	print('layer = 0')
		# num_layer_types = len(population['layer_specs'][model_i])
		# old_input_shape = self.input_shapes[layer_i]
		old_layer_specs = self.layer_specs[layer_i]

		prior_layer_out = self.layer_specs[layer_i][4]
		# new_input_shape = tf.constant([1, prior_layer_out])
		new_input_shape = [1, prior_layer_out]
		# logger.debug(prior_layer_out)
		# new_layer_input_shape = [self.input_shapes[layer_i][1], self.base_output_dim]

		new_layer_types = [[] for _ in range(len(self.layer_types) + 1)]
		new_layer_specs = [['', '', '', 0.0, 0] for _ in range(len(self.layer_types) + 1)]
		new_input_shapes = [[0, 0] for _ in range(len(self.layer_types) + 1)]
		delta = 0
		for i in range(len(self.layer_types) + 1):
			if i == layer_i + 1:
				delta = 1
				new_layer_types[i] = layer_type
				new_layer_specs[i] = self.random_init_values()
				if layer_i == len(self.layer_types):
					new_layer_specs[i][4] = prior_layer_out
				new_input_shapes[i] = new_input_shape
				# new_input_shapes[i] = (1, new_layer_specs[i - 1][4])
			else:
				# if i == layer_i + 2:
					# new_input_shapes[i] = tf.constant((1, new_layer_specs[i - 1][4]))
					# new_input_shapes[i] = (1, new_layer_specs[i - 1][4])
					# new_input_shapes[i] = new_input_shape
				# else:
				new_input_shapes[i] = self.input_shapes[i - delta]
				new_layer_types[i] = self.layer_types[i - delta]
				new_layer_specs[i] = self.layer_specs[i - delta] # if i != layer_i else self.random_init_values()
				if new_layer_specs[i][0] == 0:
					logger.error(new_layer_specs[i])
					quit(-1)
		# if layer_i == 0:
		# 	new_layer_specs[layer_i] = old_layer_specs
		new_layer_specs[layer_i][4] = self.random_init_values()[4]
		self.layer_types = new_layer_types
		self.layer_specs = new_layer_specs
		self.input_shapes = new_input_shapes

		# logger.debug(self.model.summary())



	def mutate_helper(self, layer_i, h_params, base_output_dim):

		if random.random() < h_params['structure_rate']: # and population['layer_types'][model_i][layer_i] != m_type_dict['conv']:
			'''
				Structure is added instead of altering the model's existing architecture
				Everything changes: change model layer_specs, input shapes, layer_types, then remake model to change population
			'''

			l_type = random.choice([self.m_type_dict[self.model_type], self.m_type_dict['dense']])
			# l_type = self.m_type_dict['dense']
			# l_type = self.m_type_dict['uni']

			# if layer_i == 0:
			# 	self.add_layer(1, l_type)
			# else:
			self.add_layer(layer_i, l_type)

		else:
			"""
				Only layer_specs and the model itself changes. Everything else stays the same. 
				This is because model_layer_specs shows each layer's composition, and we are changing a single layer_types' composition.
				TODO: instead of random values, look into minor adjustments
			"""
			if layer_i == 0:
				return
			change = random.choice(range(1,5))
			# change = 4

			logger.debug(self.layer_specs[layer_i])
			if change == 4:
				current = self.layer_specs[layer_i][change]
				current = max( max( int((random.random() * 2.0 - 1.0) * h_params['mutation_percentage'] * current + current), 1), self.base_output_dim)
				self.layer_specs[layer_i][4] = current
			elif change == 2:
				self.layer_specs[layer_i][change] = 0
			else:
				self.layer_specs[layer_i][change] = None

			new_init_values = self.random_init_values(activation=self.layer_specs[layer_i][0], initializer=self.layer_specs[layer_i][1], constraint=self.layer_specs[layer_i][2], 
																dropout=self.layer_specs[layer_i][3], output_dim=self.layer_specs[layer_i][4])
			# logger.debug(new_init_values)
			# if new_init_values[0] == 0:
			# 	logger.error(new_init_values)
			# 	quit(-1)
			self.layer_specs[layer_i] = new_init_values


	def random_init_values(self, activation=None, initializer=None, constraint=None, dropout=None, output_dim=None):
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
			logger.debug("global2: " + str(self.base_output_dim))
		output_dim = int(random.random() * 10.0 * self.base_output_dim) + self.base_output_dim if output_dim is None else output_dim
		return [activation, initializer, constraint, dropout, output_dim]


	def make_uni_LSTM(self, output_dim, input_shape, init_values=None, return_sequences=False):
		init_values = self.random_init_values() if init_values is None else init_values
		# if input_shape[0] == None and len(input_shape) == 3:
		# 	input_shape = (input_shape[1], input_shape[2])
			# return Reshape(target_shape=input_shape, input_shape=input_shape), LSTM(output_dim, activation=init_values[0], kernel_initializer=init_values[1],
			# 		kernel_constraint=init_values[2], return_sequences=return_sequences, input_shape=input_shape)

		# return Reshape(target_shape=input_shape), LSTM(output_dim, activation=init_values[0], kernel_initializer=init_values[1],
		# 			kernel_constraint=init_values[2], return_sequences=return_sequences, input_shape=input_shape)
		# return Reshape(target_shape=input_shape), LSTM(output_dim, activation=init_values[0],
		#                                                kernel_initializer=init_values[1],
		#                                                kernel_constraint=init_values[2],
		#                                                return_sequences=return_sequences)

		target = (1, output_dim)
		return LSTM(output_dim, activation=init_values[0],
		                                          kernel_initializer=init_values[1],
		                                          kernel_constraint=init_values[2],
		                                          return_sequences=return_sequences),  Reshape(target_shape=target)
		# return LSTM(output_dim, activation=init_values[0], kernel_initializer=init_values[1],
		#             kernel_constraint=init_values[2],
		#             return_sequences=return_sequences)


	def make_bi_LSTM(self, output_dim, input_shape, init_values=None, return_sequences=False):
		init_values = self.random_init_values() if init_values is None else init_values
		if input_shape[0] == None and len(input_shape) == 3:
			input_shape = (input_shape[1], input_shape[2])
			return Reshape(target_shape=input_shape, input_shape=input_shape), Bidirectional(
				LSTM(output_dim, activation=init_values[0], kernel_initializer=init_values[1], kernel_constraint=init_values[2]), input_shape=input_shape)

		return Reshape(target_shape=input_shape), Bidirectional(
			LSTM(output_dim, activation=init_values[0], kernel_initializer=init_values[1], kernel_constraint=init_values[2]), input_shape=input_shape)


	def make_cascaded_LSTM(self, output_dim, input_shape, init_values=None, return_sequences=False):
		init_values = self.random_init_values() if init_values is None else init_values
		if input_shape[0] == None and len(input_shape) == 3:
			input_shape = (input_shape[1], input_shape[2])
			return Reshape(target_shape=input_shape, input_shape=input_shape), Bidirectional(
				LSTM(output_dim, activation=init_values[0], kernel_initializer=init_values[1], kernel_constraint=init_values[2]), input_shape=input_shape), 
			Reshape(target_shape=input_shape, input_shape=input_shape), LSTM(output_dim, activation=init_values[0], input_shape=input_shape, kernel_initializer=init_values[1], kernel_constraint=init_values[2])
		
		return Reshape(target_shape=input_shape), Bidirectional(
			LSTM(output_dim, activation=init_values[0], kernel_initializer=init_values[1], kernel_constraint=init_values[2]), input_shape=input_shape), 
		Reshape(target_shape=input_shape), LSTM(output_dim, activation=init_values[0], input_shape=input_shape, kernel_initializer=init_values[1], kernel_constraint=init_values[2])


	def make_Flatten(self, input_shape, output_dim):
		target_shape = (None, input_shape[0], output_dim)
		return Flatten(input_shape=target_shape)

	# def make_2d_cnn(self, output_dim, input_shape, init_values=None, return_sequences=False):
	# 	init_values = self.random_init_values() if init_values is None else init_values
	# 	new_input_shape = [int(input_shape[0] ** 0.5), int(input_shape[0] ** 0.5), input_shape[1]] if len(input_shape) == 2 else input_shape
	# 	return Reshape(target_shape=new_input_shape, input_shape=input_shape), Conv2D(filters=new_input_shape[2], kernel_size=1, activation=init_values[0], 
	# 		kernel_initializer=init_values[1], kernel_constraint=init_values[2]), Reshape(target_shape=(new_input_shape[1] ** 2, new_input_shape[2]))


	def make_Dense(self, output_dim, input_shape, init_values=None):
		init_values = self.random_init_values() if init_values is None else init_values
		# return Dense(output_dim, input_shape=input_shape, activation=init_values[0], kernel_initializer=init_values[1],
		# 			 kernel_constraint=init_values[2])
		return Dense(output_dim, activation=init_values[0], kernel_initializer=init_values[1],
		             kernel_constraint=init_values[2])

	def get_model(self):
		return self.model

	def get_layer_specs(self):
		return self.layer_specs

	def get_input_shapes(self):
		return self.input_shapes

	def get_layer_types(self):
		return self.layer_types


if __name__ == "__main__":
	pass


