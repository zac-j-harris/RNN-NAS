#
# @author - zac-j-harris
#

import random
from keras.models import Sequential
from lstm import make_uni_LSTM
from lstm import make_bi_LSTM
from lstm import make_cascaded_LSTM
from lstm import make_Dense
# from lstm import make_2d_cnn
from lstm import random_init_values
from math import ceil
# from lstm import mutate_init_values
import random, logging

logger = logging.getLogger("Model")

class Model():

	m_type_dict = {'uni': 0,'bi': 1, 'cascaded': 2, 'conv': 3, 'dense': 4}



	def __init__(self, output_dim, model=None, layer_types=None, layer_specs=None, model_type=None, input_shapes=None):
		self.model = Sequential() if model is None else model
		self.model_type = model_type
		self.base_output_dim = output_dim

		if layer_specs == None:
			self.input_shapes = [input_shapes, input_shapes, input_shapes]
			self.layer_types = [self.m_type_dict[self.model_type], self.m_type_dict['dense'], self.m_type_dict['dense']]
			self.layer_specs = [random_init_values(output_dim=self.base_output_dim*10), random_init_values("sigmoid", "normal", None, output_dim=self.base_output_dim), random_init_values("sigmoid", "normal", None, output_dim=self.base_output_dim)]
		else:
			self.input_shapes, self.layer_types, self.layer_specs = input_shapes, layer_types, layer_specs


		for layer_i in range(len(self.layer_types)):
			# self.layer_types = layer_types
			layer = self.layer_types[layer_i]

			not_final_layer = (layer_i != len(self.layer_types) - 1)
			

			if layer == self.m_type_dict['uni']:
				self.model.add(make_uni_LSTM(self.layer_specs[layer_i][4], self.input_shapes[layer_i], init_values=self.layer_specs[layer_i], return_sequences=not_final_layer)) 
				#True = many to many

			elif layer == self.m_type_dict['bi']:
				self.model.add(make_bi_LSTM(self.layer_specs[layer_i][4], self.input_shapes[layer_i], init_values=self.layer_specs[layer_i], return_sequences=not_final_layer))

			elif layer == self.m_type_dict['cascaded']:
				(a, b) = make_cascaded_LSTM(self.layer_specs[layer_i][4], self.input_shapes[layer_i], init_values=self.layer_specs[layer_i], return_sequences=not_final_layer)
				self.model.add(a)
				self.model.add(b)


			elif layer == self.m_type_dict['dense']:
				self.model.add(make_Dense(self.layer_specs[layer_i][4], self.input_shapes[layer_i], init_values=self.layer_specs[layer_i])) 
				# pop_spec does have a value, because it's never not created

		self.model.compile(loss="mean_absolute_error", optimizer='adam', metrics=['accuracy'])


	def crossover(self, parent1):

		old_input_shape, old_layer_specs, old_layer_types = self.input_shapes, self.layer_specs, self.layer_types

		self.layer_types = parent1.get_layer_types()
		self.layer_specs = parent1.get_layer_specs()
		self.input_shapes = parent1.get_input_shapes()


		for layer_ind in range(len(self.layer_types)):
			if parent1.get_layer_types()[layer_ind] == self.layer_types[layer_ind]:
				# Take from fit parent for non-matching genes, otherwise random
				if random.random() < 0.5:
					# Here we take random
					self.layer_types[layer_ind], self.layer_specs[layer_ind], self.input_shapes[layer_ind] = old_layer_types[layer_ind], old_layer_specs[layer_ind], old_input_shape[layer_ind]
			else:
				break




	def mutate(self, h_params):
		for layer_i in range(len(self.layer_types)-1):
			if random.random() < h_params['mutation_rate']:
				self.mutate_helper(layer_i, h_params, self.base_output_dim)


	def add_layer(self, layer_i, layer_type):
		"""
			Updates population with new layer, but does not make new model
		"""
		# num_layer_types = len(population['layer_specs'][model_i])
		# old_input_shape = self.input_shapes[layer_i]
		new_input_shape = [self.input_shapes[layer_i][0], self.input_shapes[layer_i][0], self.base_output_dim]
		new_layer_input_shape = [self.input_shapes[layer_i][1], self.base_output_dim]

		new_layer_types = [[] for _ in range(len(self.layer_types) + 1)]
		new_layer_specs = [['', '', '', 0.0, 0] for _ in range(len(self.layer_types) + 1)]
		new_input_shapes = [[0, 0] for _ in range(len(self.layer_types) + 1)]
		delta = 0
		for i in range(len(self.layer_types) + 1):
			if i == layer_i + 1:
				delta = 1
				new_layer_types[i] = layer_type
				new_layer_specs[i] = random_init_values()
				new_input_shapes[i] = new_input_shape
			else:
				if i == layer_i + 2:
					new_input_shapes[i] = (new_input_shapes[i - 1][1], new_layer_specs[i - 1][4])
				else:
					new_input_shapes[i] = self.input_shapes[i - delta]
				new_layer_types[i] = self.layer_types[i - delta]
				new_layer_specs[i] = self.layer_specs[i - delta]
				if new_layer_specs[i][0] == 0:
					logger.error(new_layer_specs[i])
					quit(-1)

		self.layer_types = new_layer_types
		self.layer_specs = new_layer_specs
		self.input_shapes = new_input_shapes



	def mutate_helper(self, layer_i, h_params, base_output_dim):

		if random.random() < h_params['structure_rate']: # and population['layer_types'][model_i][layer_i] != m_type_dict['conv']:
			'''
				Structure is added instead of altering the model's existing architecture
				Everything changes: change model layer_specs, input shapes, layer_types, then remake model to change population
			'''

			l_type = random.choice([self.model_type, 'dense'])

			self.add_layer(layer_i, l_type)


		else:
			"""
				Only layer_specs and the model itself changes. Everything else stays the same. 
				This is because model_layer_specs shows each layer's composition, and we are changing a single layer_types' composition.
				TODO: instead of random values, look into minor adjustments
			"""
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

			new_init_values = random_init_values(activation=self.layer_specs[layer_i][0], initializer=self.layer_specs[layer_i][1], constraint=self.layer_specs[layer_i][2], 
																dropout=self.layer_specs[layer_i][3], output_dim=self.layer_specs[layer_i][4])
			# logger.debug(new_init_values)
			# if new_init_values[0] == 0:
			# 	logger.error(new_init_values)
			# 	quit(-1)
			self.layer_specs[layer_i] = new_init_values


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


