#
# @author - zac-j-harris
#

import random
from keras.models import Sequential
from lstm import make_uni_LSTM
from lstm import make_bi_LSTM
from lstm import make_cascaded_LSTM
from lstm import make_Dense
from lstm import random_init_values
from math import ceil
# from lstm import mutate_init_values
import random, logging

logger = logging.getLogger("NAS")

# def dec_to_bin(dec, buf_len):
# 	pass
# TODO: test

# def pop_layer_specs_to_bin(layer_types, layer_specs):
# 	bin_list = [['' for __ in range(len(layer_types[i]))] for i in range(len(layer_types))]
# 	buf = [3,3,2,3,7,3,3,2,3,9]
# 	for model_i in range(len(layer_types)):
# 		model = layer_types[model_i]
# 		for layer_i in range(len(model)):
# 			if model[layer_i] == '3':
# 				b_str = ''
# 				for i in range(5):
# 					b_str += dec_to_bin(layer_specs[model_i][layer_i][i], buf[i+5])[2:]
# 			else:
# 				b_str = '00000'
# 				for i in range(5):
# 					b_str += dec_to_bin(layer_specs[model_i][layer_i][i], buf[i])[2:]

def make_model(model_i, model=None, input_shapes=None, layer_types=None, layer_specs=None):
	# TODO: test2
	model = Sequential() if model is None else model
	for layer_i in range(len(layer_types[model_i])):
		not_final_layer = (layer_i != len(layer_types[model_i]) - 1)
		layer = layer_types[model_i][layer_i]
		if layer == 0:
			model.add(make_uni_LSTM(layer_specs[model_i][layer_i][4], input_shapes[model_i][layer_i], init_values=layer_specs[model_i][layer_i], return_sequences=not_final_layer)) #True = many to many
		elif layer == 1:
			model.add(make_bi_LSTM(layer_specs[model_i][layer_i][4], input_shapes[model_i][layer_i], init_values=layer_specs[model_i][layer_i], return_sequences=not_final_layer))
		elif layer == 2:
			(a, b) = make_cascaded_LSTM(layer_specs[model_i][layer_i][4], input_shapes[model_i][layer_i], init_values=layer_specs[model_i][layer_i], return_sequences=not_final_layer)
			model.add(a)
			model.add(b)
		elif layer == 3:
			model.add(make_Dense(layer_specs[model_i][layer_i][4], init_values=layer_specs[model_i][layer_i])) # pop_spec does have a value, because it's never not created
	model.compile(loss="mean_absolute_error", optimizer='adam', metrics=['accuracy'])

	return model


def make_pop(output_dim=None, input_shapes=None, layer_types=None, layer_specs=None, pop_size=10, m_type=None):
	"""
		Returns an array of models consisting of a single LSTM layer followed by a single Dense layer.
	"""
	m_type_dict = {'uni': 0,'bi': 1, 'cascaded': 2}
	model_type = m_type_dict[m_type]
	models = [Sequential() for _ in range(pop_size)]

	if layer_specs is not None:
		# TODO: build the pop according to specifications.
		for model_i in range(len(models)):
			model = models[model_i]
			models[model_i] = make_model(model_i, model=model, input_shapes=input_shapes, layer_types=layer_types, layer_specs=layer_specs)
	else:
		new_input_shapes = [[input_shapes, (None, output_dim)] for _ in range(pop_size)]
		layer_types = [[model_type, 3] for _ in range(pop_size)]
		layer_specs = [[random_init_values(output_dim=output_dim*10), random_init_values("sigmoid", "normal", None, output_dim=output_dim)] for _ in range(pop_size)]
		for model_i in range(pop_size):
			model = models[model_i]
			if m_type=="uni":
				model.add(make_uni_LSTM(layer_specs[model_i][0][4], input_shapes, init_values=layer_specs[model_i][0], return_sequences=False)) #True = many to many
			elif m_type == "bi":
				model.add(make_bi_LSTM(layer_specs[model_i][0][4], input_shapes, init_values=layer_specs[model_i][0], return_sequences=False))
			elif m_type == "cascaded":
				(a, b) = make_cascaded_LSTM(layer_specs[model_i][0][4], input_shapes, init_values=layer_specs[model_i][0], return_sequences=False)
				model.add(a)
				model.add(b)
			model.add(make_Dense(layer_specs[model_i][1][4], init_values=layer_specs[model_i][1]))
			# model.add(make_Dense(layer_specs[model_i][2][4], init_values=layer_specs[model_i][2] ))
			# model.compile(loss='mse',optimizer ='adam',metrics=['accuracy'])
			model.compile(loss="mean_absolute_error",optimizer ='adam',metrics=['accuracy'])

		# TODO: make pop_binary_specifications
		# output_dims = new_output_dims
		input_shapes = new_input_shapes
		# pop_binary_specifications = pop_layer_specs_to_bin(layer_types, layer_specs)
	return {'models': models, 'layer_types': layer_types, 'layer_specs': layer_specs, # 'bin_layer_specs': pop_binary_specifications, 
			'm_type': m_type, 'pop_size': pop_size, 'input_shapes': input_shapes}




def get_elites(num_elites, pop_size, fitness):
	"""
		Returns elites, and upper half of population
	"""
	elites = [0 for _ in range(num_elites)]
	half_pop_size = pop_size // 2 + 1 if pop_size // 2 != pop_size / 2 else pop_size // 2
	above_average = [0 for _ in range(max(half_pop_size, 1 )) ]
	num_elites = 0
	min_fit = ceil(round(min(fitness), 3))
	# Not a very efficient calculation, but population sizes aren't large
	while num_elites < len(above_average):
		current_max = 0
		for fit_ind in range(pop_size):
			if (fitness[fit_ind] > fitness[current_max] and fit_ind not in above_average) or ceil(round(fitness[current_max], 3)) == min_fit:
				current_max = fit_ind
		if num_elites < len(elites):
			elites[num_elites] = current_max
		above_average[num_elites] = current_max
		num_elites += 1
	logger.info(elites)
	return elites, above_average




# for structure itself, take initial input, and then mutate new structures that cannot be compared to others

def crossover(population, h_params, fitness, input_shape=(3, 1024)):
	# population = {0: population, 1: layer_types, 2: layer_specs, 3: pop_binary_specifications, 4: population['m_type'], 5: population['pop_size'], 6: input_shapes}
	# h_params = {'generations': 1, 'pop_size': 10, 'crossover_rate': 0.9, 'mutation_rate': 0.3, 'elitism_rate': 0.1}
	# if model has above mean fitness, add it to crossover array of indices (random selection for parents), NEAT crossover methods or coin toss? (try both?)
	# Currently, we randomly take matching layer types from either parent
	# Non-matching layer_types are taken from more fit parent



	elites, above_average = get_elites( max(int(h_params['pop_size'] * h_params['elitism_rate']), 1), 
										h_params['pop_size'], fitness)
	
	layer_types = [[0 for _ in range(len(population['layer_specs'][i]))] for i in range(h_params['pop_size'])]
	layer_specs = [ [(0,0,0,0,0) for _ in range(len(population['layer_specs'][i]))] for i in range(h_params['pop_size'])]
	# pop_binary_specifications = None
	input_shapes = [[(0, input_shape[0]) for _ in range(len(population['layer_specs'][i]))] for i in range(h_params['pop_size'])] # input shapes?

	def add_data(new_index, old_index, layer_types, layer_specs, input_shapes, layer_i=None):
		if layer_i is not None:
			layer_types[new_index][layer_i] = population['layer_types'][old_index][layer_i]
			layer_specs[new_index][layer_i] = population['layer_specs'][old_index][layer_i]
			input_shapes[new_index][layer_i] = population['input_shapes'][old_index][layer_i]
		else:
			layer_types[new_index] = population['layer_types'][old_index]
			layer_specs[new_index] = population['layer_specs'][old_index]
			input_shapes[new_index] = population['input_shapes'][old_index]
		return layer_types, layer_specs, input_shapes

	# Carry elites over from previous gen
	for i in range(len(elites)):
		layer_types, layer_specs, input_shapes = add_data(i, elites[i], layer_types, layer_specs, input_shapes)
		
	# New data for crossed over population
	for i in range(len(elites), h_params['pop_size']):
		parent1 = random.choice(above_average)
		parent2 = random.choice(above_average)
		if fitness[parent1] < fitness[parent2]:
			temp = parent1
			parent1 = parent2
			parent2 = temp
		smallest_num_layers = len(population['layer_types'][parent1]) if len(population['layer_types'][parent1]) < len(population['layer_types'][parent2]) else len(population['layer_types'][parent2])
		# population[i] = population['models'][parent1] # this moves parent1 as a base to the new index in population
		layer_types, layer_specs, input_shapes = add_data(i, parent1, layer_types, layer_specs, input_shapes)
		for layer_ind in range(smallest_num_layers-1):
			if population['layer_types'][parent1][layer_ind] == population['layer_types'][parent2][layer_ind] and random.random() < 0.5:
				# Take from fit parent for non-matching genes, otherwise random
				# Here we take random
				layer_types, layer_specs, input_shapes = add_data(i, parent2, layer_types, layer_specs, input_shapes, layer_i=layer_ind)
			else:
				break
		# Now we take from parent1 (the parent with higher fitness)
		# if final_ind < len(population['layer_types'][parent1]):
		# 	for layer_ind in range(final_ind, len(population['layer_types'][parent1])):
		# 		layer_types, layer_specs, input_shapes = add_data(i, parent1, layer_types, layer_specs, input_shapes, layer_i=layer_ind)


	new_population = {'models': population['models'], 'layer_types': layer_types, 'layer_specs': layer_specs, # 'bin_layer_specs': pop_binary_specifications, 
					'm_type': population['m_type'], 'pop_size': population['pop_size'], 'input_shapes': input_shapes}
	# print(layer_specs)
	return new_population, max(int(h_params['pop_size'] * h_params['elitism_rate']), 1)



def add_layer(population, model_i, layer_i, layer_type):
	"""
		Updates population with new layer, but does not make new model
	"""
	output_dim = population['layer_specs'][model_i][layer_i][4]
	num_layer_types = len(population['layer_specs'][model_i])
	old_input_shape = population['input_shapes'][model_i][layer_i]
	new_input_shape = (old_input_shape[1], output_dim)

	layer_types = [[] for _ in range(num_layer_types + 1)]
	layer_specs = [('', '', '', 0.0, 0) for _ in range(num_layer_types + 1)]
	input_shapes = [(0, 0) for _ in range(num_layer_types + 1)]
	delta = 0
	for i in range(num_layer_types + 1):
		if i == layer_i + 1:
			delta = 1
			layer_types[i] = layer_type
			# layer_specs[i] = random_init_values(output_dim=int(random.random() * 10 * output_dim))
			layer_specs[i] = random_init_values()
			input_shapes[i] = new_input_shape
		else:
			if i == layer_i + 2:
				input_shapes[i] = (input_shapes[i - 1][1], layer_specs[i - 1][4])
			else:
				input_shapes[i] = population['input_shapes'][model_i][i - delta]
			layer_types[i] = population['layer_types'][model_i][i - delta]
			layer_specs[i] = population['layer_specs'][model_i][i - delta]
			if layer_specs[i][0] == 0:
				logger.error(layer_specs[i])
				quit(-1)

	population['layer_types'][model_i] = layer_types
	population['layer_specs'][model_i] = layer_specs
	population['input_shapes'][model_i] = input_shapes
	return population



def mutate(model_i, layer_i, population, h_params, base_output_dim):

	if random.random() < h_params['structure_rate']:
		'''
			Structure is added instead of altering the model's existing architecture
			Everything changes: change model layer_specs, input shapes, layer_types, then remake model to change population
		'''
		m_type_dict = {'uni': 0,'bi': 1, 'cascaded': 2}
		model_type = m_type_dict[population['m_type']]
		l_type = random.choice([model_type, 3])

		population = add_layer(population, model_i, layer_i, l_type)


	else:
		"""
			Only layer_specs and the model itself changes. Everything else stays the same. 
			This is because model_layer_specs shows each layer's composition, and we are changing a single layer_types' composition.
			TODO: instead of random values, look into minor adjustments
		"""
		change = random.choice([0,1,2,3,4])
		# change = 4
		layer_specs = [i for i in population['layer_specs'][model_i][layer_i]]
		logger.debug(layer_specs)
		if change == 4:
			current = layer_specs[change]
			current = max( max( int((random.random() * 2.0 - 1.0) * h_params['mutation_percentage'] * current + current), 1), base_output_dim)
			layer_specs[4] = current
		elif change == 2:
			layer_specs[change] = 0
		else:
			layer_specs[change] = None

		new_init_values = random_init_values(activation=layer_specs[0], initializer=layer_specs[1], constraint=layer_specs[2], 
															dropout=layer_specs[3], output_dim=layer_specs[4])
		# logger.debug(new_init_values)
		# if new_init_values[0] == 0:
		# 	logger.error(new_init_values)
		# 	quit(-1)
		population['layer_specs'][model_i][layer_i] = new_init_values

	return population



def mutation(population, h_params, num_elites, base_output_dim):
	# population = {0: population, 1: layer_types, 2: layer_specs, 3: pop_binary_specifications, 4: m_type, 5: pop_size, 6: input_shapes}
	# h_params = {'generations': 1, 'pop_size': 10, 'crossover_rate': 0.9, 'mutation_rate': 0.3, 'elitism_rate': 0.1}

	for model_i in range(num_elites, h_params['pop_size']):
		for layer_i in range(len(population['layer_types'][model_i])-1):
			if random.random() < h_params['mutation_rate']:
				population = mutate(model_i, layer_i, population, h_params, base_output_dim)
	return population



if __name__ == "__main__":
	pass


