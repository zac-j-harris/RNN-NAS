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
import random, logging

logger = logging.getLogger("NAS")

# def dec_to_bin(dec, buf_len):
# 	pass


# def pop_specs_to_bin(layers, specs):
# 	bin_list = [['' for __ in range(len(layers[i]))] for i in range(len(layers))]
# 	buf = [3,3,2,3,7,3,3,2,3,9]
# 	for model_i in range(len(layers)):
# 		model = layers[model_i]
# 		for layer_i in range(len(model)):
# 			if model[layer_i] == '3':
# 				b_str = ''
# 				for i in range(5):
# 					b_str += dec_to_bin(specs[model_i][layer_i][i], buf[i+5])[2:]
# 			else:
# 				b_str = '00000'
# 				for i in range(5):
# 					b_str += dec_to_bin(specs[model_i][layer_i][i], buf[i])[2:]

def make_model(model_i, model=None, input_shapes=None, layers=None, model_specifications=None):
	model = Sequential() if model == None else model
	for layer_i in range(len(layers[model_i])):
		not_final_layer = (layer_i != len(layers[model_i]) - 1)
		layer = layers[model_i][layer_i]
		if layer == 0:
			model.add(make_uni_LSTM(model_specifications[model_i][layer_i][4], input_shapes[model_i][layer_i], init_values=model_specifications[model_i][layer_i], return_sequences=not_final_layer)) #True = many to many
		elif layer == 1:
			model.add(make_bi_LSTM(model_specifications[model_i][layer_i][4], input_shapes[model_i][layer_i], init_values=model_specifications[model_i][layer_i], return_sequences=not_final_layer))
		elif layer == 2:
			(a, b) = make_cascaded_LSTM(model_specifications[model_i][layer_i][4], input_shapes[model_i][layer_i], init_values=model_specifications[model_i][layer_i], return_sequences=not_final_layer)
			model.add(a)
			model.add(b)
		elif layer == 3:
			model.add(make_Dense(model_specifications[model_i][layer_i][4], init_values=model_specifications[model_i][layer_i])) # pop_spec does have a value, because it's never not created
	model.compile(loss="mean_absolute_error",optimizer ='adam',metrics=['accuracy'])

	return model


def make_pop(output_dim=None, input_shapes=None, layers=None, model_specifications=None, pop_binary_specifications=None, pop_size=10, m_type=None):
	'''
		Returns an array of models consisting of a single LSTM layer followed by a single Dense layer.
	'''
	m_type_dict = {'uni': 0,'bi': 1, 'cascaded': 2}
	model_type = m_type_dict[m_type]
	population = [Sequential() for _ in range(pop_size)]

	if model_specifications != None:
		# TODO: build the pop according to specifications.
		for model_i in range(len(population)):
			model = population[model_i]
			population[model_i] = make_model(model_i, model=model, input_shapes=input_shapes, layers=layers, model_specifications=model_specifications)
	else:
		new_input_shapes = [[input_shapes, (None, output_dim)] for _ in range(pop_size)]
		layers = [(model_type, 3) for _ in range(pop_size)]
		model_specifications = [(random_init_values(output_dim=output_dim), random_init_values("sigmoid", "normal", None, output_dim=output_dim)) for _ in range(pop_size)]
		for model_i in range(pop_size):
			model = population[model_i]
			if m_type=="uni":
				model.add(make_uni_LSTM(model_specifications[model_i][0][4], input_shapes, init_values=model_specifications[model_i][0], return_sequences=False)) #True = many to many
			elif m_type == "bi":
				model.add(make_bi_LSTM(model_specifications[model_i][0][4], input_shapes, init_values=model_specifications[model_i][0], return_sequences=False))
			elif m_type == "cascaded":
				(a, b) = make_cascaded_LSTM(model_specifications[model_i][0][4], input_shapes, init_values=model_specifications[model_i][0], return_sequences=False)
				model.add(a)
				model.add(b)
			model.add(make_Dense(model_specifications[model_i][0][4], init_values=model_specifications[model_i][1]))
			# model.compile(loss='mse',optimizer ='adam',metrics=['accuracy'])
			model.compile(loss="mean_absolute_error",optimizer ='adam',metrics=['accuracy'])

		# TODO: make pop_binary_specifications
		# output_dims = new_output_dims
		input_shapes = new_input_shapes
		# pop_binary_specifications = pop_specs_to_bin(layers, model_specifications)


	return {0: population, 1: layers, 2: model_specifications, 3: pop_binary_specifications, 4: m_type, 5: pop_size, 6: input_shapes}



# for structure itself, take initial input, and then mutate new structures that cannot be compared to others



def crossover(pop_data, hyperparams, fitness):
	# pop_data = {0: population, 1: layers, 2: model_specifications, 3: pop_binary_specifications, 4: pop_data[4], 5: pop_data[5], 6: input_shapes}
	# hyperparams = {'generations': 1, 'pop_size': 10, 'crossover_rate': 0.9, 'mutation_rate': 0.3, 'elitism_rate': 0.1}
	# if model has above mean fitness, add it to crossover array of indices (random selection for parents), NEAT crossover methods or coin toss? (try both?)
	pop_size = hyperparams['pop_size']
	mean_fitness = sum(fitness) / pop_size
	num_elites = int(pop_size * hyperparams['elitism_rate'])
	elites = [0 for _ in range(num_elites)]
	half_pop_size = pop_size // 2 + 1 if pop_size // 2 != pop_size / 2 else pop_size // 2
	above_average = [0 for _ in range(half_pop_size)]
	num_elites = 0
	current_max = 0
	# Not a very efficient calculation, but population sizes aren't large
	while num_elites < len(above_average):
		for f_i in range(pop_size):
			f = fitness[f_i]
			if f >= fitness[current_max] and f_i not in above_average:
				current_max = f_i
		if num_elites < len(elites):
			elites[num_elites] = current_max
		above_average[num_elites] = current_max
		num_elites += 1
	
	population = [Sequential() for i in range(pop_size)]
	layers = [[0 for _ in range(len(pop_data[2][i]))] for i in range(pop_size)]
	model_specifications = [[(0,0,0,0,0) for _ in range(len(pop_data[2][i]))] for i in range(pop_size)]
	# pop_binary_specifications = [[] for i in range(pop_size)]
	pop_binary_specifications = None
	input_shapes = [[(0, 1) for _ in range(len(pop_data[2][i]))] for i in range(pop_size)]
	# output_dims = [[] for i in range(pop_size)]

	def add_data(new_index, old_index, layer_i=None):
		if layer_i != None:
			# population[new_index] = pop_data[0][old_index]
			layers[new_index][layer_i] = pop_data[1][old_index][layer_i]
			model_specifications[new_index][layer_i] = pop_data[2][old_index][layer_i]
			# pop_binary_specifications[new_index] = pop_data[3][old_index][layer_i]
			# output_dims[new_index] = pop_data[5][old_index][layer_i]
			input_shapes[new_index][layer_i] = pop_data[6][old_index][layer_i]
		else:
			population[new_index] = pop_data[0][old_index]
			layers[new_index] = pop_data[1][old_index]
			model_specifications[new_index] = pop_data[2][old_index]
			# pop_binary_specifications[new_index] = pop_data[3][old_index]
			# output_dims[new_index] = pop_data[5][old_index]
			input_shapes[new_index] = pop_data[6][old_index]

	# Carry elites over from previous gen
	for i in range(len(elites)):
		add_data(i, elites[i])
		
	# New data for crossed over population
	for i in range(len(elites), pop_size):
		p1 = random.choice(above_average)
		p2 = random.choice(above_average)
		if fitness[p1] < fitness[p2]:
			temp = p1
			p1 = p2
			p2 = temp
		smallest_len = len(pop_data[1][p1]) if len(pop_data[1][p1]) < len(pop_data[1][p2]) else len(pop_data[1][p2])
		final_ind = 0
		population[i] = pop_data[0][p1] # this moves p1 as a base to the new index in population
		for layer_i in range(smallest_len):
			final_ind = layer_i
			if pop_data[1][p1][layer_i] == pop_data[1][p2][layer_i]:
				# Take from fit parent for non-matching genes, otherwise random
				# Here we take random
				if random.random() < 0.5:
					add_data(i, p1, layer_i=layer_i)
				else:
					add_data(i, p2, layer_i=layer_i)
			else:
				break
		# Now we take from p1
		if final_ind < len(pop_data[1][p1]):
			for layer_i in range(final_ind, len(pop_data[1][p1])):
				add_data(i, p1, layer_i=layer_i)


	new_pop_data = {0: population, 1: layers, 2: model_specifications, 3: pop_binary_specifications, 4: pop_data[4], 5: pop_data[5], 6: input_shapes}
	return new_pop_data



def mutate(model_i, layer_i, pop_data, hyperparams):
	
	def add_layer(pop_data, model_i, layer_i, layer_type):
		'''
			Updates pop_data with new layer, but does not make new model
		'''
		output_dim = pop_data[2][model_i][layer_i][4]
		num_layers = len(pop_data[2][model_i])
		old_input_shape = pop_data[6][model_i][layer_i]
		new_input_shape = (old_input_shape[1], output_dim)
		is_last_layer = layer_i == num_layers - 1
		
		layers = [[] for _ in range(num_layers + 1)]
		specs = [('','','',0.0,0) for _ in range(num_layers + 1)]
		input_shapes = [(0, 0) for _ in range(num_layers + 1)]
		delta = 0
		for i in range(num_layers + 1):
			if i == layer_i + 1:
				delta = 1
				layers[i] = layer_type
				# specs[i] = random_init_values(output_dim=int(random.random() * 10 * output_dim))
				specs[i] = random_init_values()
				input_shapes[i] = new_input_shape
			else:
				if i == layer_i + 2:
					input_shapes[i] = (input_shapes[i-1][1], specs[i-1][4])
				else:
					input_shapes[i] = pop_data[6][model_i][i - delta]
				layers[i] = pop_data[1][model_i][i - delta]
				specs[i] = pop_data[2][model_i][i - delta]
				
		pop_data[1][model_i] = layers
		pop_data[2][model_i] = specs
		pop_data[6][model_i] = input_shapes
		return pop_data

	if random.random() < hyperparams['structure_rate']:
		'''
			Everything changes: change model specs, input shapes, layers, then remake model to change population
		'''
		m_type_dict = {'uni': 0,'bi': 1, 'cascaded': 2}
		model_type = m_type_dict[pop_data[4]]
		l_type = random.choice([model_type, 3])

		pop_data = add_layer(pop_data, model_i, layer_i, l_type)

		# model = make_model(model_i, input_shapes=pop_data[6], layers=pop_data[1], model_specifications=pop_data[2])
		# pop_data[0][model_i] = model

	else:
		'''
			Only model_specifications and the model itself changes. Everything else stays the same. 
			This is because model_specs shows each layer's composition, and we are changing a single layers' composition.
		'''
		change = random.choice([0,1,2,3,4])
		# change = random.choice([0,1,2,3])
		layer_specs = [i for i in pop_data[2][model_i][layer_i]]
		layer_specs[change] = None
		# logger.debug(layer_specs)
		# (activation, initializer, constraint, dropout, output_dim)

		new_init_values = random_init_values(activation=layer_specs[0], initializer=layer_specs[1], constraint=layer_specs[2], 
															dropout=layer_specs[3], output_dim=layer_specs[4])
		if change == 4:
			logger.debug(new_init_values)
		pop_data[2][model_i][layer_i] = new_init_values
		# quit(0)
		
		# model = make_model(model_i, input_shapes=pop_data[6], layers=pop_data[1], model_specifications=pop_data[2])
		# pop_data[0][model_i] = model
		# pop_data[1][model_i][layer_i] = pop_data[1][old_index]
		# pop_data[2][model_i][layer_i] = pop_data[2][old_index]
		# pop_data[6][model_i][layer_i] = pop_data[6][old_index]
	return pop_data

def mutation(pop_data, hyperparams):
	# pop_data = {0: population, 1: layers, 2: model_specifications, 3: pop_binary_specifications, 4: m_type, 5: pop_size, 6: input_shapes}
	# hyperparams = {'generations': 1, 'pop_size': 10, 'crossover_rate': 0.9, 'mutation_rate': 0.3, 'elitism_rate': 0.1}
	for model_i in range(hyperparams['pop_size']):
		for layer_i in range(len(pop_data[1][model_i])):
			if random.random() < hyperparams['mutation_rate']:
				pop_data = mutate(model_i, layer_i, pop_data, hyperparams)
	return pop_data



if __name__ == "__main__":
	pass


