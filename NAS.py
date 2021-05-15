#
# @author - zac-j-harris
#

import random
from keras.models import Sequential
from math import ceil
from Model import Model
# from lstm import mutate_init_values
import random, logging

logger = logging.getLogger("NAS")




def make_pop(output_dim=None, input_shapes=None, layer_types=None, layer_specs=None, pop_size=10, m_type=None, population=None):
	"""
		Returns an array of models consisting of a single LSTM layer followed by a single Dense layer.
	"""
	if population is None:
		population = [Model(output_dim=output_dim, model=None, layer_types=layer_types, layer_specs=layer_specs, model_type=m_type, input_shapes=input_shapes) for _ in range(pop_size)]

	return population




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
	# logger.info(elites)
	return elites, above_average




# for structure itself, take initial input, and then mutate new structures that cannot be compared to others

def crossover(population, h_params, fitness, input_shape=(3, 1024)):
	# population = {0: population, 1: layer_types, 2: layer_specs, 3: pop_binary_specifications, 4: population['m_type'], 5: population['pop_size'], 6: input_shapes}
	# h_params = {'generations': 1, 'pop_size': 10, 'crossover_rate': 0.9, 'mutation_rate': 0.3, 'elitism_rate': 0.1}
	# if model has above mean fitness, add it to crossover array of indices (random selection for parents), NEAT crossover methods or coin toss? (try both?)
	# Currently, we randomly take matching layer types from either parent
	# Non-matching layer_types are taken from more fit parent


	num_elites = max(int(h_params['pop_size'] * h_params['elitism_rate']), 1)

	elites, above_average = get_elites( num_elites, h_params['pop_size'], fitness)
	
	# New data for crossed over population
	for i in range(h_params['pop_size']):
		if i in above_average:
			continue
		
		parent1 = population[random.choice(above_average)]
		parent2 = population[i]

		parent2.crossover(parent1)


	
	return population, num_elites




def mutation(population, h_params, num_elites):
	# population = {0: population, 1: layer_types, 2: layer_specs, 3: pop_binary_specifications, 4: m_type, 5: pop_size, 6: input_shapes}
	# h_params = {'generations': 1, 'pop_size': 10, 'crossover_rate': 0.9, 'mutation_rate': 0.3, 'elitism_rate': 0.1}

	for model_i in range(num_elites, h_params['pop_size']):
		population[model_i].mutate(h_params)
	return population



if __name__ == "__main__":
	pass


