#
#  @author: zac-j-harris 2021
#


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
	population = init_pop(base_output_dim, input_shape=(1, state_size), m_type="uni",
	                      pop_size=hyperparameters['pop_size'])

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
