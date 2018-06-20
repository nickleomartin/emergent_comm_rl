import random
import numpy as np
import time 
import json

import keras
from keras.models import Sequential, load_model, Model 
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Dense, Convolution2D, LSTM
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras import backend as K



# class BaseAgent(object):
# 	""" Parent agent """
# 	def __init__(self):
# 		pass

# 	def save_policy_networks(self):
# 		pass




class RandomAgent(object):
	"""
	Agent that randomly chooses messages and targets as a baseline

	Example
	-------
	from config import config_dict
	from data_generator import generate_dummy_data
	from networks import RandomAgent
	
	## Get training data
	train_data = generate_dummy_data()

	## Initialize and train agent
	ra = RandomAgent(config_dict)
	ra.fit(train_data)

	"""
	def __init__(self, config_dict, save_training_stats=True, save_testing_stats=True):
		self.config_dict = config_dict
		self.save_training_stats = save_training_stats
		self.save_testing_stats = save_testing_stats
		self.training_stats = None
		self.testing_stats = None
		self.initialize_parameters()

	def initialize_parameters(self):
		""" Assign config parameters to local vars """
		self.max_message_length = self.config_dict['max_message_length']
		self.alphabet = self.config_dict['alphabet']

	def speaker_policy(self,target_input,max_message_length=2):
		""" Randomly generate a message """
		return "".join([np.random.choice(self.alphabet) for i in range(self.max_message_length)])

	def listener_policy(self,message,candidates):
		""" Randomly choose a target """
		return np.random.randint(len(candidates))

	def calculate_reward(self, chosen_target_idx, target_candidate_idx):
		""" Determine reward given indices """
		if chosen_target_idx==target_candidate_idx:
			return 1
		else:
			return 0

	def fit(self, train_data):
		""" Random Sampling of messages and candidates for training"""
		self.training_stats = []
		total_reward = 0
		for target_input, candidates, target_candidate_idx in train_data:
			message = self.speaker_policy(target_input)
			chosen_target_idx = self.listener_policy(message,candidates)
			reward = self.calculate_reward(chosen_target_idx,target_candidate_idx)
			total_reward += reward

			if self.save_training_stats:
				self.training_stats.append({
											"reward": reward,
											"input": target_input,
											"message": message,
											"chosen_target": candidates[chosen_target_idx]
											})

	def predict(self,test_data):
		""" Random Sampling of messages and candidates for testing"""
		self.testing_stats = []
		total_reward = 0
		for target_input, candidates, target_candidate_idx in test_data:
			message = self.speaker_policy(target_input)
			chosen_target_idx = self.listener_policy(message,candidates)
			reward = self.calculate_reward(chosen_target_idx,target_candidate_idx)
			total_reward += reward

			if self.save_training_stats:
				self.testing_stats.append({
											"reward": reward,
											"input": target_input,
											"message": message,
											"chosen_target": candidates[chosen_target_idx]
											})



class DenseAgents(object):
	"""
	Two independent agents which use fully connected neural networks and 
	jointly optimize given only the reward

	Example
	-------
	from config import config_dict
	from data_generator import generate_dummy_data
	from networks import RandomAgent
	
	## Get training data
	train_data = generate_dummy_data()

	## Initialize and train agent
	ifca = DenseAgents(config_dict)
	ifca.fit(train_data)
	"""
	def __init__(self, config_dict, speaker, listener, save_training_stats=True, save_testing_stats=True):
		self.config_dict = config_dict
		self.speaker_model = speaker
		self.listener_model = listener
		self.save_training_stats = save_training_stats
		self.save_testing_stats = save_testing_stats
		self.training_stats = []
		self.testing_stats = []
		self.initialize_parameters()

	def initialize_parameters(self):
		""" Assign config parameters to local vars """
		self.max_message_length = self.config_dict['max_message_length']
		self.alphabet = self.config_dict['alphabet']
		self.alphabet_size = self.config_dict['alphabet_size']
		self.speaker_lr = self.config_dict['speaker_lr']
		self.speaker_dim = self.config_dict['speaker_dim']
		self.listener_lr = self.config_dict['listener_lr']
		self.listener_dim = self.config_dict['listener_dim']
		self.training_epoch = self.config_dict['training_epoch']
		self.batch_size = self.config_dict['batch_size']
		self.n_classes = self.config_dict['n_distractors'] + 1

	def calculate_reward(self, chosen_target_idx, target_candidate_idx):
		""" Determine reward given indices """
		print(chosen_target_idx, target_candidate_idx)
		if np.array_equal(chosen_target_idx, target_candidate_idx):
			return 1
		else:
			return 0


	def sample_from_networks_on_batch(self, target_input, candidates, target_candidate_idx):

		## Sample from speaker
		message, speaker_action, speaker_probs = self.speaker_model.sample_speaker_policy_for_message(target_input)
		print("Message: %s, Probs: %s"%(message, speaker_probs))

		## Sample from listener
		listener_action, listener_probs = self.listener_model.sample_from_listener_policy(message, candidates)

		## Calculate reward
		reward = self.calculate_reward(listener_action, target_candidate_idx)

		## Store batch inputs and outputs
		self.speaker_model.remember_speaker_training_details(target_input, speaker_action, speaker_probs, reward)
		self.listener_model.remember_listener_training_details(target_input, listener_action, listener_probs, reward)

		## Increment batch statistics
		self.total_training_reward += reward
		self.batch_counter += 1

		## Record training statistics
		if self.save_training_stats:
				self.training_stats.append({
											"reward": reward,
											"input": target_input,
											"message": message,
											"chosen_target": candidates[np.where(listener_action==1)[0][0]]
											})


	def train_networks_on_batch(self):
		""" """

		## Train speaker model 
		self.speaker_model.train_speaker_policy_on_batch()

		## Train listener model
		self.listener_model.train_listener_policy_on_batch()

		## Reset batch counter
		self.batch_counter = 0


	def fit(self, train_data):
		""" Random Sampling of messages and candidates for training"""
		self.total_training_reward = 0
		self.batch_counter = 0

		for target_input, candidates, target_candidate_idx in train_data:

			self.sample_from_networks_on_batch(target_input, candidates, target_candidate_idx)

			if self.batch_counter==self.batch_size:
				self.train_networks_on_batch()





	def predict(self,test_data):
		""" Random Sampling of messages and candidates for testing"""
		self.testing_stats = []
		total_reward = 0
		for target_input, candidates, target_candidate_idx in test_data:
			message, message_probs = self.speaker_model.infer_from_speaker_policy(target_input)

			print("Message: %s, Probs: %s"%(message,message_probs))

			chosen_target_idx = self.listener_model.infer_from_listener_policy(message,candidates)
			reward = self.calculate_reward(chosen_target_idx,target_candidate_idx)
			total_reward += reward

			if self.save_training_stats:
				self.testing_stats.append({
											"reward": reward,
											"input": target_input,
											"message": message,
											"chosen_target": candidates[np.where(chosen_target_idx==1)[0][0]]
											})
 

#################
## Legacy Code ##
#################

# class IndependentFullyConnectedAgents(object):
# 	"""
# 	Two independent agents which use fully connected neural networks and 
# 	jointly optimize given only the reward

# 	Example
# 	-------
# 	from config import config_dict
# 	from data_generator import generate_dummy_data
# 	from networks import RandomAgent
	
# 	## Get training data
# 	train_data = generate_dummy_data()

# 	## Initialize and train agent
# 	ifca = IndependentFullyConnectedAgents(config_dict)
# 	ifca.fit(train_data)
# 	"""
# 	def __init__(self, config_dict, save_training_stats=True, save_testing_stats=True):
# 		self.config_dict = config_dict
# 		self.save_training_stats = save_training_stats
# 		self.save_testing_stats = save_testing_stats
# 		self.training_stats = None
# 		self.testing_stats = None
# 		self.initialize_parameters()
# 		self.initialize_speaker_model()

# 	def initialize_parameters(self):
# 		""" Assign config parameters to local vars """
# 		self.max_message_length = self.config_dict['max_message_length']
# 		self.alphabet = self.config_dict['alphabet']
# 		self.alphabet_size = self.config_dict['alphabet_size']
# 		self.speaker_lr = self.config_dict['speaker_lr']
# 		self.speaker_dim = self.config_dict['speaker_dim']
# 		self.listener_lr = self.config_dict['listener_lr']
# 		self.listener_dim = self.config_dict['listener_dim']
# 		self.training_epoch = self.config_dict['training_epoch']
# 		self.batch_size = self.config_dict['batch_size']
# 		self.n_classes = self.config_dict['n_distractors'] + 1

# 	def initialize_speaker_model(self):
# 		""" 2 Layer fully-connected neural network """
# 		self.speaker_model = Sequential()
# 		self.speaker_model.add(Dense(self.speaker_dim, activation="relu", input_shape=(self.speaker_dim,)))
# 		self.speaker_model.add(Dense(self.alphabet_size,activation="softmax"))
# 		self.speaker_model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=self.speaker_lr))

# 	def initialize_listener_model(self):
# 		""" 2 Layer fully-connected neural network """
# 		self.listener_model = Sequential()
# 		self.listener_model.add(Dense(self.listener_dim, activation="relu", input_shape=(self.listener_dim,)))
# 		self.listener_model.add(Dense(self.n_classes,activation="softmax"))
# 		self.listener_model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=self.listener_lr))

# 	def sample_speaker_policy_for_message(self,target_input):
# 		""" Stochastically sample message of length self.max_message_length from speaker policy """ 

# 		t_input = target_input.reshape([1,self.speaker_dim])
# 		print(t_input.shape)

# 		probs = self.speaker_model.predict_on_batch(t_input)
# 		normalized_probs = probs/np.sum(probs)
# 		print("probs shape: ", probs.shape)

# 		message = ""
# 		message_probs = []
# 		for i in range(self.max_message_length):
# 			sampled_symbol = np.random.choice(self.alphabet_size, 1, p=normalized_probs[0])[0]
# 			message += str(sampled_symbol) + "#"
# 			message_probs.append(normalized_probs[0][sampled_symbol])
		
# 		## Return action and probs
# 		return message, message_probs

# 	def train_speaker_policy_on_batch(self, target_inputs, message_probs, rewards):
# 		""" Update speaker policy given rewards """
# 		## Calculate gradients = action - probs

# 		## Batch standardise rewards

# 		## Calculate gradients * rewards

# 		## Create X

# 		## Create Y = probs + lr * gradients

# 		## Train model

# 		## Reset states, probs, gradients, rewards = [], [], [], []

# 		self.m_probs = np.vstack(message_probs)
# 		self.r = np.vstack(rewards)
# 		self.X = np.squeeze(np.vstack(target_inputs))
# 		self.Y = self.r.flatten() * np.sum(self.m_probs,axis=1) 
# 		print("X.shape: %s , Y.shape: %s"%(self.X.shape,self.Y.shape))

# 		self.X_ = self.X.reshape([self.batch_size,self.speaker_dim])
# 		#self.Y_ = self.Y.reshape([self.batch_size,1])

# 		self.speaker_model.train_on_batch(self.X_, self.Y)
# 		print("Batch training complete")

# 	def infer_from_speaker_policy(self,target_input):
# 		""" Greedily obtain message from speaker policy """
# 		## Get symbol probabilities given target input
# 		probs = self.speaker_model.predict_on_batch(target_input.reshape([self.speaker_dim,1]),batch_size=1)
# 		normalized_probs = probs/np.sum(probs)

# 		## Greedy get symbols with largest probabilities
# 		argmax_indices = list(normalized_probs[0].argsort()[-self.max_message_length:][::-1])
# 		message_probs = normalized_probs[0][argmax_indices]
# 		message = "#".join([str(e) for e in list(argmax_indices)])
		
# 		## TODO: Also return sum[log prob () mi | target input and weights)]??
# 		return message, message_probs

# 	def listener_policy(self,message,candidates):
# 		""" Randomly choose a target """
# 		# return np.random.randint(len(candidates))
# 		y_pred = np.zeros(self.n_classes)
# 		rand_idx = np.random.randint(self.n_classes)
# 		y_pred[rand_idx] = 1
# 		return y_pred

# 	def calculate_reward(self, chosen_target_idx, target_candidate_idx):
# 		""" Determine reward given indices """
# 		print(chosen_target_idx, target_candidate_idx)
# 		if np.array_equal(chosen_target_idx, target_candidate_idx):
# 			return 1
# 		else:
# 			return 0

# 	def store_past(self):
# 		pass

# 	def train_agents_on_batch(self):
# 		pass

# 	def remember_speaker_training_details(self, state, action, y_true, prob, reward):
# 		""" Store inputs and outputs needed for training """
# 		# gradient = y_true 
# 		# self.gradients = 

# 	def fit(self, train_data):
# 		""" Random Sampling of messages and candidates for training"""
# 		self.training_stats = []
# 		message_probs_storage = []
# 		rewards_storage = []

# 		total_reward = 0
# 		batch_counter = 0
# 		batch_training_inputs = []
# 		for target_input, candidates, target_candidate_idx in train_data:
# 			message, message_probs = self.sample_speaker_policy_for_message(target_input)
# 			print("Message: %s, Probs: %s"%(message,message_probs))

# 			chosen_target_idx = self.listener_policy(message, candidates)
# 			reward = self.calculate_reward(chosen_target_idx,target_candidate_idx)
# 			total_reward += reward
# 			batch_counter += 1

# 			## Storage for training
# 			batch_training_inputs.append(target_input)
# 			rewards_storage.append(reward)
# 			message_probs_storage.append(message_probs)

# 			if self.save_training_stats:
# 				self.training_stats.append({
# 											"reward": reward,
# 											"input": target_input,
# 											"message": message,
# 											"chosen_target": candidates[np.where(chosen_target_idx==1)[0][0]]
# 											})

# 			if batch_counter==self.batch_size:
# 				self.train_speaker_policy_on_batch(batch_training_inputs, message_probs_storage, rewards_storage)
# 				batch_counter = 0
# 				batch_training_inputs,  message_probs_storage, rewards_storage = [], [], []

# 	def predict(self,test_data):
# 		""" Random Sampling of messages and candidates for testing"""
# 		self.testing_stats = []
# 		total_reward = 0
# 		for target_input, candidates, target_candidate_idx in test_data:
# 			message, message_probs = self.infer_from_speaker_policy(target_input)

# 			print("Message: %s, Probs: %s"%(message,message_probs))

# 			chosen_target_idx = self.listener_policy(message,candidates)
# 			reward = self.calculate_reward(chosen_target_idx,target_candidate_idx)
# 			total_reward += reward

# 			if self.save_training_stats:
# 				self.testing_stats.append({
# 											"reward": reward,
# 											"input": target_input,
# 											"message": message,
# 											"chosen_target": candidates[np.where(chosen_target_idx==1)[0][0]]
# 											})




# T = Input(shape=[50])
# t = Dense(50, activation="relu", kernel_initializer="he_normal")(t)
# # t = BatchNormalization()(t)
# t = LSTM(50,return_sequences=False,input_shape=(50,))(t)
# m = Dense(alphabet_size, activation="linear", kernel_initializer="zero")(t)
# model = Model(T,m)
# model.compile(loss="mse",optimizer=RMSprop(lr=0.0001))


"""
Speaker policy network
t => MLP => t_dense => LSTM => m
"""
