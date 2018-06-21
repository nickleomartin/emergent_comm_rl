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
		self.alphabet_size = self.config_dict['alphabet_size']

	def speaker_policy(self,target_input, max_message_length=2):
		""" Randomly generate a message """
		return [np.random.choice(range(self.alphabet_size)) for i in range(self.max_message_length)]

	def listener_policy(self, speaker_message, candidates):
		""" Randomly choose a target """
		return np.random.randint(len(candidates))

	def calculate_reward(self, chosen_target_idx, target_candidate_idx):
		""" Determine reward given indices """
		if target_candidate_idx[chosen_target_idx]==1.:
			return 1
		else:
			return 0

	def fit(self, train_data):
		""" Random Sampling of messages and candidates for training"""
		self.training_stats = []
		total_reward = 0
		for target_input, candidates, target_candidate_idx in train_data:
			speaker_message = self.speaker_policy(target_input)
			chosen_target_idx = self.listener_policy(speaker_message,candidates)
			reward = self.calculate_reward(chosen_target_idx,target_candidate_idx)
			total_reward += reward

			if self.save_training_stats:
				self.training_stats.append({
											"reward": reward,
											"input": target_input,
											"message": speaker_message,
											"chosen_target_idx": chosen_target_idx
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
											"chosen_target_idx": chosen_target_idx,
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
		if target_candidate_idx[chosen_target_idx]==1.:
			return 1
		else:
			return 0


	def sample_from_networks_on_batch(self, target_input, candidates, target_candidate_idx):

		## Sample from speaker
		speaker_message, speaker_probs = self.speaker_model.sample_speaker_policy_for_message(target_input)
		# print("Message: %s, Probs: %s"%(speaker_message, speaker_probs))

		## Sample from listener
		listener_action, listener_probs = self.listener_model.sample_from_listener_policy(speaker_message, candidates)

		## Calculate reward
		reward = self.calculate_reward(listener_action, target_candidate_idx)

		## Store batch inputs and outputs
		self.speaker_model.remember_speaker_training_details(target_input, speaker_message, speaker_probs, reward)
		self.listener_model.remember_listener_training_details(target_input, listener_action, listener_probs, reward)

		## Increment batch statistics
		self.total_training_reward += reward
		self.batch_counter += 1

		## Record training statistics
		if self.save_training_stats:
			self.training_stats.append({
										"reward": reward,
										"input": target_input,
										"message": speaker_message,
										"chosen_target_idx": listener_action
										})


	def train_networks_on_batch(self):
		""" Train Speaker and Listener network on batch """
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

		# len_training_set = len(train_data)

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
			# print("Message: %s, Probs: %s"%(message,message_probs))

			chosen_target_idx = self.listener_model.infer_from_listener_policy(message,candidates)
			reward = self.calculate_reward(chosen_target_idx,target_candidate_idx)
			total_reward += reward

			if self.save_training_stats:
				self.testing_stats.append({
											"reward": reward,
											"input": target_input,
											"message": message,
											"chosen_target_idx": chosen_target_idx,
											})
 
