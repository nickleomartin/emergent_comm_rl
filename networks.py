import random
import numpy as np
import time 
import json

import keras
from keras.models import Sequential, load_model, Model 
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Dense, Convolution2D, LSTM
from keras.optimizers import rmsprop
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
	def __init__(self, config_dict):
		self.config_dict = config_dict
		self.initialize_parameters()

	def initialize_parameters(self):
		""" Assign config parameters to local vars """
		self.max_message_length = self.config_dict['max_message_length']
		self.alphabet = self.config_dict['alphabet']

	def speaker_policy(self,target_input,max_message_length=2):
		""" Randomly generate a message """
		return [np.random.choice(self.alphabet) for i in range(self.max_message_length)]

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
		""" Placeholder function """
		total_reward = 0
		for target_input, candidates, target_candidate_idx in train_data:
			message = self.speaker_policy(target_input)
			chosen_target_idx = self.listener_policy(message,candidates)
			total_reward += self.calculate_reward(chosen_target_idx,target_candidate_idx)
		print("Total Reward: %s"%total_reward)
		print("Percentage Reward: %s"%(total_reward*100/float(len(train_data))))





"""
Speaker policy network
t => MLP => t_dense => LSTM => m
"""
# model = Sequential()
# # model.add(InputLayer(batch_input_shape=(50,)))
# model.add(Dense(output_dim=50))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(LSTM(50))
# ## Generate samples of letters
# rms = rmsprop(lr=0.0001)
# model.compile(loss="categorical_crossentropy",optimizer=rms)


