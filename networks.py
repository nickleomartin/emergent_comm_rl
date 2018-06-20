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


""" See https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py for Boltzmann Policy """

## TODO: Inherit from BasePolicyNetwork later....

class DenseSpeakerPolicyNetwork(object):
	""" Fully connected speaker policy model 
	
	Example:
	--------
	from config import config_dict
	from networks import DenseSpeakerPolicyNetwork
	
	speaker = DenseSpeakerPolicyNetwork(config_dict)

	"""
	def __init__(self, config_dict):
		self.config_dict = config_dict
		self.batch_target_inputs = []
		self.batch_rewards = []
		self.batch_actions = []
		self.batch_probs = []
		self.batch_gradients = []
		self.initialize_parameters()
		self.initialize_speaker_model()

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

	def initialize_speaker_model(self):
		""" 2 Layer fully-connected neural network """
		self.speaker_model = Sequential()
		self.speaker_model.add(Dense(self.speaker_dim, activation="relu", input_shape=(self.speaker_dim,)))
		self.speaker_model.add(Dense(self.alphabet_size,activation="softmax"))
		self.speaker_model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=self.speaker_lr))
	
	def sample_speaker_policy_for_message(self, target_input):
		""" Stochastically sample message of length self.max_message_length from speaker policy """ 
		t_input = target_input.reshape([1,self.speaker_dim])
		probs = self.speaker_model.predict_on_batch(t_input)
		normalized_probs = probs / np.sum(probs)

		speaker_message = []
		for i in range(self.max_message_length):
			## TODO: Implement Policy class: EpsilonGreedy if training else np.argmax
			sampled_symbol = np.random.choice(self.alphabet_size, 1, p=normalized_probs[0])[0]
			speaker_message.append(sampled_symbol)
		
		## Return action and probs
		return speaker_message, normalized_probs

	def remember_speaker_training_details(self, target_input, action, speaker_probs, reward):
		""" Store inputs and outputs needed for training """
		self.batch_target_inputs.append(target_input) 
		self.batch_actions.append(action)
		self.batch_rewards.append(reward)
		self.batch_probs.append(speaker_probs)

		y = np.zeros(self.alphabet_size)

		for i in range(self.max_message_length):
			y[action[i]] = 1

		gradients = np.array(y).astype("float32") - speaker_probs
		self.batch_gradients.append(gradients)

	def train_speaker_policy_on_batch(self):
		""" Update speaker policy given rewards """
		## Calculate gradients = action - probs
		gradients = np.vstack(self.batch_gradients)

		## Batch standardise rewards. Note: no discounting of rewards
		rewards = np.vstack(self.batch_rewards)

		if np.count_nonzero(rewards)>0:
			rewards = rewards / np.std(rewards - np.mean(rewards)) ## TODO: Handle zero rewards

		## Calculate gradients * rewards
		gradients *= rewards

		## Create X
		X = np.vstack([self.batch_target_inputs])

		## Create Y = probs + lr * gradients
		Y = np.squeeze(np.array(self.batch_probs)) + self.speaker_lr * np.squeeze(np.vstack([gradients]))

		## Train model
		self.speaker_model.train_on_batch(X, Y)

		## Reset batch memory
		self.batch_target_inputs, self.batch_actions, \
		self.batch_rewards, self.batch_gradients, \
		self.batch_probs = [], [], [], [], []


	def infer_from_speaker_policy(self, target_input):
		""" Greedily obtain message from speaker policy """
		## Get symbol probabilities given target input
		probs = self.speaker_model.predict_on_batch(target_input.reshape([1,self.speaker_dim]))
		normalized_probs = probs/np.sum(probs)

		## Greedily get symbols with largest probabilities
		message_indices = list(normalized_probs[0].argsort()[-self.max_message_length:][::-1])
		message_probs = normalized_probs[0][message_indices]
		message = message_indices
		
		## TODO: Also return sum[log prob () mi | target input and weights)]??
		return message, message_probs





class DenseListenerPolicyNetwork(object):
	""" Fully connected speaker policy model 
	
	Example:
	--------
	from networks import DenseListenerPolicyNetwork
	
	config_dict = .....
	listener = DenseListenerPolicyNetwork(config_dict)

	"""
	def __init__(self, config_dict):
		self.config_dict = config_dict
		self.batch_messages = []
		self.batch_rewards = []
		self.batch_actions = []
		self.batch_probs = []
		self.batch_gradients = []
		self.initialize_parameters()
		self.initialize_listener_model()

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

	def initialize_listener_model(self):
		""" 2 Layer fully-connected neural network """
		self.listener_model = Sequential()
		self.listener_model.add(Dense(self.alphabet_size, activation="relu", input_shape=(self.alphabet_size,)))
		self.listener_model.add(Dense(self.listener_dim, activation="relu"))
		self.listener_model.add(Dense(self.n_classes,activation="softmax"))
		self.listener_model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=self.listener_lr))
	
	def sample_from_listener_policy(self, speaker_message, candidates):
		""" """
		## Message representation as one-hot for now....
		m = np.zeros(self.alphabet_size)

		# print(speaker_message)
		for i in range(len(speaker_message)):
			m[speaker_message[i]] = 1

		t_input = m.reshape([1,self.alphabet_size])
		probs = self.listener_model.predict_on_batch(t_input)
		normalized_probs = probs / np.sum(probs)

		## TODO: Implement Policy class: EpsilonGreedy if training else np.argmax
		action = np.random.choice(self.n_classes, 1, p=normalized_probs[0])[0]
		
		## Return action and probs
		return action, normalized_probs

	def train_listener_policy_on_batch(self):
		""" Update listener policy given rewards """
		gradients = np.vstack(self.batch_gradients)

		## Batch standardise rewards. Note: no discounting of rewards
		rewards = np.vstack(self.batch_rewards)

		if np.count_nonzero(rewards)>0:
			rewards = rewards / np.std(rewards - np.mean(rewards))

		## Calculate gradients * rewards
		gradients *= rewards

		## Create X
		X = np.vstack([self.batch_messages])

		## Create Y = probs + lr * gradients
		Y = np.squeeze(np.array(self.batch_probs)) + self.speaker_lr * np.squeeze(np.vstack([gradients]))

		## Train model
		self.listener_model.train_on_batch(X, Y)

		## Reset batch memory
		self.batch_messages, self.batch_actions, \
		self.batch_rewards, self.batch_gradients, \
		self.batch_probs = [], [], [], [], []

	def remember_listener_training_details(self, speaker_message, action, listener_probs, reward):
		""" Store inputs and outputs needed for training """
		m = np.zeros(self.alphabet_size)
		for i in range(len(speaker_message)):
			m[speaker_message[i]] = 1

		self.batch_messages.append(m) 
		self.batch_actions.append(action)
		self.batch_rewards.append(reward)
		self.batch_probs.append(listener_probs)

		gradients = np.array(action).astype("float32") - listener_probs
		self.batch_gradients.append(gradients)

	def infer_from_listener_policy(self, speaker_message, candidates):
		""" Randomly choose a target for now ! """
		## Get symbol probabilities given target input
		m = np.zeros(self.alphabet_size)
		for i in range(len(speaker_message)):
			m[speaker_message[i]] = 1

		probs = self.listener_model.predict_on_batch(m.reshape([1,self.alphabet_size]))
		normalized_probs = probs/np.sum(probs)

		## Greedily get symbols with largest probabilities
		target_idx = np.argmax(normalized_probs)
		return target_idx



## TODO: Implement Parent class at first re-factor....

# class BasePolicyNetwork(object):
# 	""" Parent model with save and load methods """
# 	def __init__(self):
# 		self._model = None
# 		self._config_dict = None

# 	def load_config(self):
# 		""" Read in config parameters """
# 		assert self._config_dict is not None, "Please pass in the config dictionary"
# 		self._max_message_length = self._config_dict['max_message_length']
# 		self._alphabet = self._config_dict['alphabet']
# 		self._alphabet_size = self._config_dict['alphabet_size']
# 		self._speaker_lr = self._config_dict['speaker_lr']
# 		self._speaker_dim = self._config_dict['speaker_dim']
# 		self._listener_lr = self._config_dict['listener_lr']
# 		self._listener_dim = self._config_dict['listener_dim']
# 		self._training_epoch = self._config_dict['training_epoch']
# 		self._batch_size = self._config_dict['batch_size']
# 		self._n_classes = self._config_dict['n_distractors'] + 1

# 	def save(self, weights_file, params_file):
# 		""" Save weights and parameters to file """
# 		self.save_weights(weights_file)
# 		self.save_params(params_file)

# 	def save_weights(self, file_path):
# 		""" Save model weights to file """
# 		self._model.save_weights(file_path)

# 	def save_params(self, file_path):
# 		""" Save model parameters to file """
# 		with open(file_path, 'w') as f:
# 			params = {name.lstrip('_'): val for name, val in vars(self).items()
# 					if name not in {'_loss','model','_embeddings'}}
# 			json.dump(params, f, sort_keys=True, indent=4)

# 	@classmethod
# 	def load_params(cls, weights_file, params_file):
# 		""" Instantiate and load previous model """
# 		params = cls.load_params(params_file)
# 		self = cls(**params)
# 		self.construct()
# 		self._model.load_weights(weights_file)
# 		return self 

# 	@classmethod
# 	def load(cls, file_path):
# 		""" Load parameters from file """
# 		with open(file_path) as f:
# 			params = json.load(f)
# 		return params


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
