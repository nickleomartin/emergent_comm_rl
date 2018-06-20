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


class BasePolicyNetwork(object):
	""" Parent model with save and load methods """
	def __init__(self):
		self._model = None
		self._config_dict = None

	def load_config(self):
		""" Read in config parameters """
		assert self._config_dict is not None, "Please pass in the config dictionary"
		self._max_message_length = self._config_dict['max_message_length']
		self._alphabet = self._config_dict['alphabet']
		self._alphabet_size = self._config_dict['alphabet_size']
		self._speaker_lr = self._config_dict['speaker_lr']
		self._speaker_dim = self._config_dict['speaker_dim']
		self._listener_lr = self._config_dict['listener_lr']
		self._listener_dim = self._config_dict['listener_dim']
		self._training_epoch = self._config_dict['training_epoch']
		self._batch_size = self._config_dict['batch_size']
		self._n_classes = self._config_dict['n_distractors'] + 1

	def save(self, weights_file, params_file):
		""" Save weights and parameters to file """
		self.save_weights(weights_file)
		self.save_params(params_file)

	def save_weights(self, file_path):
		""" Save model weights to file """
		self._model.save_weights(file_path)

	def save_params(self, file_path):
		""" Save model parameters to file """
		with open(file_path, 'w') as f:
			params = {name.lstrip('_'): val for name, val in vars(self).items()
					if name not in {'_loss','model','_embeddings'}}
			json.dump(params, f, sort_keys=True, indent=4)

	@classmethod
	def load_params(cls, weights_file, params_file):
		""" Instantiate and load previous model """
		params = cls.load_params(params_file)
		self = cls(**params)
		self.construct()
		self._model.load_weights(weights_file)
		return self 

	@classmethod
	def load(cls, file_path):
		""" Load parameters from file """
		with open(file_path) as f:
			params = json.load(f)
		return params





class SpeakerPolicyNetwork(BasePolicyNetwork):
	""" Parent speaker policy model """
	def __inti__(self, model, config_dict):
		super(BasePolicyNetwork).__init__()
		self._model = model
		self._config_dict = config_dict
		
	def sample_speaker_policy(self, target_input):
		""" Stochastically sample message of length self.max_message_length from speaker policy """ 
		x = target_input.reshape([1, self._speaker_dim, 1])
		probs = self._model.predict(x, batch_size=1)
		normalized_probs = probs/np.sum(probs)

		message = ""
		message_probs = []
		for i in range(self._max_message_length):
			sampled_symbol = np.random.choice(self._alphabet_size, 1, p=normalized_probs[0])[0]
			message += str(sampled_symbol) + "#"
			message_probs.append(normalized_probs[0][sampled_symbol])
		
		## TODO: Also return sum[log prob () mi | target input and weights)]??
		return message, message_probs

	def train_on_batch(self, reward):
		pass

	def infer_speaker_policy(self):
		pass




class MLPSpeakerPolicyNetwork(SpeakerPolicyNetwork):
	""" """
	def __init__(self, model, config_dict):
		super(SpeakerPolicyNetwork).__init__()
		self._model = model
		self._config_dict = config_dict
		self.load_config()




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
