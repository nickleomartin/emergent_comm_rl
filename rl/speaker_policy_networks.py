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

from rl.base_policy_networks import BaseSpeakerPolicyNetwork


class RandomSpeakerPolicyNetwork(BaseSpeakerPolicyNetwork):
  """ 
  Random speaker policy model 
  
  Example:
  --------
  from config import random_config_dict as config_dict
  from rl.speaker_policy_networks import RandomSpeakerPolicyNetwork
  
  speaker = RandomSpeakerPolicyNetwork(config_dict)
  """
  def __init__(self, config_dict):
    super(RandomSpeakerPolicyNetwork, self).__init__(config_dict)

  def sample_speaker_policy_for_message(self, target_input):
    """ Sample message of length self.max_message_length from speaker policy """ 
    speaker_message = [np.random.choice(range(self.alphabet_size)) for i in range(self.max_message_length)]
    probs = np.array([1/float(self.alphabet_size)]*self.alphabet_size)
    return speaker_message, probs

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
    ## Reset batch memory
    self.batch_target_inputs, self.batch_actions, \
    self.batch_rewards, self.batch_gradients, \
    self.batch_probs = [], [], [], [], []

  def infer_from_speaker_policy(self, target_input):
    """ Obtain message from speaker policy """
    speaker_message = [np.random.choice(range(self.alphabet_size)) for i in range(self.max_message_length)]
    probs = np.array([1/float(self.alphabet_size)]*self.alphabet_size)
    return speaker_message, probs


class DenseSpeakerPolicyNetwork(BaseSpeakerPolicyNetwork):
  """ 
  Fully connected speaker policy model 
  
  Example:
  --------
  from config import config_dict
  from rl.speaker_policy_networks import DenseSpeakerPolicyNetwork
  
  speaker = DenseSpeakerPolicyNetwork(config_dict)
  """
  def __init__(self, config_dict):
    super(DenseSpeakerPolicyNetwork, self).__init__(config_dict)

  def initialize_model(self):
    """ 2 Layer fully-connected neural network """
    self.speaker_model = Sequential()
    self.speaker_model.add(Dense(self.speaker_dim, activation="relu", input_shape=(self.speaker_dim,)))
    self.speaker_model.add(Dense(self.alphabet_size,activation="softmax"))
    self.speaker_model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=self.speaker_lr))
  
  def reshape_target(self, target_input):
    """ Reshape target_input to (1, input_dim) """
    return target_input.reshape([1,self.speaker_dim])

  def sample_speaker_policy_for_message(self, target_input):
    """ Stochastically sample message of length self.max_message_length from speaker policy """ 
    t_input = self.reshape_target(target_input)
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
    probs = self.speaker_model.predict_on_batch(self.reshape_target(target_input))
    normalized_probs = probs/np.sum(probs)

    ## Greedily get symbols with largest probabilities
    message_indices = list(normalized_probs[0].argsort()[-self.max_message_length:][::-1])
    message_probs = normalized_probs[0][message_indices]
    message = message_indices

    ## TODO: Also return sum[log prob () mi | target input and weights)]??
    return message, message_probs

