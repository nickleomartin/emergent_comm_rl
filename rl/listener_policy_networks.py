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

from rl.base_policy_networks import BaseSpeakerPolicyNetwork, BaseListenerPolicyNetwork
from rl.policy import EpsilonGreedyMessagePolicy


class RandomListenerPolicyNetwork(BaseListenerPolicyNetwork):
  """ 
  Random listener policy model 
  
  Example:
  --------
  from config import random_config_dict as config_dict
  from rl.listener_policy_networks import RandomListenerPolicyNetwork
  
  listener = RandomListenerPolicyNetwork(config_dict)
  """
  def __init__(self, config_dict):
    super(RandomListenerPolicyNetwork, self).__init__(config_dict)

  def sample_from_listener_policy(self, speaker_message, candidates):
    """ Sample message of length self.max_message_length from speaker policy """ 
    return np.random.randint(len(candidates)), np.array([1/float(len(candidates))]*len(candidates))

  def remember_listener_training_details(self,  speaker_message, action, listener_probs, reward):
    """ Store inputs and outputs needed for training """
    self.batch_messages.append(speaker_message) 
    self.batch_actions.append(action)
    self.batch_rewards.append(reward)
    self.batch_probs.append(listener_probs)
    gradients = np.array(action).astype("float32") - listener_probs
    self.batch_gradients.append(gradients)

  def train_listener_policy_on_batch(self):
    """ Update speaker policy given rewards """
    ## Reset batch memory
    self.batch_messages, self.batch_actions, \
    self.batch_rewards, self.batch_gradients, \
    self.batch_probs = [], [], [], [], []

  def infer_from_listener_policy(self, speaker_message, candidates):
    """ Obtain message from speaker policy """
    return np.random.randint(len(candidates))



class DenseListenerPolicyNetwork(BaseListenerPolicyNetwork):
  """ 
  Fully connected listener policy model 
  
  Example:
  --------
  from config import config_dict
  from networks import DenseListenerPolicyNetwork
  
  listener = DenseListenerPolicyNetwork(config_dict)
  """
  def __init__(self, config_dict):
    super(DenseListenerPolicyNetwork, self).__init__(config_dict)
    self.policy = EpsilonGreedyMessagePolicy(eps=0.1)

  def initialize_model(self):
    """ 2 Layer fully-connected neural network """
    self.listener_model = Sequential()
    self.listener_model.add(Dense(self.alphabet_size, activation="relu", input_shape=(self.alphabet_size,)))
    self.listener_model.add(Dense(self.listener_dim, activation="relu"))
    self.listener_model.add(Dense(self.n_classes,activation="softmax"))
    self.listener_model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=self.listener_lr))
  
  def one_hot_encode_message(self, speaker_message):
    """ list of ints to one hot message vector """
    m = np.zeros(self.alphabet_size)
    for i in range(len(speaker_message)):
      m[speaker_message[i]] = 1
    return m.reshape([1,self.alphabet_size])

  def sample_from_listener_policy(self, speaker_message, candidates):
    """ """
    ## Message representation as one-hot for now....
    m = self.one_hot_encode_message(speaker_message)
    probs = self.listener_model.predict_on_batch(m)
    normalized_probs = probs / np.sum(probs)

    ## TODO: Implement Policy class: EpsilonGreedy if training else np.argmax
    action = self.policy.select_action(normalized_probs, self.n_classes,1)
    
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
    X = np.squeeze(np.vstack([self.batch_messages]),axis=1)

    ## Create Y = probs + lr * gradients
    Y = np.squeeze(np.array(self.batch_probs)) + self.listener_lr * np.squeeze(np.vstack([gradients]))

    ## Train model
    self.listener_model.train_on_batch(X, Y)

    ## Reset batch memory
    self.batch_messages, self.batch_actions, \
    self.batch_rewards, self.batch_gradients, \
    self.batch_probs = [], [], [], [], []

  def remember_listener_training_details(self, speaker_message, action, listener_probs, reward):
    """ Store inputs and outputs needed for training """
    m = self.one_hot_encode_message(speaker_message)

    self.batch_messages.append(m) 
    self.batch_actions.append(action)
    self.batch_rewards.append(reward)
    self.batch_probs.append(listener_probs)

    gradients = np.array(action).astype("float32") - listener_probs
    self.batch_gradients.append(gradients)

  def infer_from_listener_policy(self, speaker_message, candidates):
    """ Randomly choose a target for now ! """
    ## Get symbol probabilities given target input
    m = self.one_hot_encode_message(speaker_message)

    probs = self.listener_model.predict_on_batch(m)
    normalized_probs = probs/np.sum(probs)

    ## Greedily get symbols with largest probabilities
    target_idx = np.argmax(normalized_probs)
    return target_idx

