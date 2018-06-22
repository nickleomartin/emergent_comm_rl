import random
import numpy as np
import time 
import json

import keras
from keras.models import Sequential, load_model, Model 
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Dense, Convolution2D, LSTM
from keras.optimizers import RMSprop, Adam
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.utils.np_utils import to_categorical

from rl.base_policy_networks import BaseSpeakerPolicyNetwork
from rl.policy import EpsilonGreedyMessagePolicy


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
    self.policy = EpsilonGreedyMessagePolicy(eps=0.4) ## TODO: add as parameter later...
    self.__build_train_fn()

  def initialize_model(self):
    """ 2 Layer fully-connected neural network """
    self.speaker_model = Sequential()
    self.speaker_model.add(Dense(self.speaker_dim, activation="relu", input_shape=(self.speaker_dim,), kernel_initializer='he_uniform'))
    self.speaker_model.add(BatchNormalization())
    self.speaker_model.add(Dense(self.speaker_dim, activation="relu", kernel_initializer='he_uniform'))
    self.speaker_model.add(BatchNormalization())
    self.speaker_model.add(Dense(self.alphabet_size, activation="softmax"))
  
  def __build_train_fn(self):
    action_prob_placeholder = self.speaker_model.output
    action_onehot_placeholder = K.placeholder(shape=(None, self.alphabet_size), name="action_onehot")
    reward_placeholder = K.placeholder(shape=(None,), name="reward")
    action_prob = K.sum(action_prob_placeholder*action_onehot_placeholder, axis=1)
    log_action_prob = K.log(action_prob)
    loss = -log_action_prob*reward_placeholder

    ## Add entropy to the loss
    entropy = K.sum(action_prob_placeholder * K.log(action_prob_placeholder + 1e-10), axis=1)
    entropy = K.sum(entropy)
    
    loss = loss + 0.1*entropy
    loss = K.mean(loss)
    adam = Adam()
    updates = adam.get_updates(params=self.speaker_model.trainable_weights,loss=loss)
    self.train_fn = K.function(
        inputs=[self.speaker_model.input, action_onehot_placeholder, reward_placeholder],
        outputs=[loss, entropy], updates=updates)

  def sample_from_speaker_policy(self, target_input):
    if len(target_input.shape)==1:
      target_input = np.expand_dims(target_input, axis=0)
    action_prob = np.squeeze(self.speaker_model.predict(target_input))
    ## Assume single token message!!!!!!!
    action = np.random.choice(np.arange(self.alphabet_size), p=action_prob)
    # print("Speaker action_prob: ", action_prob)
    print("Speaker action: ", action)
    return [action], action_prob

  def infer_from_speaker_policy(self, target_input):
    if len(target_input.shape)==1:
      target_input = np.expand_dims(target_input, axis=0)
    action_prob = np.squeeze(self.speaker_model.predict(target_input))
    ## Assume single token message!!!!!!!
    print("speaker action_probs: ", action_prob)
    action = np.argmax(action_prob)
    print("speaker action: ", action)
    return [action], action_prob

  def train_speaker_policy_on_batch(self):
    target_input = self.trial_target_input 
    action = self.trial_action 
    reward = self.trial_reward

    action_onehot = to_categorical(action, num_classes=self.alphabet_size)
    target_input = self.reshape_target(target_input)
    reward = np.array([reward])
    loss_, entropy_ = self.train_fn([target_input, action_onehot, reward])
    print("Speaker loss: ", loss_)
    print("Speaker entropy: ", entropy_)

  def remember_speaker_training_details(self, target_input, action, speaker_probs, reward):
    """ Store inputs and outputs needed for training """
    ## Assumes batch_size ==1!!!!
    self.trial_target_input = target_input 
    self.trial_action = action 
    self.trial_reward = reward 

  def reshape_target(self, target_input):
    """ Reshape target_input to (1, input_dim) """
    return target_input.reshape([1,self.speaker_dim])


