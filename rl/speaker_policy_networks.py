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

from rl.base_policy_networks import BaseSpeakerNetwork
from rl.policy import EpsilonGreedyMessagePolicy


class DenseSpeakerNetwork(BaseSpeakerNetwork):
  """ 
  Fully connected speaker policy model 
  
  Example:
  --------
  from config import config_dict
  from rl.speaker_policy_networks import DenseSpeakerNetwork
  
  speaker = DenseSpeakerNetwork(config_dict)
  """
  def __init__(self, config_dict):
    super(DenseSpeakerNetwork, self).__init__(config_dict)
    self.policy = EpsilonGreedyMessagePolicy(eps=0.4) ## TODO: add as parameter later...
    self.__build_train_fn()

  def initialize_model(self):
    """ 2 Layer fully-connected neural network """
    self.speaker_model = Sequential()
    self.speaker_model.add(Dense(self.speaker_dim, activation="relu", input_shape=(self.speaker_input_dim,), kernel_initializer='he_uniform'))
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
    return target_input.reshape([1,self.speaker_input_dim])



class RandomSpeakerNetwork(BaseSpeakerNetwork):
  """ 
  Random speaker policy model 
  
  Example:
  --------
  from config import random_config_dict as config_dict
  from rl.speaker_policy_networks import RandomSpeakerNetwork
  
  speaker = RandomSpeakerNetwork(config_dict)
  """
  def __init__(self, config_dict):
    super(RandomSpeakerNetwork, self).__init__(config_dict)

  def sample_from_speaker_policy(self, target_input):
    """ Sample message of length self.max_message_length from speaker policy """ 
    speaker_message = [np.random.choice(range(self.alphabet_size)) for i in range(self.max_message_length)]
    probs = np.array([1/float(self.alphabet_size)]*self.alphabet_size)
    return [speaker_message], probs

  def remember_speaker_training_details(self, target_input, action, speaker_probs, reward):
    """ Store inputs and outputs needed for training """
    pass

  def train_speaker_policy_on_batch(self):
    """ Update speaker policy given rewards """
    pass

  def infer_from_speaker_policy(self, target_input):
    """ Obtain message from speaker policy """
    speaker_message = [np.random.choice(range(self.alphabet_size)) for i in range(self.max_message_length)]
    probs = np.array([1/float(self.alphabet_size)]*self.alphabet_size)
    return [speaker_message], probs



########################
## Under construction ##
########################
class PaperSpeakerNetwork(BaseSpeakerNetwork):
  """ 
  Speaker policy model 
  
  Example:
  --------
  from config import visa_config_dict as config_dict
  from rl.speaker_policy_networks import PaperSpeakerNetwork
  
  speaker = PaperSpeakerNetwork(config_dict)
  """
  def __init__(self, config_dict):
    super(PaperSpeakerNetwork, self).__init__(config_dict)
    self.policy = EpsilonGreedyMessagePolicy(eps=0.4) ## TODO: add as parameter later...
    self.alphabet_tokens = np.array(self.alphabet).reshape([1,-1])
    self.char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    self.int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    self.__build_train_fn()

  def initialize_model(self):
    """ 2 Layer fully-connected neural network """
    ## Encoder
    ## Hack to get over LSTM's need for a two inputs
    encoder_inputs = Input(shape=(self.speaker_input_dim,))
    encoder_1 = Dense(self.speaker_dim, activation="relu")
    encoder_rep_1 = encoder_1(encoder_inputs)
    encoder_2 = Dense(self.speaker_dim, activation="relu")
    encoder_rep_2 = encoder_2(encoder_inputs)
    encoder_states = [encoder_rep_1, encoder_rep_2]

    ## Decoder
    decoder_inputs = Input(shape=(None, self.alphabet_size))
    decoder_lstm = LSTM(self.speaker_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(self.alphabet_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    self.speaker_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
  
  def __build_train_fn(self):
    """ Custom function to handle policy gradients """
    action_prob_placeholder = self.speaker_model.output
    action_onehot_placeholder = K.placeholder(shape=(None, self.alphabet_size), name="action_onehot")
    reward_placeholder = K.placeholder(shape=(None,), name="reward")
    action_prob = K.sum(action_prob_placeholder*action_onehot_placeholder, axis=1)
    log_action_prob = K.log(action_prob)
    loss = -log_action_prob*reward_placeholder

    ## Add entropy to the loss
    entropy = K.sum(action_prob_placeholder * K.log(action_prob_placeholder + 1e-10), axis=1)
    entropy = K.sum(entropy)
    
    ## TODO: add entropy regularization parameter ...
    loss = loss + 0.1*entropy
    loss = K.mean(loss)
    adam = Adam()
    updates = adam.get_updates(params=self.speaker_model.trainable_weights,loss=loss)
    self.train_fn = K.function(
        inputs=[
        self.speaker_model.input[0],
        self.speaker_model.input[1], 
        action_onehot_placeholder, 
        reward_placeholder],
        outputs=[loss, entropy], updates=updates)

  def sample_from_speaker_policy(self, target_input):
    if len(target_input.shape)==1:
      target_input = np.expand_dims(target_input, axis=0)
    action_prob = np.squeeze(self.speaker_model.predict([target_input,self.alphabet_tokens]))
    ## Assume single token message!!!!!!!
    action = np.random.choice(np.arange(self.alphabet_size), p=action_prob)
    # print("Speaker action_prob: ", action_prob)
    print("Speaker action: ", action)
    return [action], action_prob

  def infer_from_speaker_policy(self, target_input):
    if len(target_input.shape)==1:
      target_input = np.expand_dims(target_input, axis=0)
    action_prob = np.squeeze(self.speaker_model.predict([target_input,self.alphabet_tokens]))
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

    loss_, entropy_ = self.train_fn([
        target_input,
        self.alphabet_tokens, 
        action_onehot, 
        reward])
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
    return target_input.reshape([1,self.speaker_input_dim])



