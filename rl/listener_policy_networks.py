import random
import numpy as np
import time 
import json

import keras
from keras.models import Sequential, load_model, Model 
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Dense, Convolution2D, LSTM, concatenate
from keras.optimizers import RMSprop, Adam
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.utils.np_utils import to_categorical

from rl.base_policy_networks import BaseSpeakerPolicyNetwork, BaseListenerPolicyNetwork
from rl.policy import EpsilonGreedyMessagePolicy


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
    self.policy = EpsilonGreedyMessagePolicy(eps=0.3) ## TODO: add as parameter later...
    self.__build_train_fn()

  def initialize_model(self):
    """ 2 Layer fully-connected neural network """
    ## Speakers Message 
    t_input = Input(shape=(self.alphabet_size,))
    z = Dense(self.alphabet_size, activation="relu", kernel_initializer='he_uniform')(t_input)
    z = BatchNormalization()(z)
    ## Candidate set
    u = Dense(self.speaker_dim, activation="relu", kernel_initializer='he_uniform')

    c1_input = Input(shape=(self.speaker_dim,))
    c2_input = Input(shape=(self.speaker_dim,))
    c3_input = Input(shape=(self.speaker_dim,))
    c4_input = Input(shape=(self.speaker_dim,))
    c5_input = Input(shape=(self.speaker_dim,))
    ## Encode candidates
    u_c1 = u(c1_input)
    u_c2 = u(c2_input)
    u_c3 = u(c3_input)
    u_c4 = u(c4_input)
    u_c5 = u(c5_input)
    ## Concatenate
    U = concatenate([z, u_c1, u_c2, u_c3, u_c4, u_c5],axis=-1)
    ## Final layer
    final_output = Dense(self.n_classes,activation="softmax")(U)
    ## Define model
    self.listener_model = Model(inputs=[t_input,c1_input,c2_input,c3_input,c4_input,c5_input], outputs=[final_output])
    self.listener_model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=self.listener_lr))
  
  def __build_train_fn(self):
    action_prob_placeholder = self.listener_model.output
    action_onehot_placeholder = K.placeholder(shape=(None, self.n_classes), name="action_onehot")
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
    updates = adam.get_updates(params=self.listener_model.trainable_weights,loss=loss)
    self.train_fn = K.function(
        inputs=[
        self.listener_model.input[0],
        self.listener_model.input[1],
        self.listener_model.input[2],
        self.listener_model.input[3],
        self.listener_model.input[4],
        self.listener_model.input[5],
        action_onehot_placeholder, 
        reward_placeholder],
        outputs=[loss,entropy], updates=updates)

  def sample_from_listener_policy(self, speaker_message, candidates):
    ## Organise input
    m = to_categorical(speaker_message[0], num_classes=self.alphabet_size)
    m = np.expand_dims(m, axis=0)
    X_ = [m] + [c.reshape([1,-1]) for c in candidates]
    ## Predict
    action_prob = np.squeeze(self.listener_model.predict(X_))
    print("listener action_probs: ", action_prob)
    action = np.random.choice(np.arange(self.n_classes), p=action_prob)
    print("listener action: ", action)
    return action, action_prob

  def infer_from_listener_policy(self, speaker_message, candidates):
    ## Organise input
    m = to_categorical(speaker_message[0], num_classes=self.alphabet_size)
    m = np.expand_dims(m, axis=0)
    X_ = [m] + [c.reshape([1,-1]) for c in candidates]
    ## Predict
    action_prob = np.squeeze(self.listener_model.predict(X_))
    print("listener action_probs: ", action_prob)
    action = np.argmax(action_prob)
    print("listener action: ", action)
    return [action], action_prob

  def train_listener_policy_on_batch(self):
    speaker_message = self.trial_speaker_message
    action = self.trial_action 
    reward = self.trial_reward
    candidates = self.trail_candidates

    action_onehot = to_categorical(action, num_classes=self.n_classes)
    action_onehot = action_onehot.reshape([-1,self.n_classes])
    m = to_categorical(speaker_message[0], num_classes=self.alphabet_size)
    model_input = [m.reshape([-1,self.alphabet_size])] + [c.reshape([1,-1]) for c in candidates]
    reward = np.array([reward])

    loss_, entropy_ = self.train_fn([
        model_input[0],
        model_input[1],
        model_input[2],
        model_input[3],
        model_input[4],
        model_input[5],
        action_onehot, 
        reward])
    print("Listener loss: ",loss_)
    print("Listener entropy: ",entropy_)

  def remember_listener_training_details(self, speaker_message, action, listener_probs, candidates, reward):
    """ Store inputs and outputs needed for training """
    ## Assumes batch_size ==1!!!!
    self.trial_speaker_message = speaker_message 
    self.trial_action = action 
    self.trail_candidates = candidates
    self.trial_reward = reward 





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
    return [np.random.randint(len(candidates))], np.array([1/float(len(candidates))]*len(candidates))

  def remember_listener_training_details(self,  speaker_message, action, listener_probs, candidates, reward):
    """ Store inputs and outputs needed for training """
    pass

  def train_listener_policy_on_batch(self):
    """ Update speaker policy given rewards """
    pass

  def infer_from_listener_policy(self, speaker_message, candidates):
    """ Obtain message from speaker policy """
    return [np.random.randint(len(candidates))]
