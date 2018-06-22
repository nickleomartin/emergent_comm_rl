



class BasePolicyNetwork(object):
	""" Abstract policy network """
	def __init__(self, config_dict):
		self.config_dict = config_dict
		self.batch_target_inputs = []
		self.batch_rewards = []
		self.batch_actions = []
		self.batch_probs = []

	def initialize_parameters(self):
		assert self.config_dict, "self.config_dict does not exist"
		self.max_message_length = self.config_dict['max_message_length']
		self.alphabet = self.config_dict['alphabet']
		self.alphabet_size = self.config_dict['alphabet_size']
		self.speaker_lr = self.config_dict['speaker_lr']
		self.speaker_dim = self.config_dict['speaker_dim']
		self.speaker_input_dim = self.config_dict['speaker_input_dim']
		self.listener_lr = self.config_dict['listener_lr']
		self.listener_dim = self.config_dict['listener_dim']
		self.training_epoch = self.config_dict['training_epoch']
		self.batch_size = self.config_dict['batch_size']
		self.n_distractors = self.config_dict['n_distractors']
		self.n_classes = self.config_dict['n_distractors'] + 1

	def initialize_model(self):
		""" Build and compile Keras model """
		pass

	
class BaseSpeakerPolicyNetwork(BasePolicyNetwork):
	""" Abstraction of Speaker Network """
	def __init__(self, config_dict):
		super(BaseSpeakerPolicyNetwork, self).__init__(config_dict)
		self.batch_target_inputs = []
		self.initialize_parameters()
		self.initialize_model()

	def sample_speaker_policy_for_message(self, target_input):
		""" Sample message of length self.max_message_length from speaker policy """ 
		pass

	def remember_speaker_training_details(self, target_input, action, speaker_probs, reward):
		""" Store inputs and outputs needed for training """
		pass

	def train_speaker_policy_on_batch(self):
		""" Update speaker policy given rewards """
		pass

	def infer_from_speaker_policy(self, target_input):
		""" Obtain message from speaker policy """
		pass



class BaseListenerPolicyNetwork(BasePolicyNetwork):
	""" Abstraction of Listner Network """
	def __init__(self, config_dict):
		super(BaseListenerPolicyNetwork, self).__init__(config_dict)
		self.batch_messages = []
		self.batch_candidates = []
		self.initialize_parameters()
		self.initialize_model()

	def sample_from_listener_policy(self):
		""" Sample target index from speaker policy """ 
		pass

	def remember_listener_training_details(self, target_input, action, speaker_probs, reward):
		""" Store inputs and outputs needed for training """
		pass

	def train_listener_policy_on_batch(self):
		""" Update listener policy given rewards """
		pass

	def infer_from_listener_policy(self, target_input):
		""" Obtain target index from listener policy """
		pass