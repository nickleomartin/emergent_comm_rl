import numpy as np 



class BasePolicy(object):
	""" Abstract base class for policies """
	def select_action(self, **kwargs):
		""" Sample action according to probaility """
		raise NotImplementedError()

	def get_parameters(self):
		""" Return policy parameters """
		return {}
	

class EpsilonGreedyMessagePolicy(BasePolicy):
	""" 
	Takes a random action with probability eps, 
	otherwise best action chosen 

	Example:
	--------
	import numpy as np
	from policy import EpsilonGreedyMessagePolicy
	
	probs = np.array([[0.1,0.2,0.3,0.1,0.1,0.2]])
	n_actions = 2
	m_length = 2

	egmp = EpsilonGreedyMessagePolicy()
	egmp.select_action(probs,n_actions, m_length)

	"""
	def __init__(self, eps=0.1):
		super(EpsilonGreedyMessagePolicy, self).__init__()
		self.eps = eps

	def select_action(self, probs, n_actions, m_length):
		""" Sample action according to probaility """
		if np.random.uniform() < self.eps:
			return np.random.choice(n_actions, m_length, p=probs[0])
		else:
			return list(probs[0].argsort()[-m_length:][::-1])

	def get_parameters(self):
		""" Return policy parameters """
		params = super(EpsilonGreedyPolicy, self).__init__()
		params['eps'] = self.eps
		return params






""" Reference: See https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py for Boltzmann Policy """















