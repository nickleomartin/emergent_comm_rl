import numpy as np


def generate_dummy_data(config_dict):
	""" Generate random binary vector dataset """
	v_dim = config_dict['v_dim']
	n_dim = config_dict['n_dim']
	n_distractors = config_dict['n_distractors']

	training_data_tuples = []
	for _ in range(n_dim):
		candidate_set = [np.random.randint(0,2,size=v_dim) for idx in range(n_distractors+1)]
		target_idx = np.random.randint(n_distractors+1)
		target = candidate_set[target_idx]
		training_data_tuples.append((target,candidate_set,target_idx))
	return training_data_tuples










