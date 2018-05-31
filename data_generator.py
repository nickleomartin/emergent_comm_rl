import numpy as np


def generate_dummy_data(v_dim=50, n_dim=100, n_distractors=4):
	""" Generate random binary vector dataset """
	training_data_tuples = []
	for _ in range(n_dim):
		candidate_set = [np.random.randint(0,2,size=v_dim) for idx in range(n_distractors+1)]
		target_idx = np.random.randint(n_distractors+1)
		target = candidate_set[target_idx]
		training_data_tuples.append((target,candidate_set,target_idx))
	return training_data_tuples










