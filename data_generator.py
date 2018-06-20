import numpy as np
from keras.utils.np_utils import to_categorical

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


def generate_dummy_categorical_dataset(config_dict, dataset="training"):
	""" Generate random binary vector dataset with categorical labels """
	v_dim = config_dict['v_dim']
	n_dim = config_dict['training_n_dim'] if dataset=="training" else config_dict['testing_n_dim']
	n_distractors = config_dict['n_distractors']

	targets, candidate_sets, labels = [], [], []
	for _ in range(n_dim):
		candidate_set = [np.random.randint(0,2,size=v_dim) for idx in range(n_distractors+1)]
		target_idx = np.random.randint(n_distractors+1)
		target = candidate_set[target_idx]
		
		candidate_sets.append(candidate_set)
		labels.append(target_idx)
		targets.append(target)

	## Convert scalar labels to vector
	labels = to_categorical(labels)

	## Return zipped object
	return zip(targets, candidate_sets, labels)
