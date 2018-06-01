import numpy as np 
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
import Levenshtein as levenshtein

def task_accuracy_metrics(reward_list):
	""" Accuracy as percentage of examples that received rewards """
	accuracy = sum(reward_list)*100/float(len(reward_list))
	print("Total Reward: %s, Accuracy: %s %%"%(sum(reward_list),accuracy))
	return accuracy

def levenshtein_message_distance(m1, m2):
	""" Use python-levenshtein package to calculate edit distance """
	return levenshtein.distance(m1,m2) 

def topographic_similarity(input_vectors,messages):
	""" 
	Calculate negative spearman correlation between message levenshtein distances
	and cosine similarities of input vectors 
	"""
	## Calculate levenshtein distance between all message pairs
	message_similarities = []
	for idx, message in enumerate(messages):
		other_messages = messages
		other_messages.pop(idx)
		for other_message in messages:
			lev_dist = levenshtein_message_distance(message, other_message)
			message_similarities.append(lev_dist)	

	## Calculate cosine similarity of target and chosen vectors
	input_vect_similarities = []
	for idx, input_vect in enumerate(input_vectors):
		other_input_vectors = input_vectors
		other_input_vectors.pop(idx)
		for other_input in other_input_vectors:
			cos_dist = cosine(input_vect,other_input)
			input_vect_similarities.append(cos_dist)

	## Calculate negative Spearman correlation between message distances and vector similarities
	rho = spearmanr(message_similarities,input_vect_similarities) 

	return - rho.correlation

def obtain_metrics(training_stats):
	""" Compute metrics given trianing stats list of dicts"""
	metrics = {}

	## Accuracy
	reward_list = [e['reward'] for e in training_stats]
	metrics['accuracy'] = task_accuracy_metrics(reward_list)

	## Topographic similarity
	input_vectors = [e['input'] for e in training_stats]
	messages = [e['message'] for e in training_stats]
	metrics['topographical_sim'] = topographic_similarity(input_vectors,messages)

	return metrics



























































