import numpy as np 
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
import Levenshtein as levenshtein
import collections

def task_accuracy_metrics(reward_list):
	""" Accuracy as percentage of examples that received rewards """
	accuracy = sum(reward_list)*100/float(len(reward_list))
	print("Total Reward: %s, Accuracy: %s %%"%(sum(reward_list),accuracy))
	return accuracy

def action_distribution(action_list):
	counter = collections.Counter(action_list)
	return counter

def levenshtein_message_distance(m1, m2):
	""" Use python-levenshtein package to calculate edit distance """
	return levenshtein.distance(m1,m2) 

def message_sequence_to_alphabet(message, alphabet):
	return "".join(alphabet[int(idx)] for idx in message)

def topographic_similarity(input_vectors, messages):
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

def obtain_metrics(training_stats, config_dict):
	""" Compute metrics given trianing stats list of dicts"""
	metrics = {}

	## Accuracy
	reward_list = [e['reward'] for e in training_stats]
	metrics['accuracy'] = task_accuracy_metrics(reward_list)

	## Speaker action distribution 
	action_list = [e["chosen_target_idx"] for e in training_stats]
	metrics["listener_action_dist"] = action_distribution(action_list)
	print("Listener action distribution: %s"%(metrics["listener_action_dist"]))

	message_list = []
	for e in training_stats:
		for m in e["message"]:
			message_list.append(m)

	metrics["speaker_action_dist"] = action_distribution(message_list)
	print("Speaker action distribution: %s"%(metrics["speaker_action_dist"]))

	## Topographic similarity
	input_vectors = [e['input'] for e in training_stats]
	messages = [message_sequence_to_alphabet(e['message'], config_dict['alphabet']) for e in training_stats]
	metrics['topographical_sim'] = topographic_similarity(input_vectors, messages)

	return metrics



























































