import numpy as np 
from scipy.stats import spearmanr
import Levenshtein as levenshtein

def task_accuracy_metrics(reward_list):
	""" Accuracy as percentage of examples that received rewards """
	accuracy = sum(reward_list)*100/float(len(reward_list))
	print("Total Reward: %s, Accuracy: %s %%"%(sum(reward_list),accuracy))
	return accuracy

def levenshtein_message_distance(m1, m2):
	""" Use python-levenshtein package to calculate edit distance """
	message_length = len(m1) 
	return levenshtein.distance("".join(m1),"".join(m2)) 

def topographic_similarity(target_pairs,message_pairs):
	
	## Calculate levenshtein distance between all message pairs

	## Calculate cosine similarity of target and chosen vectors

	## Calculate negative Spearman correlation between message distances and vector similarities
	# spearmanr() 

	pass

def obtain_metrics(training_stats):
	""" Compute metrics given trianing stats list of dicts"""
	metrics = {}

	## Accuracy
	reward_list = [e['reward'] for e in training_stats]
	metrics['accuracy'] = task_accuracy_metrics(reward_list)

	## Topographic similarity
	metrics['topographical_sim'] = None

	return metrics



























































