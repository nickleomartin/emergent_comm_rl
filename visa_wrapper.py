import os
import glob
import numpy as np
import xml.etree.ElementTree as ET 
from keras.utils.np_utils import to_categorical

class VisaDatasetWrapper(object):
	""" 
	Handles retrieval and creation of Visa Dataset 
	
	Example:
	--------
	from config import config_dict
	from visa_wrapper import VisaDatasetWrapper

	vdw = VisaDatasetWrapper()
	vdw.create_train_test_datasets(config_dict)
	b1 = vdw.training_batch_generator()

	for b in b1:
		i,j,k = b
		print(i,j,k)
	"""
	def __init__(self, dataset_dir="visa_dataset",file_extension=".xml", ):
		self.dataset_dir = dataset_dir
		self.file_extension = file_extension
		self.attribute_list = []
		self.concept_list = []
		self.attribute_to_id_dict = {}
		self.id_to_attribute_dict = {}
		self.retrieve_xml_file_names()

	def retrieve_xml_file_names(self, dataset_dir="visa_dataset",file_extension=".xml"): 	
		""" Loop through files in dataset_dir and get names of those with extension """
		self.xml_file_names = []
		for file_name in os.listdir(dataset_dir):
			if file_name.endswith(file_extension):
				self.xml_file_names.append(os.path.join(dataset_dir,file_name))
	
	def get_category_name_from_file(self, file_path):
		""" Get category from xml file name """
		return os.path.basename(file_path).split("_")[0].lower()

	## TODO: Refactor to remove redundancy! 
	def create_concept_dictionary(self):
		""" Create concept dictionary """
		self.concept_dict = {}

		for xml_file_name in self.xml_file_names:
			
			## Get file name
			file_category_name = self.get_category_name_from_file(xml_file_name)
			print("Reading in XML file for %s"%file_category_name)

			## Parse ElementTree
			tree = ET.parse(xml_file_name)

			## Get root of tree
			root = tree.getroot()

			## Define category's dict
			self.concept_dict[file_category_name] = {} 

			for subcategory in root:

				if subcategory.tag=="concept":
					concept_attributes = []
					for item in subcategory:
						for attribute in item.text.split("\n"):
							if attribute.strip()!="":
								string_attr = attribute.replace("\t","").strip()
								if string_attr not in self.attribute_list:
									self.attribute_list.append(string_attr)
								concept_attributes.append(string_attr)

					## Add list of attributes for concept
					concept_name = subcategory.attrib["name"]
					self.concept_dict[file_category_name][concept_name] = concept_attributes
					self.concept_list.append(concept_name)
				else:
					for concept in subcategory:
						concept_attributes = []
						for item in concept:
							for attribute in item.text.split("\n"):
								if attribute.strip()!="":
									string_attr = attribute.replace("\t","").strip()
									if string_attr not in self.attribute_list:
										self.attribute_list.append(string_attr)
									concept_attributes.append(string_attr)

						## Add list of attributes for concept
						concept_name = concept.attrib["name"] 
						self.concept_dict[file_category_name][concept_name] = concept_attributes
						self.concept_list.append(concept_name)

	def create_symbolic_attribute_vectors(self):
		""" """

		## Attribute to id dict
		self.attribute_to_id_dict = {attribute:key for key, attribute in enumerate(self.attribute_list)}

		## Id to attribute dict
		self.id_to_attribute_dict = {key:attribute for attribute, key in self.attribute_to_id_dict.items()}

		## Binary vectors
		self.symbolic_vectors = []
		n_attributes = len(self.attribute_list)

		for category, subcategory in self.concept_dict.items():
			print("Creating symbolic vectors for %s category"%category)
			for subcategory, items in subcategory.items():
				vect = np.zeros(n_attributes)
				attr_indices = [self.attribute_to_id_dict[item] for item in items]
				vect[attr_indices] = 1.
				self.symbolic_vectors.append(vect)

	def sample_target_idx(self, dataset="training"):
		""" """
		if dataset=="training":
			return np.random.randint(0, self.n_training_rows)

		elif dataset=="testing":
			return np.random.randint(0, self.n_testing_rows)


	def negatively_sample_distractors(self, target_idx, n_dataset_rows, n_distractors):
		""" Negatively sample n_distractors """
		distractors_idx = []
		while len(distractors_idx) < n_distractors:
			sampled_idx = np.random.randint(0, n_dataset_rows)
			if sampled_idx!=target_idx:
				distractors_idx.append(sampled_idx)
		return distractors_idx

	def create_train_test_datasets(self, config_dict):
		""" Randomly split dataset into train and test sets """
		self.config_dict = config_dict
		self.batch_size = config_dict["batch_size"]
		self.n_distractors = config_dict["n_distractors"]

		## TODO: Set random seed

		## TODO: Add comments
		self.create_concept_dictionary()
		self.create_symbolic_attribute_vectors()

		## Add test-train split parameter 
		self.train_split_percent = config_dict["train_split_percent"] 
		self.n_dataset_rows = len(self.concept_list)
		self.n_training_rows = int(round(self.n_dataset_rows*self.train_split_percent, 0))
		self.n_testing_rows = self.n_dataset_rows - self.n_training_rows

		## Generate random indixes to partition train and test set
		indices = np.random.permutation(len(self.symbolic_vectors))
		training_indices, test_indices = indices[:self.n_training_rows], indices[self.n_training_rows:]
		self.training_set = np.array([self.symbolic_vectors[idx] for idx in training_indices])
		self.testing_set = np.array([self.symbolic_vectors[idx] for idx in test_indices])

	def categorical_label(self, label):
		""" Convert scalar idx into array """
		l = np.zeros(self.n_distractors+1)
		l[label] = 1.
		return l

	def training_batch_generator(self):
		""" """
		for i in range(self.batch_size):
			sampled_target_idx = self.sample_target_idx(dataset="training")
			distractors_idx = self.negatively_sample_distractors(sampled_target_idx, self.n_training_rows, self.n_distractors)

			## Naive shuffling with record. TODO: improve..
			rand_idx = np.random.randint(0, self.n_distractors+1)
			candidate_idx_set = []
			for dist_idx in distractors_idx:
				if i==rand_idx:
					candidate_idx_set.append(sampled_target_idx)
				candidate_idx_set.append(dist_idx)

			target = self.training_set[sampled_target_idx]
			candidate_set = self.training_set[candidate_idx_set]
			y_label = self.categorical_label(rand_idx)
			yield target, candidate_set, y_label


	def testing_batch_generator(self):
		""" """
		for idx in range(self.n_testing_rows):
			distractors_idx = self.negatively_sample_distractors(idx, self.n_testing_rows, self.n_distractors)

			## Naive shuffling with record. TODO: improve..
			rand_idx = np.random.randint(0, self.n_distractors+1)
			candidate_idx_set = []
			for dist_idx in distractors_idx:
				if idx==rand_idx:
					candidate_idx_set.append(idx)
				candidate_idx_set.append(dist_idx)

			target = self.testing_set[idx]
			candidate_set = self.testing_set[candidate_idx_set]
			y_label = self.categorical_label(rand_idx)
			yield target, candidate_set, y_label

