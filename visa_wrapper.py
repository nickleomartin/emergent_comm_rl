import os
import glob
import numpy as np
import xml.etree.ElementTree as ET 




class VisaDatasetWrapper(object):
	""" 
	Handles retrieval and creatiom of Visa Dataset 
	
	Example:
	--------
	from visa_wrapper import VisaDatasetWrapper

	vdw = VisaDatasetWrapper()
	vdw.create_concept_dictionary()
	vdw.create_symbolic_attribute_vectors()

	"""
	def __init__(self, dataset_dir="visa_dataset",file_extension=".xml", ):
		self.dataset_dir = dataset_dir
		self.file_extension = file_extension
		self.attribute_list = []
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

			self.concept_dict[file_category_name] = {} 

			for subcategory in root:
				# print("subcategory: ", subcategory.tag, subcategory.attrib)

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
				else:
					for concept in subcategory:
						concept_attributes = []
						# print("concept: ",concept.tag, concept.attrib)
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

	def create_symbolic_attribute_vectors(self):
		""" """

		## Attribute to id dict
		self.attribute_to_id_dict = {attribute:key for key, attribute in enumerate(self.attribute_list)}

		## Id to attribute dict
		self.id_to_attribute_dict = {key:attribute for attribute, key in self.attribute_to_id_dict.items()}

		## Binary vectors
		self.symbolic_vectors = []
		self.concept_list = []
		n_attributes = len(self.attribute_list)

		for category, subcategory in self.concept_dict.items():
			print("Creating symbolic vectors for %s category"%category)
			for subcategory, items in subcategory.items():
				vect = np.zeros(n_attributes)
				attr_indices = [self.attribute_to_id_dict[item] for item in items]
				vect[attr_indices] = 1.
				self.symbolic_vectors.append(vect)
				self.concept_list.append(subcategory)


