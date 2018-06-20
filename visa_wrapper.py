import os
import glob
import xml.etree.ElementTree as ET 

## Glob all file in visa_dataset/





class VisaDatasetWrapper(object):
	""" 
	Handles retrieval and creatiom of Visa Dataset 
	
	Example:
	--------
	from visa_wrapper import VisaDatasetWrapper

	vdw = VisaDatasetWrapper()
	vdw.create_concept_dictionary()
	"""

	def __init__(self, dataset_dir="visa_dataset",file_extension=".xml", ):
		self.dataset_dir = dataset_dir
		self.file_extension = file_extension
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

	def create_concept_dictionary(self):
		""" Create concept dictionary """
		self.concept_dict = {}

		for xml_file_name in self.xml_file_names:
			
			## Get file name
			file_category_name = self.get_category_name_from_file(xml_file_name)
			print(file_category_name)

			## Parse ElementTree
			tree = ET.parse(xml_file_name)

			## Get root of tree
			root = tree.getroot()

			self.concept_dict[file_category_name] = {} 

			for subcategory in root:
				print("subcategory: ", subcategory.tag, subcategory.attrib)

				if subcategory.tag=="concept":
					concept_attributes = []
					for item in subcategory:
						for attribute in item.text.split("\n"):
							if attribute!="":
								concept_attributes.append(attribute.replace("\t",""))

					## Add list of attributes for concept
					concept_name = subcategory.attrib["name"]
					self.concept_dict[file_category_name][concept_name] = concept_attributes
				else:
					for concept in subcategory:
						concept_attributes = []
						print("concept: ",concept.tag, concept.attrib)
						for item in concept:
							for attribute in item.text.split("\n"):
								if attribute!="":
									concept_attributes.append(attribute.replace("\t",""))

						## Add list of attributes for concept
						concept_name = concept.attrib["name"] 
						self.concept_dict[file_category_name][concept_name] = concept_attributes





