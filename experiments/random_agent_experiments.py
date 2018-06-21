from config import config_dict
from data_generator import generate_dummy_categorical_dataset
from agents import RandomAgent
from evaluation import obtain_metrics

""" Create data """
print("Generating training and testing data")
train_data = generate_dummy_categorical_dataset(config_dict,"training")
test_data = generate_dummy_categorical_dataset(config_dict,"testing")

""" Train Agents """
print("Training agents")
ra = RandomAgent(config_dict)
ra.fit(train_data)
obtain_metrics(ra.training_stats, config_dict)

""" Evaluate Agent Generalisation """
print("Evaluating agents on novel input")
ra.predict(test_data)
obtain_metrics(ra.testing_stats, config_dict)

