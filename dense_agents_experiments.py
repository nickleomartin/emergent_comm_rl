from config import config_dict
from data_generator import generate_dummy_categorical_dataset
from agents import DenseAgents
from evaluation import obtain_metrics
from policy_networks import DenseListenerPolicyNetwork, DenseSpeakerPolicyNetwork


""" Create data """
print("Generating training and testing data")
train_data = generate_dummy_categorical_dataset(config_dict,"training")
test_data = generate_dummy_categorical_dataset(config_dict,"testing")


""" Train Agents """
speaker = DenseSpeakerPolicyNetwork(config_dict)
listener = DenseListenerPolicyNetwork(config_dict)

da = DenseAgents(config_dict,speaker,listener)
da.fit(train_data)
obtain_metrics(da.training_stats, config_dict)

""" Evaluate Agent Generalisation """
print("Evaluating agents on novel input")
da.predict(test_data)
obtain_metrics(da.testing_stats,config_dict)


