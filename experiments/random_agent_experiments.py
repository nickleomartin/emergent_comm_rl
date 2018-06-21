from config import random_config_dict as config_dict
from data_generator import generate_dummy_categorical_dataset
from evaluation import obtain_metrics
from rl.agents import RandomBaselineAgents
from rl.speaker_policy_networks import RandomSpeakerPolicyNetwork
from rl.listener_policy_networks import RandomListenerPolicyNetwork


""" Create data """
print("Generating training and testing data")
train_data = generate_dummy_categorical_dataset(config_dict,"training")
test_data = generate_dummy_categorical_dataset(config_dict,"testing")

print("Training Agents")
speaker = RandomSpeakerPolicyNetwork(config_dict)
listener = RandomListenerPolicyNetwork(config_dict)
agent = RandomBaselineAgents(config_dict, speaker, listener)
agent.fit(train_data)
obtain_metrics(agent.training_stats, config_dict)

# """ Evaluate Agent Generalisation """
print("Evaluating agents on novel input")
agent.predict(test_data)
obtain_metrics(agent.testing_stats, config_dict)


