from config import visa_config_dict as config_dict
from data_generator import generate_dummy_categorical_dataset
from evaluation import obtain_metrics
from rl.agents import VisaAgents
from rl.speaker_policy_networks import DenseSpeakerNetwork
from rl.listener_policy_networks import DenseListenerNetwork
from visa_wrapper import VisaDatasetWrapper 


print("Get dataset")
data_generator = VisaDatasetWrapper()
data_generator.create_train_test_datasets(config_dict)

print("Train Agents")
speaker = DenseSpeakerNetwork(config_dict)
listener = DenseListenerNetwork(config_dict)

agent = VisaAgents(config_dict,speaker,listener)
agent.fit(data_generator)

print("Evaulating on training set")
agent.evaluate_on_training_set(data_generator)
obtain_metrics(agent.training_eval_stats, config_dict)

print("Evaluate Agent Generalisation")
agent.predict(data_generator)
obtain_metrics(agent.testing_stats,config_dict)


