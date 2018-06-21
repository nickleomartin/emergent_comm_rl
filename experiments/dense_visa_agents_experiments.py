from config import visa_config_dict as config_dict
from data_generator import generate_dummy_categorical_dataset
from evaluation import obtain_metrics
from rl.agents import VisaAgents
from rl.speaker_policy_networks import DenseSpeakerPolicyNetwork
from rl.listener_policy_networks import DenseListenerPolicyNetwork
from visa_wrapper import VisaDatasetWrapper 


print("Get dataset")
data_generator = VisaDatasetWrapper()
data_generator.create_train_test_datasets(config_dict)

print("Train Agents")
speaker = DenseSpeakerPolicyNetwork(config_dict)
listener = DenseListenerPolicyNetwork(config_dict)

da = VisaAgents(config_dict,speaker,listener)
da.fit(data_generator)
#obtain_metrics(da.training_stats, config_dict)

print("Evaluate Agent Generalisation")
da.predict(data_generator)
obtain_metrics(da.testing_stats,config_dict)


