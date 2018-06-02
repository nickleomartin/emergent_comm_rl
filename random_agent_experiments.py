from config import config_dict
from data_generator import generate_dummy_data
from networks import RandomAgent
from evaluation import obtain_metrics

#################
## Create data ##
#################
print("Generating training and testing data")
train_data = generate_dummy_data(config_dict)
test_data = generate_dummy_data(config_dict)

#################################
#################################
print("Training agents")
ra = RandomAgent(config_dict)
ra.fit(train_data)
training_stats = ra.training_stats
obtain_metrics(training_stats)

###################################
## Evaluate Agent Generalisation ##
###################################
print("Evaluating agents on novel input")
ra.predict(test_data)
testing_stats = ra.testing_stats
obtain_metrics(testing_stats)

