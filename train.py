from config import config_dict
from data_generator import generate_dummy_data
from networks import RandomAgent


#################
## Create data ##
#################
train_data = generate_dummy_data(v_dim=50,n_dim=100)

#######################
## Initialize Agents ##
#######################
ra = RandomAgent(config_dict)
ra.fit(train_data)




















