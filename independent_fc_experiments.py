from config import config_dict
from data_generator import generate_dummy_data
from networks import IndependentFullyConnectedAgents
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
ifca = IndependentFullyConnectedAgents(config_dict)
ifca.fit(train_data)
training_stats = ifca.training_stats
obtain_metrics(training_stats)

# ###################################
# ## Evaluate Agent Generalisation ##
# ###################################
# print("Evaluating agents on novel input")
# ifca.predict(test_data)
# testing_stats = ifca.testing_stats
# obtain_metrics(testing_stats)






"""
#------------- Legacy Code -------------
from config import config_dict
alphabet_size = config_dict["alphabet_size"]


model = Sequential()
model.add(Dense(50,activation="relu",input_dim=50))
# model.add(BatchNormalization())
# model.add(Flatten())
model.add(LSTM(50))
# model.add(Flatten())
rms = RMSprop(lr=0.0001)
model.compile(loss="mse",optimizer=rms)


xi = train_data[0][0].reshape([1,50])
model.predict(xi,batch_size=1)

#-----------------------
from config import config_dict
alphabet_size = config_dict["alphabet_size"]


model = Sequential()
# model.add(Dense(50,activation="relu", input_shape=(50,)))
model.add(LSTM(50,input_shape=(50,1)))
model.add(BatchNormalization())
model.add(Dense(alphabet_size,activation="relu"))
rms = RMSprop(lr=0.0001)
model.compile(loss="mse",optimizer=rms)

## Predict softmax output
xi = train_data[0][0].reshape([1,50,1])
probs = model.predict(xi,batch_size=1)
normalized_probs = probs/np.sum(probs)

print(np.random.choice(alphabet_size,1,p=normalized_probs[0]))
print(np.random.choice(alphabet_size,1,p=normalized_probs[0]))

"""