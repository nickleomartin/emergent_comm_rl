# emergent_comm_rl
This repo will contain reinforcement learning models used in "Emergence of Linguistic Communication from Referential Games with Symbolic and Pixel Input". The paper can be found [here](https://arxiv.org/abs/1804.03984). 

- Can we induce reinforcement learning agents to learn a compositional language using a cooperative task which features an information asymmetry?
- How do environmental pressures affect the generated language?

![alt text](https://raw.githubusercontent.com/NickLeoMartin/emergent_comm_rl/master/images/emergent_comm.png)

Get Setup
---------
Clone the repo:
```
git clone git@github.com:NickLeoMartin/emergent_comm_rl.git
```

Head into the repo and create and activate a virtual environment:
```
virtualenv --no-site-packages -p python3 venv
source venv/bin/activate
```

Install the packages used:
```
pip install -r requirements.txt
```

Running Experiments
-------------------
Parameters for the experiments can be found in config.py.

Run the random baseline with RandomAgents:
```
python -m experiments.random_agent_experiments

Generating training and testing data
Training Agents
Total Reward: 198, Accuracy: 19.8 %
Speaker action distribution: Counter({50: 44, 4: 42, 32: 41, 58: 41, 16: 40, 36: 39, 38: 39, 42: 38, 28: 36, 35: 36, 41: 36, 60: 36, 1: 35, 22: 35, 33: 35, 34: 35, 54: 35, 59: 35, 10: 34, 15: 34, 19: 34, 25: 34, 26: 34, 29: 34, 37: 34, 11: 33, 12: 33, 30: 33, 7: 32, 18: 32, 23: 32, 39: 32, 45: 32, 53: 32, 57: 32, 2: 31, 3: 31, 31: 31, 43: 31, 47: 31, 49: 31, 61: 31, 6: 30, 14: 30, 20: 30, 27: 30, 48: 30, 0: 29, 8: 29, 13: 29, 24: 29, 56: 29, 46: 28, 17: 27, 44: 27, 51: 27, 52: 26, 55: 26, 5: 23, 9: 23, 40: 23, 21: 19})
Listener action distribution: Counter({2: 225, 1: 205, 3: 200, 4: 188, 0: 182})
Evaluating agents on novel input
Total Reward: 200, Accuracy: 20.0 %
Speaker action distribution: Counter({46: 45, 56: 45, 0: 40, 14: 40, 29: 40, 51: 40, 18: 39, 6: 38, 13: 38, 20: 38, 58: 38, 4: 37, 28: 37, 37: 37, 9: 36, 15: 36, 32: 36, 5: 35, 8: 35, 22: 35, 36: 35, 60: 35, 10: 34, 53: 34, 19: 33, 41: 33, 49: 33, 55: 33, 57: 33, 1: 32, 3: 32, 23: 32, 38: 32, 40: 32, 42: 32, 30: 31, 16: 30, 27: 30, 33: 30, 44: 30, 61: 30, 12: 29, 17: 29, 31: 29, 34: 29, 45: 29, 26: 28, 35: 28, 52: 28, 24: 27, 47: 27, 54: 27, 59: 27, 2: 26, 11: 26, 25: 26, 48: 26, 39: 25, 43: 25, 21: 24, 50: 23, 7: 21})
Listener action distribution: Counter({1: 224, 3: 204, 0: 197, 2: 196, 4: 179})
```
We can see that the distribution of speaker and listener actions is relatively uniform. Random performance around 20%. 

Run the Visa Dataset with MLP speaker and listener networks for 10000 batches:
```
python -m experiments.dense_visa_agents_experiments

Getting dataset
Training Agents
Total Reward: 84, Accuracy: 20.63882063882064 %
Speaker action distribution: Counter({25: 407, 18: 407})
Listener action distribution: Counter({0: 407})
Topographical Similarity: 0.06759830381841883

Evaluate Agent Generalisation
Total Reward: 16, Accuracy: 15.686274509803921 %
Speaker action distribution: Counter({25: 102, 18: 102})
Listener action distribution: Counter({0: 102})
Topographical Similarity: 0.1005088994156145
```
We can see that under-trained agents exhibit degenerative policies by sticking to one or two actions irrespective of the inputs. The small topographical similarity suggests that there is negligible correlation between the message similarities and the input vector distances.

To-Do
-----
V1: Random Baseline
- [x] Dummy data generator
- [x] RandomAgent
- [x] Basic training pipeline
- [x] Evaluation on novel inputs 
- [x] Evaluation using topographical similarity

V2: Symbolic Input: MLP/Dense Agents
- [x] Download [Visual Attributes of Concepts dataset](http://homepages.inf.ed.ac.uk/s1151656/resources.html)
- [x] Preprocess symbolic input dataset
- [x] Listener policy network
- [x] Speaker policy network
- [x] Experiments on symbolic input dataset

V3: Symbolic Input: Models from paper
- [ ] Listener model from paper
- [ ] Speaker model from paper 
- [ ] Experiments on symbolic input dataset
- [ ] Jupyter notebook explaining symbolic input experiments

V4: Pixel Input Experiment
- [ ] Listener model from paper
- [ ] Speaker model from paper 
- [ ] Experiments on [MuJoCo](http://www.mujoco.org/) engine dataset or similar task
- [ ] Jupyter notebook explaining pixel input experiments

V5:
- [ ] Extend to more realistic and complex environments e.g. 3D cooperative games
- [ ] Extend to different types of cooperative tasks e.g. OpenAI Gym games
- [ ] Extend to multi-step games and turn-taking
- [ ] Analyse the biases introduced by pre-trained language and vision models on the generated language
- [ ] Transition from symbols to simple words grounded by a coopertative task










