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
Total Reward: 193, Accuracy: 19.3 %
Speaker action distribution: Counter({35: 41, 61: 41, 2: 40, 17: 40, 21: 40, 53: 40, 31: 39, 40: 39, 52: 39, 3: 38, 41: 38, 46: 38, 24: 37, 43: 37, 4: 36, 20: 36, 51: 36, 5: 35, 7: 35, 9: 35, 29: 35, 36: 35, 57: 35, 10: 34, 37: 34, 58: 34, 1: 33, 42: 33, 48: 33, 26: 32, 49: 32, 11: 31, 23: 30, 25: 30, 30: 30, 39: 30, 44: 30, 47: 30, 50: 30, 14: 29, 16: 29, 33: 29, 56: 29, 60: 29, 0: 28, 8: 28, 12: 28, 19: 28, 22: 28, 38: 28, 54: 28, 59: 28, 6: 27, 15: 27, 18: 27, 27: 27, 34: 27, 32: 26, 45: 26, 55: 25, 13: 24, 28: 24})
Listener action distribution: Counter({1: 213, 0: 210, 3: 208, 2: 189, 4: 180})
Topographical Similarity: -0.0006967304579020988

Evaluating agents on novel input
Total Reward: 170, Accuracy: 17.0 %
Speaker action distribution: Counter({13: 47, 19: 47, 15: 44, 44: 44, 21: 43, 51: 42, 35: 41, 10: 39, 38: 39, 55: 39, 28: 37, 8: 36, 14: 36, 47: 36, 2: 35, 4: 35, 5: 35, 46: 35, 56: 35, 7: 34, 11: 34, 20: 34, 24: 34, 25: 34, 34: 34, 42: 34, 0: 33, 9: 33, 43: 33, 53: 33, 54: 33, 59: 33, 26: 32, 29: 32, 40: 32, 12: 31, 30: 31, 16: 30, 37: 30, 45: 30, 22: 29, 23: 29, 32: 29, 57: 29, 3: 28, 18: 28, 27: 28, 39: 28, 52: 28, 61: 28, 6: 27, 17: 27, 33: 26, 50: 26, 58: 26, 1: 24, 49: 24, 36: 23, 48: 23, 41: 22, 60: 22, 31: 17})
Listener action distribution: Counter({2: 214, 4: 201, 3: 196, 0: 195, 1: 194})
Topographical Similarity: -0.0022818464956945307
```
We can see that the distribution of speaker and listener actions is relatively uniform. Random performance around 20%. 

Run the Visa Dataset with MLP speaker and listener networks for 100 batches:
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










