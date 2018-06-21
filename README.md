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
Run the random baseline with RandomAgents:
```
python -m experiments.random_agent_experiments
```

Run the random baseline with DenseAgents:
```
python -m experiments.dense_agents_experiments
```

To-Do
-----
V1: Random Baseline
- [x] Dummy data generator
- [x] RandomAgent
- [x] Basic training pipeline
- [x] Evaluation on novel inputs 
- [x] Evaluation using topographical similarity
- [ ] Jupyter notebook explaining random baseline

V2: Dense Agents
- [x] Listener policy network
- [x] Speaker policy network
- [x] Random dataset baseline
- [ ] Jupyter notebook explaining DenseAgent random baseline

V3: Symbolic Input Experiment
- [x] Download [Visual Attributes of Concepts dataset](http://homepages.inf.ed.ac.uk/s1151656/resources.html)
- [x] Preprocess symbolic input dataset
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










