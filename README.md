# Deep Reinforcement Learning for Navigation

A bananas collector navigation problem of Unity ML-Agents is solved with DRL using DQN and some improvements 

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6 with conda.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnavigation python=3.6
	source activate drlnavigation
	```
	- __Windows__: 
	```bash
	conda create --name drlnavigation python=3.6 
	activate drlnavigation
	```

2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnavigation --display-name "drlnavigation"
```