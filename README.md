# Teach-a-Quadcopter-How-to-Fly
Reinforcement Learning(RL) Project [Udacity Deep Learning Nanodegree]

## Project Overview
### Project Description
Use Deep Deterministic Policy Gradients(DDPG), an actor-critic method, to build the agent that can learn takeoff task on its own. 
Shape reward functions and train the agent to achieve this task.

### Project Procedure
- Define the taskoff task
  - Reach the target height(Z-axis value) of 10 units above the ground
  - Shape reward function
- Design the agent
  - Build critic model
  - Build actor model
  - Build DDPG agent
- Train the agent
- Evaluate agent's performance

### Project Results
- Achieved at most the height of 5 units on testing.

### Reflection
- Craft reward functions is critical to the agent's performance.
- Need more professional parameter estimation techniques and have further knowledge of drone physics to shape better reward functions.
- Balance reward and penalty.


## Getting Started
### Prerequisites
This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)


### Run
In a terminal or command window, run one of the following commands:

```bash
ipython notebook Your_first_neural_network.ipynb
```  
or
```bash
jupyter notebook Your_first_neural_network.ipynb
```

This will open the iPython Notebook software and project file in your browser.

## Reference
[Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf)
