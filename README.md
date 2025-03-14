# DQN_deadcells
A DQN model to fight with the boss in dead cells (default The Concierge).
Using double DDQN with a prioritized experience buffer



https://github.com/user-attachments/assets/b4ff9013-e671-46fb-b28b-3aae9bbb224a

# Environment
- windows11
- Python 3.9.21
- python library: find in requirments.txt
- Dead Cells
- CUDA and cudnn for pytorch
- Cheat Engine for 2x game speed

# usage
- please make sure there is a ./checkpoints folder under the main folder because I did not do the detection for that.
- if you want to train the new model, please open Cheat Engine and use the speed hack to make Dead Cell run 2x speed
- Then stand in front of this door![image](https://github.com/user-attachments/assets/b747ebca-0bfc-4a51-ba1f-26170a7cfdfb)
- run train.py
- I hard-coded the sleep time for the agent to restart. If there is any problem, please try to adjust the sleep time in the  env.py env._reset() function
- press q to quit training and press p to stop training. Remember to stop the game process by pressing tab or esc when pressing p.

# Code structure
- Most training configuration is in train.py
- Most hyperparameters are in run.py or train.py, near line 20.
- model.py is the Model architecture
- lib/Actions.py is for sending the Action into game
- lib/env.py is for restarting the game and getting the reward for action. If you want to modify the reward strategy, you can modify the env.step function
- lib/GetHp.py is for getting hp for the agent and boss
- lib/GetScreen.py is for getting the screen as the input of the agent
- lib/SendKey.py is for sending the key.

# Acknowledgement
- https://github.com/ailec0623/DQN_HollowKnight
- https://arxiv.org/pdf/1710.02298
