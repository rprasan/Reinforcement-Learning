import gym
import pybullet_envs
import cv2
from numpngw import write_apng

# Models and computation
import torch # will use pyTorch to handle NN 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from random import sample

# Visualization
#import matplotlib
#import matplotlib.pyplot as plt



''' 
    Simple test to verify that you can import the LunarLander environment.  
    The agent samples a random action

'''


env = gym.make("HopperBulletEnv-v0")
render = True
images = []
s = env.reset()
done = False
steps = 0
images = []

while not done:
    action = env.action_space.sample()
    steps += 1

    if render:
        image = env.render('rgb_array')
        images.append(image)
        cv2.imshow("DRL Project 4", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

    obs, reward, done, info = env.step(action)
    if steps > 60:
        break

if render:
    write_apng('rollout.png', images, delay=20)

env.close()


#The observation consists of a vector with 15 continuous variables including the height 
#of the agent's torso, the agent's global orientation in the forward direction, 
#the angular velocity, the joint angles and linear speeds, roll and pitch information, 
#and a Boolean that denotes the state of the foot (1 if the foot is in contact with the 
#ground, 0 otherwise).
print('observation space:', env.observation_space)

#There are three continuous actions that denote the torques applied to the thigh, leg, 
#and foot joints of the agent. The expected range of each of the two actions is [-1, 1]. 
#Please make sure that your policy network outputs values in that range.
print('action space:', env.action_space)

print('solved_score', env.spec.reward_threshold)


def save_checkpoint(model, filename):
    """
    Saves a model to a file 
        
    Parameters
    ----------
    model: your policy network
    filename: the name of the checkpoint file
    """
    torch.save(model.state_dict(), filename)


'''                         
    ############# PPO with clipped objective and GAE ########################### 
    Implement below the Proximal Policy Optimization (PPO) algorithm using a  
    clipped objective and generalized advantage estimatation (GAE) to solve the 
    HopperBulletEnv-v0 enviroment. 
    Please, refer to the project descriprtion and lecture 19 for more details.  
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################## Your code ####################################
