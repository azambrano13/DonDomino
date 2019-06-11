import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.utils import shuffle
from collections import namedtuple
import copy
import itertools
import time as t
import matplotlib.pyplot as plt
import numpy as np
import random
from copy import deepcopy
from torch.autograd import Variable

def calcular_recompenza(quien_gano,LAMBDA,jugadas_invalidas):
    veces=len(jugadas_invalidas)
    recompenzas=[]
    if quien_gano==0:
        base=np.array([LAMBDA]*veces)
        exponente=np.array(range(len(base)))
        recompenzas=np.power(base,exponente)*1
        recompenzas=list(recompenzas)
        recompenzas.reverse()
    else:
        base = np.array([LAMBDA] * veces)
        exponente = np.array(range(len(base)))
        recompenzas = np.power(base, exponente) * -1
        recompenzas = list(recompenzas)
        recompenzas.reverse()
    # if quien_gano==0:
    #     base = np.array([0] * veces)
    #     base[-1]=1
    #     recompenzas=base
    # else:
    #     base = np.array([0] * veces)
    #     base[-1] = -1
    #     recompenzas = base
    recompenzas=np.array(recompenzas)
    recompenzas[jugadas_invalidas]=-10

    return recompenzas


def update_policy(policy,optimizer):
    R = 0
    rewards = []

    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    if len(rewards) > 1 : rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1))

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and intialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor())
    policy.reward_episode = []

class Policy(nn.Module):

    def __init__(self,dim_state,gamma=0.9):
        super(Policy, self).__init__()
        self.state_space = dim_state
        self.action_space = 28*2

        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_space, bias=False)

        self.gamma = gamma

        # Episode policy and reward history
        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
            model = torch.nn.Sequential(
                self.l1,
                nn.Dropout(p=0.6),
                nn.ReLU(),
                self.l2,
                nn.Softmax(dim=-1)
            )
            return model(x)

