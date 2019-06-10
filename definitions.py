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

