#%%
import numpy as np
import matplotlib as plt
import math
import random
from itertools import count
from collections import namedtuple
from collections import deque
import ipynb
from Game_AI import SnakeAI, Point, Direction
from ipynb.fs.full.Model import SnakeModel, Trainer, SnakeModel2, SnakeNet
from ipynb.fs.full.plot import make_plot

#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#%%
# By-pass OMP: error
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#%%
Capacity = 100000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=Capacity) 
        # self.model = SnakeNet(480, 640, 3) # Not really working
        # self.model = SnakeModel()
        self.model = SnakeModel2()
        self.trainer = Trainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        # Performance less checks
        cw_dirs = [
            Direction.RIGHT == game.direction, 
            Direction.DOWN == game.direction,
            Direction.LEFT == game.direction,
            Direction.UP == game.direction]
        
        cw_angs = np.array([0, np.pi/2, np.pi, -np.pi/2])

        # Position - straight: 0, right: 1, left: -1
        getPoint = lambda pos: Point(
            head.x + 20*np.cos(cw_angs[(cw_dirs.index(True)+pos) % 4]),
            head.y + 20*np.sin(cw_angs[(cw_dirs.index(True)+pos) % 4]))

        state = [
          # Danger
          game.is_collision(getPoint(0)),
          game.is_collision(getPoint(1)),
          game.is_collision(getPoint(-1)),

          # Move direction
          cw_dirs[2],
          cw_dirs[0],
          cw_dirs[3],
          cw_dirs[1],
            
            # Food location 
          game.food.x < game.head.x,  # food left
          game.food.x > game.head.x,  # food right
          game.food.y < game.head.y,  # food up
          game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 100000:
            self.memory.popleft() 

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Episode', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            make_plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()