import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

s = [51.1, 0.85, 14.07, 9.849]

class DQN_PRED(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Sequential(
                                nn.Linear(4, 20, bias=True),
                                nn.BatchNorm2d(10),
                                nn.ReLU(inplace=True),
                                nn.Linear(20, 10, bias=True),
                                nn.Softmax(dim=1)
                                )
    def forward(self, x):
        inp = T.tensor(x)
        #inp = inp.unsqueeze(dim=0)#batchsize,d,h,w
        #inp = inp.unsqueeze(dim=1)
        inp = self.lin(inp)
        return inp
class Agent(object):
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size = 10000, eps_end = 0.01, eps_dec = 0.996):
        self.gamma = gamma
        self.epsilon = epsilon
        self.input_dims = input_dims
        self.lr = lr
        self.batch_size = batch_size
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.n_actions = n_actions
        self.mem_size = max_mem_size
        self.mem_cntr = 0

        self.action_space = np.arange(0, self.n_actions, dtype=np.int32)

        self.Q_eval = DQN_PRED(lr, input_dims=input_dims, fc1_dim=512, fc2_dim=512, n_actions=self.n_actions)

        self.state_memory = np.zeros((self.mem_size, *input_dims))#(10000,4) 4 states are there
            
        self.action_memory = np.zeros((self.mem_size, self.n_actions), dtype=np.uint8)#(10000,10) 10 actions

        self.reward_memory = np.zeros(self.mem_size)
            
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)
    

    def store_transitions(self, state, action, reward, state_, terminal):#for storing the parameters during the training
        index = self.mem_cntr % self.mem_size
            
        self.state_memory[index] = state

        actions = np.zeros(self.n_actions)
        actions[action] = 1.0
        self.action_memory[index] = actions
            
        self.reward_memory[index] = reward
        self.state_memory[index+1] = state_
        self.terminal_memory[index] = 1 - terminal                                                              
            
        self.mem_cntr += 1

    def choose_actions(self, observation):#epsilon greedy policy has been used here to choose actions
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.Q_eval.forward(observation)
            action = T.argmax(actions).item()
        return action

    def learn(self):
        if self.mem_cntr > self.batch_size:

            self.Q_eval.optimizer.zero_grad()

            max_mem = self.mem_cntr if self.mem_cntr < self.mem_size else self.mem_size # not to exceed the total mem size.

            batch = np.random.choice(max_mem, self.batch_size)#selecting batchsize no of samples from max_mem randomly

            state_batch = self.state_memory[batch]

            action_batch = self.action_memory[batch]
            action_indices = np.array(np.where(action_batch == 1)[1])#identifying which action among 10 actions is set to one
                
            reward_batch = self.reward_memory[batch]
            terminal_batch = self.terminal_memory[batch]
            new_state_batch = self.state_memory[batch + 1]

            reward_batch = T.Tensor(reward_batch)
            terminal_batch = T.Tensor(terminal_batch)

            q_eval = self.Q_eval.forward(state_batch)
                
            q_target = q_eval.clone()
            q_next = self.Q_eval.forward(new_state_batch)
            batch_index = np.arange(self.batch_size, dtype=np.int32)
        
            q_target[batch_index, action_indices] = reward_batch + self.gamma * T.max(q_next, dim=1)[0] * terminal_batch #bellman equation to update the q value

            self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min else self.eps_min
            loss = self.Q_eval.loss(q_target, q_eval)
            loss.backward()

            self.Q_eval.optimizer.step()

    
