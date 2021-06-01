import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DQN_PRED(nn.Module):
    def __init__(self, lr):
        super().__init__()
        self.lin = nn.Sequential(
                                nn.Linear(4, 20, bias=True),
                                nn.BatchNorm2d(10),
                                nn.ReLU(inplace=True),
                                nn.Linear(20, 10, bias=True),
                                )
        self.optimiser = optim.Adam(lr = lr)
        self.loss = nn.MSELoss()

    def forward(self, x):
        inp = T.tensor(x)
        inp = self.lin(inp)
        return inp

class Agent(object):
    def __init__(self, gamma, epsilon, lr, batch_size, max_mem_size = 10000, eps_end = 0.01, eps_dec = 0.996):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.mem_size = max_mem_size
        self.mem_cntr = 0

        self.Q_eval = DQN_PRED(lr)

        self.state_memory = np.zeros((self.mem_size, 4))             # (10000,4) STATE VECTOR CONTAINS 4 VALUES
            
        self.action_memory = np.zeros(self.mem_size, dtype=np.float32)

        self.reward_memory = np.zeros(self.mem_size)
            
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
    

    def store_transitions(self, state, action, reward, state_, terminal):     # STORING THE TRANSITIONS IN REPLAY MEMORY
        index = self.mem_cntr % self.mem_size
            
        self.state_memory[index] = state

        self.action_memory[index] = action
            
        self.reward_memory[index] = reward
        self.state_memory[index+1] = state_
        self.terminal_memory[index] = 1 - terminal                                                              
            
        self.mem_cntr += 1

    def choose_actions(self, observation):                                   # EPSILON GREEDY EXPLORATION
        rand = np.random.random()

        if rand < self.epsilon:
            action = np.random.choice(list(range(10)))
        else:
            actions = self.Q_eval.forward(observation)
            action = T.argmax(actions).item()

        return action

    def learn(self):
        if self.mem_cntr > self.batch_size:

            self.Q_eval.optimizer.zero_grad()                               # FOR NULLIFYING GRADIENTS OF PREV STEPS OR PREV BATCHES

            max_mem = self.mem_cntr \
                if self.mem_cntr < self.mem_size else self.mem_size         # NOT TO EXCEED THE TOTAL REPLAY MEMORY SIZE

            batch = np.random.choice(max_mem, self.batch_size)              # SELECTING BATCHSIZE NO. OF SAMPLES FROM MAXMEM RANDOMLY

            state_batch = self.state_memory[batch]

            action_batch = self.action_memory[batch]
                
            reward_batch = self.reward_memory[batch]
            terminal_batch = self.terminal_memory[batch]
            new_state_batch = self.state_memory[batch + 1]

            reward_batch = T.Tensor(reward_batch)
            terminal_batch = T.Tensor(terminal_batch)

            q_s_a = self.Q_eval.forward(state_batch)                                 # Q_PRED
            q_target = q_s_a.clone()

            q_S_A = self.Q_eval.forward(new_state_batch)

            batch_index = np.arange(self.batch_size, dtype=np.int32)
        
            q_target[batch_index, action_batch] = reward_batch \
                        + self.gamma * T.max(q_S_A, dim=1)[0] * terminal_batch       # Q_TARGET

            self.epsilon = self.epsilon * self.eps_dec \
                if self.epsilon > self.eps_min else self.eps_min 

            loss = self.Q_eval.loss(q_target, q_s_a)
            loss.backward()

            self.Q_eval.optimizer.step()

    
