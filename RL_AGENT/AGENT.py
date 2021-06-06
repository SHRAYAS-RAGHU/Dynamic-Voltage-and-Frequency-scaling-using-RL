import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
                                                                                            # CREATING A CLASS FOR DEFINING THE FUNCTION APPROXIMATOR FOR DQN
class DQN_PRED(nn.Module):
    def __init__(self, lr):
        super().__init__()
        self.lin = nn.Sequential(
                                nn.Linear(4, 25, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(25, 10, bias=True),
                                )
        self.optimizer = optim.Adam(self.lin.parameters(),lr=lr)
        self.loss = nn.MSELoss()                                                            # MSE LOSS USED, (Q_TARGET - Q_PRED) ** 2

    def forward(self, x):
        inp = T.tensor(x)
        inp = T.flatten(inp, start_dim=1)
        #inp = inp.unsqueeze(1)
        inp = self.lin(inp)
        return inp

class Agent(object):
    def __init__(self, gamma, epsilon, lr, batch_size):
        self.gamma = gamma                                                                  # DISCOUNT FACTOR
        self.epsilon = epsilon                                                              # EPS FOR EPSILON-GREEDY
        self.lr = lr                                                                        # LEARNING RATE FOR THE GRADIENT DESCENT GIVEN THROUGH OPTIMISER
        self.batch_size = batch_size
        self.eps_min = 0.01
        self.eps_dec = 0.996
        self.mem_size = 10000                                                               # MAXIMUM REPLAY MEMORY
        self.mem_cntr = 0                                                                   # INDEX FOR REPLAY MEMORY

        self.Q_eval = DQN_PRED(self.lr)                                                          # INSTANCE FOR NEURAL NETWORK CLASS

        self.state_memory = np.zeros((self.mem_size, 1, 4), dtype=np.float32)                                    # (10000,4) STATE VECTOR CONTAINS 4 VALUES
            
        self.action_memory = np.zeros(self.mem_size, dtype=np.float32)

        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
            
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)                    # STORES WHETHER EPISODE IS COMPLETE OR NOT
    

    def store_transitions(self, state, action, reward, state_, terminal):                   # STORING THE TRANSITIONS IN REPLAY MEMORY
        index = self.mem_cntr % self.mem_size                                               # KEEPNG THE INDEX IN RANGE (0, 10000)
            
        self.state_memory[index] = state

        self.action_memory[index] = action
            
        self.reward_memory[index] = reward
        self.state_memory[index+1] = state_
        self.terminal_memory[index] = 1 - terminal                                          # 1-TERMINAL DONE FOR EASY UPDATION OF Q ONLY FOR UNCOMPLETED EPISODES
            
        self.mem_cntr += 1

    def choose_actions(self, observation):                                                  # EPSILON GREEDY EXPLORATION
        rand = np.random.random()

        if rand < self.epsilon:
            action = np.random.choice(list(range(10)))                                     # TAKING RANDOM ACTION FOR EXPLORING VALUES FOR VARIOUS ACTIONS
        else:
            observation = np.reshape(observation, (1,1,4))
            actions = self.Q_eval.forward(observation)
            action = T.argmax(actions).item()                                               # TAKING GREEDY ACTION

        return action

    def learn(self):
        if self.mem_cntr >= self.batch_size:

            self.Q_eval.optimizer.zero_grad()                                               # FOR NULLIFYING GRADIENTS OF PREVIOUS BATCHES

            max_mem = self.mem_cntr \
                if self.mem_cntr < self.mem_size else self.mem_size                         # NOT TO EXCEED THE TOTAL REPLAY MEMORY SIZE

            batch = np.random.choice(max_mem, self.batch_size)                              # SELECTING BATCHSIZE NO. OF SAMPLES FROM MAXMEM RANDOMLY

            state_batch = self.state_memory[batch]
            action_batch = self.action_memory[batch]           
            reward_batch = T.Tensor(self.reward_memory[batch])                              # COLLECTING REQUIRED PARAMETERS FROM TRANSITION MEMORY FOR A GIVEN BATCH
            terminal_batch = T.Tensor(self.terminal_memory[batch])
            new_state_batch = self.state_memory[batch + 1]

            q_s_a = self.Q_eval.forward(state_batch)                                        # Q_PRED FOR THE UPDATE
            q_target = q_s_a.clone()                                                        # TARGET IS Q_PRED ADD WITH THE REWARD AND MAX_VALUE FROM NEXT_STATE

            q_S_A = self.Q_eval.forward(new_state_batch)                                    # MAXIMUM VALUE FROM NEXT STATE

            batch_index = np.arange(self.batch_size, dtype=np.int32)                        # BATCHINDEX VALUES FOR UPDATING THE Q_TARGET

            #print(f'{q_target} \n {action_batch} \n {reward_batch} \n {T.max(q_S_A, dim=1)[0]}')
            q_target[batch_index, action_batch] = reward_batch + \
                          self.gamma * T.max(q_S_A, dim=1)[0] * terminal_batch              # Q_TARGET UPDATE
            #print(f'after update {q_target}')
            self.epsilon = self.epsilon * self.eps_dec \
                if self.epsilon > self.eps_min else self.eps_min                            # UPDATING EPSILON AS EPISODES PROGRESSES TO TAKE MORE OF GREEDY ACTION

            loss = self.Q_eval.loss(q_s_a, q_target)                                        # Q_TARGET - Q_PRED IS THE TD-ERROR AND MSE LOSS FINDS THE SQUARE OF IT
            loss.backward()                                                                 # BACKPROPAGATING THE LOSS TO FING GRADIENTS

            self.Q_eval.optimizer.step()                                                    # UPDATING THE PARAMETERS I.E WEIGHTS USING GRADIENT DESCENT FOR THE Q_VALUE FUNCTION