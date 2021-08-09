import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path
import datetime
import copy
                                                                                            # CREATING A CLASS FOR DEFINING THE FUNCTION APPROXIMATOR FOR DQN
class DQN_PRED(nn.Module):
    def __init__(self, lr):
        super().__init__()
        self.lin = nn.Sequential(
                                nn.Linear(5, 256, bias=True),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(256, 256, bias=True),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(256, 10, bias=True)
                                )
        self.tgt = copy.deepcopy(self.lin)
        for p in self.tgt.parameters():
          p.requires_grad = False

    def forward(self, x, model):
        #inp = T.tensor(x)
        inp = T.flatten(x, start_dim=1)
        if model == 'online':
          return self.lin(inp)
        else:
          return self.tgt(inp)

class Agent(object):
    def __init__(self, gamma = 0.999, epsilon = 1, lr = 0.00025, batch_size = 16):
        self.gamma = gamma                                                                  # DISCOUNT FACTOR
        self.epsilon = epsilon                                                              # EPS FOR EPSILON-GREEDY
        self.lr = lr                                                                        # LEARNING RATE FOR THE GRADIENT DESCENT GIVEN THROUGH OPTIMISER
        self.batch_size = batch_size
        self.eps_min = 0.1
        self.eps_dec = 0.995
        self.mem_size = 1000                                                             # MAXIMUM REPLAY MEMORY
        self.mem_cntr = 0   
        
        self.train_start = batch_size + 1
        self.log = 1e4     
        self.tgt_update = 500

        self.q_log = r'/home/pi/Desktop/PROJECT/RL_AGENT/New_training/Q_LOG/Q_NEW' 
        self.training_data = {}                                              

        self.Q_eval = DQN_PRED(self.lr)                                                          # INSTANCE FOR NEURAL NETWORK CLASS

        self.optimizer = optim.SGD(self.Q_eval.parameters(),lr=lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.999)

        self.loss_fn = nn.MSELoss()                                                            # MSE LOSS USED, (Q_TARGET - Q_PRED) ** 2

        self.state_memory = np.zeros((self.mem_size, 1, 5), dtype=np.float32)                                    # (10000,4) STATE VECTOR CONTAINS 4 VALUES
            
        self.action_memory = np.zeros(self.mem_size, dtype=np.float)

        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
            
        #self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)                    # STORES WHETHER EPISODE IS COMPLETE OR NOT

        self.save_dir = Path('/home/pi/Desktop/PROJECT/RL_AGENT/New_training')

        with open(self.q_log, "w") as f:
            f.write(
                f"{'Q0':>11}{'Q1':>11}{'Q2':>11}{'Q3':>11}{'Q4':>11}"
                f"{'Q5':>11}{'Q6':>11}{'Q7':>11}{'Q8':>11}{'Q9':>11}\n"
            )
    
    def store_transitions(self, state, action, reward, state_):                   # STORING THE TRANSITIONS IN REPLAY MEMORY
        index = self.mem_cntr % self.mem_size                                               # KEEPNG THE INDEX IN RANGE (0, 10000)
        index_n = (index + 1) % self.mem_size

        self.state_memory[index] = state

        self.action_memory[index] = action
            
        self.reward_memory[index] = reward
        self.state_memory[index_n] = state_
        #self.terminal_memory[index] = 1 - terminal                                          # 1-TERMINAL DONE FOR EASY UPDATION OF Q ONLY FOR UNCOMPLETED EPISODES
        if self.mem_cntr % self.mem_size == 0 and self.mem_cntr > 0:

            print('\nSAVING TRAINING DATA\n')

            for ind, (S, A, R) in enumerate(zip(self.state_memory, self.action_memory, self.reward_memory)):
                self.training_data[f'ind_{ind}'] = [S, A, R]
            
            train_path = Path('/home/pi/Desktop/PROJECT/RL_AGENT/New_training/train_data')
            d = pd.DataFrame(self.training_data)
            d.to_csv(train_path / f"TRAIN_{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}.csv")
            print('\nSAVED TRAINING DATA\n')

        self.mem_cntr += 1

    def Q_LOG(self, q_vals):
        with open(self.q_log, "a") as f:
            for i in range(10):
                f.write(
                    f"{q_vals[0][i]:11f}"
                )
            f.write(
                    f"\n"
                )

    def choose_actions(self, observation):                                                  # EPSILON GREEDY EXPLORATION
        rand = np.random.random()

        if rand < self.epsilon:
            action = np.random.choice(list(range(10)))                                     # TAKING RANDOM ACTION FOR EXPLORING VALUES FOR VARIOUS ACTIONS
        else:
            observation = T.Tensor(observation)
            actions = self.Q_eval.forward(observation, model = 'online')
            #print('takinggg : ', actions)

            if self.mem_cntr % 100 == 0:
                self.Q_LOG(actions)

            action = T.argmax(actions, dim = 1).item()                              # TAKING GREEDY ACTION

        return action
    
    def save(self):
        # We use Path from pathlib so directly we can put / for the path.
        save_path = self.save_dir / f"Model_DVFS_{int(self.mem_cntr // self.log)}.chkpt"

        T.save(self.Q_eval.state_dict(), save_path)
        
        print(f"MODEL saved to {save_path} at step {self.mem_cntr}")

    def learn(self):
        if self.mem_cntr % self.log == 0:
            self.save()
        
        if self.mem_cntr and self.mem_cntr % self.tgt_update == 0:
            self.Q_eval.tgt.load_state_dict(self.Q_eval.lin.state_dict())

        if self.mem_cntr > self.train_start:
            if self.mem_cntr - 1 == self.train_start:
                print('Training start')

            max_mem = self.mem_cntr - 1 \
                if self.mem_cntr < self.mem_size else self.mem_size - 1                         # NOT TO EXCEED THE TOTAL REPLAY MEMORY SIZE

            batch = np.random.choice(max_mem, self.batch_size, replace=False)                              # SELECTING BATCHSIZE NO. OF SAMPLES FROM MAXMEM RANDOMLY
            
            #print('batch', batch)
            
            state_batch = T.tensor(self.state_memory[batch], dtype = T.float32)
            action_batch = T.tensor(self.action_memory[batch], dtype = T.long)           
            reward_batch = T.tensor(self.reward_memory[batch], dtype = T.float32)                              # COLLECTING REQUIRED PARAMETERS FROM TRANSITION MEMORY FOR A GIVEN BATCH
            next_state_batch = T.tensor(self.state_memory[batch + 1], dtype = T.float32)

            batch_index = np.arange(self.batch_size, dtype=np.int32)                        # BATCHINDEX VALUES FOR UPDATING THE Q_TARGET

            #print(self.Q_eval.forward(state_batch, 'online'))

            q_s_a = self.Q_eval.forward(state_batch, 'online')[batch_index, action_batch]                                        # Q_PRED FOR THE UPDATE
            
            with T.no_grad():
                best_action = T.argmax(self.Q_eval.forward(next_state_batch, 'online'), dim = 1)
                q_S_A = self.Q_eval.forward(next_state_batch, 'tgt')[batch_index, best_action]                                    # MAXIMUM VALUE FROM NEXT STATE
                q_target = reward_batch + self.gamma * q_S_A                                                      # TARGET IS Q_PRED ADD WITH THE REWARD AND MAX_VALUE FROM NEXT_STATE

            #print(f'Q_s_a {q_s_a} \n Action {action_batch} \n rew {reward_batch} \n Max_ns{q_S_A} \n Q_TGT {q_target}')

            self.epsilon = self.epsilon * self.eps_dec \
                if self.epsilon > self.eps_min else self.eps_min                            # UPDATING EPSILON AS EPISODES PROGRESSES TO TAKE MORE OF GREEDY ACTION

            loss = self.loss_fn(q_s_a, q_target)  
            self.optimizer.zero_grad()                                               # FOR NULLIFYING GRADIENTS OF PREVIOUS BATCHES                                      # Q_TARGET - Q_PRED IS THE TD-ERROR AND MSE LOSS FINDS THE SQUARE OF IT
            loss.backward()                                                                 # BACKPROPAGATING THE LOSS TO FING GRADIENTS

            self.optimizer.step()                                                    # UPDATING THE PARAMETERS I.E WEIGHTS USING GRADIENT DESCENT FOR THE Q_VALUE FUNCTION
            self.scheduler.step()
            #print(loss.item())

            return loss.item()
        else:
            return None