# Mnih, Volodymyr, et al. "Asynchronous methods for deep reinforcement learning." International conference on machine learning. PMLR, 2016.
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

import numpy as np
import gym
from collections import deque

class Memory():
    def __init__(self):
        super(Memory, self).__init__()
        self.memory = []
        
    # Add the memory
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Sample the memory
    def sample(self):
        batch = self.memory
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    # Reset the memory
    def reset(self):
        self.memory = []

class A3CNet(nn.Module):
    def __init__(self, input, output):
        super(A3CNet, self).__init__()
        self.input = nn.Linear(input, 16)
        self.fc = nn.Linear(16, 16)
	
        self.value = nn.Linear(16, 1)
        self.policy = nn.Linear(16, output)
    
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.fc(x))
        
        value = self.value(x)
        policy = F.softmax(self.policy(x))
        return value, policy
    
class A3C():
    def __init__(self, env, actor_num=4, actor_ratio=0.5, entropy_beta=0.01, gamma=0.95, learning_rate=1e-3, t_max=20):
        super(A3C, self).__init__()
        self.env = env
        self.actor_num = actor_num
        self.state_num = self.env.observation_space.shape[0]
        self.action_num = self.env.action_space.n
        
        # Global model
        self.a3c_net = A3CNet(self.state_num, self.action_num).to(self.device)
        self.optimizer = optim.Adam(self.a3c_net.parameters(), lr=learning_rate)
        
        # Learning setting
        self.gamma = gamma
        self.t_max = t_max
        
        # Loss setting
        self.actor_ratio = actor_ratio
        self.entropy_beta = entropy_beta
        
    def total_run(self):
        Actors = [Actor(self.env, self.actor_ratio, self.entropy_beta, self.gamma, self.a3c_net, self.optimizer, self.t_max)
                  for _ in range(self.actor_num)]

        # For GPU processing
        mp.set_start_method('spawn')
        
        # Multi processing
        processes = []
        for actor in Actors:
            process = mp.Process(target=actor.run)
            process.start()
            processes.append(process)
        for p in processes:
            p.join()
            
        
class Actor():
    def __init__(self, env, actor_ratio, entropy_beta, gamma, a3c_net, optimizer, t_max):
        super(Actor, self).__init__()
        self.env = env
        self.state_num = self.env.observation_space.shape[0]
        self.action_num = self.env.action_space.n
               
        # Torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Global model
        self.a3c_net = a3c_net
        self.optimizer = optimizer
        
        # Local (actor) model
        self.actor_net = A3CNet(self.state_num, self.action_num).to(self.device)
        self.actor_net.load_state_dict(self.a3c_net.state_dict())
        
        # Learning setting
        self.gamma = gamma
        self.t_max = t_max
        
        # Loss setting
        self.actor_ratio = actor_ratio
        self.entropy_beta = entropy_beta
        
        # Memory setting
        self.memory = Memory()

    # Get the action
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        _, policy = self.actor_net(state)
        policy = policy.cpu().detach().numpy()
        action = np.random.choice(self.action_num, 1, p=policy[0])
        return action[0]

    # Transfer the model's gradients to the target model
    def shared_grads(self, model, target_model):
        for param, target_param in zip(model.parameters(), target_model.parameters()):
            target_param._grad = param.grad
        
    # Learn the policy
    # j: Policy objective function
    def learn(self):
        if not self.memory.memory:
            return

        # Off-policy
        # states = torch.FloatTensor([m[0] for m in self.memory]).to(self.device)
        # actions = torch.LongTensor([m[1] for m in self.memory]).to(self.device)
        # rewards = torch.FloatTensor([m[2] for m in self.memory]).to(self.device).view(-1,1)
        # next_states = torch.FloatTensor([m[3] for m in self.memory]).to(self.device)
        # dones = torch.FloatTensor([0 if m[4] else 1 for m in self.memory]).to(self.device).view(-1,1)

        # values, policies = self.actor_net(states)
        # next_values, _ = self.actor_net(next_states)
        # target = rewards + self.gamma * next_values * dones

        # advantage = target - values
        # log_prob = torch.log(policies)
        # j = advantage * log_prob[range(actions.size(dim=0)), actions].view(-1,1)
        
        # On-policy
        # Get the episode memory
        states, actions, rewards, next_states, dones = self.memory.sample()
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_state = torch.FloatTensor(next_states[-1]).to(self.device)
        done = dones[-1]
        
        # Actor network
        values, policies = self.actor_net(states)
        next_value, _ = self.actor_net(next_state)
        
        # Calculate target values
        R = [0] * (actions.size(dim=0) + 1)
        R[-1] = next_value if not done else 0
        for i in reversed(range(len(R)-1)):
            R[i] = rewards[i] + self.gamma * R[i+1]
        R = torch.FloatTensor(R[:-1]).to(self.device).view(-1,1)

        # Calculate advantages and objective function
        advantage = R - values
        log_prob = torch.log(policies)
        j = advantage * log_prob[range(actions.size(dim=0)), actions].view(-1,1)

        # Calculate actor and critic losses
        actor_loss = -j.mean()
        critic_loss = advantage.pow(2).mean()
        entropy_loss = -(policies * log_prob).mean()
        loss = self.actor_ratio * actor_loss + critic_loss + self.entropy_beta * entropy_loss
        
        # Optimize the global network
        self.optimizer.zero_grad()
        self.shared_grads(self.actor_net, self.a3c_net)
        loss.backward()
        self.optimizer.step()
        
        # load the network and reset the memory
        self.actor_net.load_state_dict(self.a3c_net.state_dict())
        self.memory.reset()
        
    def run(self):
        ep_rewards = deque(maxlen=100)
        total_episode = 10000
    
        for i in range(total_episode):
            state = self.env.reset()
            rewards = []

            while True:
                action = self.get_action(state)
                next_state, reward , done, _ = self.env.step(action)
    
                self.memory.add(state, action, reward, next_state, done)
                if len(self.memory.memory) == self.t_max:
                    self.learn()
                    
                rewards.append(reward)
                
                if done:
                    self.learn()
                    ep_rewards.append(sum(rewards))
                    
                    if i % 100 == 0:
                        print("episode: {}\treward: {}".format(i, round(np.mean(ep_rewards), 3)))
                    break

                state = next_state
        

def main():
    env = gym.make("CartPole-v0")
    agent = A3C(env, actor_num=4, actor_ratio=0.2, gamma=0.99, learning_rate=1e-3)
    
    agent.total_run()
    

if __name__ == '__main__':
    main()
