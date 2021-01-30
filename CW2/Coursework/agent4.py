############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import numpy as np
from collections import deque
import torch
from math import sqrt

class Agent:

    # Function to initialise the agent agent.batch_size,agent.buffer_size,agent.target_freq,agent.priority_factor
    def __init__(self):
        # Set the episode length
        self.episode_length = 210
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        self.step = 0
        self.LEARNING = True
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        self.action_mapping = np.array([[1,0],[sqrt(2)/2,sqrt(2)/2],[0,1],[-sqrt(2)/2,sqrt(2)/2],[-1,0],[-sqrt(2)/2,-sqrt(2)/2],[0,-1],[sqrt(2)/2,-sqrt(2)/2]], dtype=np.float32) * 0.018
        self.dqn = DQN()
        self.episode = 1
        self.buffer = ReplayBuffer(30000)
        self.epsilon_bias = 0.0
        

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            self.episode += 1
            self.step = 0
            self.epsilon_bias = 0.0
            return True
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Here, the action is random, but you can change this
        self.step += 1
        action =self.get_greedy_action_idx(state, self.epsilon_update() )
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action
        return self.action_mapping[action] 

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
        reward = 1 - distance_to_goal**0.7
        if (distance_to_goal < 0.03):
            reward += 1 # additional reward
            #Check for early stopping
            if (self.epsilon == 0) and self.step<101 :
                self.LEARNING = False
        # Create a transition
        transition = (self.state, self.action, reward, next_state)

        # If the agent hits a wall
        if np.array_equal(self.state, next_state):
            if self.epsilon < 0.55:
                self.epsilon_bias += 0.006     #increase e bias
            reward *= 0.65                      #discount the earned reward

        #Add transition in buffer
        self.buffer.append(transition)
        #After buffer is filled begin training
        b_length = len(self.buffer.buffer)
        if b_length > 210 and self.LEARNING:
            self.buffer.set_indices(210)
            minibatch = self.buffer.sample()
            predicted_q = self.dqn.q_network.forward(torch.tensor(minibatch[0]))
            d = self.dqn.train_q_network(minibatch, predicted_q)
            self.buffer.update_w(d)
            if self.num_steps_taken % 95 == 0:
                self.dqn.update_target_network()


    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state, e=0):
        # Here, the greedy action is fixed, but you should change it so that it returns the action with the highest Q-value
        i= self.get_greedy_action_idx(state,e)
        return self.action_mapping[i]

    def get_greedy_action_idx(self, state, e=0):
        # Here, the greedy action is fixed, but you should change it so that it returns the action with the highest Q-value
        Q = self.dqn.q_network.forward(torch.tensor(state))
        policy = np.full(8, e/8)
        policy[torch.argmax(Q)] += 1-e
        return np.random.choice([0,1,2,3,4,5,6,7], p=policy)

    def epsilon_update(self):
        if self.episode % 9 != 0 and self.LEARNING:
            e = 0.55*np.exp(-self.episode/50) + self.epsilon_bias**3
        else: 
            e=0
        self.epsilon = e
        return e


class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.w = deque(maxlen=capacity)

    def append(self, experience):
        self.buffer.append(experience)
        if len(self.w) > 0:
            self.w.append(max(self.w))
        elif len(self.w) == 0:
            self.w.append(0.01)
            
    def update_w(self, d):
        for i, index in enumerate(self.indices):
            self.w[index] = np.absolute(d[i].detach().numpy()) + 0.01

    def set_indices(self, batch_size):
        w = np.array(self.w) ** 0.65
        p = w / sum(w)
        self.indices = np.random.choice(len(self.buffer), batch_size, replace=True, p=p)

    def sample(self):
        indices = self.indices
        sample = [self.buffer[index] for index in indices]
        state, action, reward, next_state = zip(*sample)
        return state, action, np.array(reward, dtype=np.float32), next_state

# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=160)
        self.dropout = torch.nn.Dropout(0.375) #0.375 => 160 * 0.375 = 100
        self.layer_3 = torch.nn.Linear(in_features=160, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_2_dropout = self.dropout(layer_2_output)
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_dropout))
        output = self.output_layer(layer_3_output)
        return output

# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=8)
        #TARGET NETWORK
        self.target_network = Network(input_dimension=2, output_dimension=8)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.0006)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        #'Target Updated'

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transition, predicted_q):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss, d = self._calculate_loss(transition, predicted_q)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return d

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transition, predicted_q):
        action = torch.tensor(transition[1], dtype=torch.int64)
        reward = torch.tensor(transition[2]) 
        next_state = torch.tensor(transition[3])
        selected_predicted_q = predicted_q.gather(dim=1, index=action.unsqueeze(-1)).squeeze(-1) 
        
        max_action = self.q_network.forward(next_state).argmax(dim=1).detach()
        max_future_q_prediction =  self.target_network.forward(next_state).gather(dim=1, index=max_action.unsqueeze(-1)).squeeze(-1).detach()
            
        d = reward + max_future_q_prediction - selected_predicted_q
        
        return torch.nn.MSELoss()(selected_predicted_q, reward + 0.9 * max_future_q_prediction), d 
