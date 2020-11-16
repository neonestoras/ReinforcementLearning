
import numpy as np
import matplotlib.pyplot as plt
import random 
from multiprocessing import Pool
import math
import pickle


cores = 24
print('start script')

CID = '01350804'
x = float(CID[len(CID)-3])
y = float(CID[len(CID)-2])
z = (float(CID[len(CID)-1])+1) % 3 + 1

p_cid = 0.25 + 0.5 * (x+1)/10
gamma_cid = 0.2 + 0.5 * y/10



class GridWorld(object):

    def __init__(self, p=p_cid):
        
        ### Attributes defining the Gridworld #######
        self.p = p
        # Shape of the gridworld
        self.shape = (6,6)
        
        # Locations of the obstacles
        self.obstacle_locs = [(1,1),(3,1),(4,1),(4,2),(2,3),(2,5),(4,4)]
        
        # Locations for the absorbing states
        self.absorbing_locs = [(1,3),(4,3)]
        
        # Rewards for each of the absorbing states 
        self.special_rewards = [10, -100] # Corresponds to each of the absorbing_locs
        
        # Reward for all the other states
        self.default_reward = -1
        
        # Starting location
        self.starting_loc = self.get_random_loc()
        
        # Action names
        self.action_names = ['N','E','S','W'] # Action 0 is 'N', 1 is 'E' and so on
        
        # Number of actions
        self.action_size = len(self.action_names)
        
        # Randomizing action results: [1 0 0 0] to no Noise in the action results.
        self.action_randomizing_array = np.array([p, (1-p)/3, (1-p)/3, (1-p)/3])
        self.action_randomizing_transition_matrix = np.array([np.roll(self.action_randomizing_array, i) for i in range(4)])
        
        ############################################
    

        #### Internal State  ####
        
        # Get attributes defining the world
        state_size, T, R, absorbing, locs = self.build_grid_world()
        
        # Number of valid states in the gridworld (there are 22 of them - 5x5 grid minus obstacles)
        self.state_size = state_size
        
        # Transition operator (3D tensor)
        self.T = T # T[st+1, st, a] gives the probability that action a will 
                   # transition state st to state st+1
        
        # Reward function (3D tensor)
        self.R = R # R[st+1, st, a ] gives the reward for transitioning to state
                   # st+1 from state st with action a
        
        # Absorbing states
        self.absorbing = absorbing
        
        # The locations of the valid states 
        self.locs = locs # State 0 is at the location self.locs[0] and so on
        
        # Number of the starting state
        self.starting_state = self.loc_to_state(self.starting_loc, locs)
        
        # Locating the initial state
        self.initial = np.zeros((1,len(locs)))
        self.initial[0,self.starting_state] = 1
        
        # Placing the walls on a bitmap
        self.walls = np.zeros(self.shape)
        for ob in self.obstacle_locs:
            self.walls[ob]=1
            
        # Placing the absorbers on a grid for illustration
        self.absorbers = np.zeros(self.shape)
        for ab in self.absorbing_locs:
            self.absorbers[ab] = -1
        
        # Placing the rewarders on a grid for illustration
        self.rewarders = np.zeros(self.shape)
        for i, rew in enumerate(self.absorbing_locs):
            self.rewarders[rew] = self.special_rewards[i]
        
        #Illustrating the grid world
        # self.paint_maps()

        ################################
    
    

    ####### Getters ###########
    
    def get_transition_matrix(self):
        return self.T
    
    def get_reward_matrix(self):
        return self.R

    def get_p(self):
        return self.p
    
    ###################
    ##### METHODS


    ##########################
    
   

    ##########################
    
    
    ########### Internal Helper Functions #####################

    ## You do not need to understand these functions in detail in order to complete the lab ##

        

    def build_grid_world(self):
        # Get the locations of all the valid states, the neighbours of each state (by state number),
        # and the absorbing states (array of 0's with ones in the absorbing states)
        locations, neighbours, absorbing = self.get_topology()
        
        # Get the number of states
        S = len(locations)
        
        # Initialise the transition matrix
        T = np.zeros((S,S,4))
        
        for action in range(4):
            for effect in range(4):
                # Randomize the outcome of taking an action
                outcome = (action+effect+1) % 4
                if outcome == 0:
                    outcome = 3
                else:
                    outcome -= 1
                # Fill the transition matrix:
                # A good way to understand the code, is to first ask ourselves what the structure 
                # of the transition probability ‘matrix’ should be, given that we have state, successor state and action. 
                # Thus, a simple row x column matrix of successor state and will not suffice, as we also have to condition 
                #  on the action. So we can therefore choose to implement this to  have a structure that is 3 dimensional
                # (technically a tensor, hence the variable name T). I would not worry too much about what a tensor is, 
                # it is simply an array that takes 3 arguments to get a value, just like conventional matrix is an array that
                # takes 2 arguments (row and column), to get a value. To touch all the elements in this structure we
                # need therefore to loop over states and actions.
                                # [0.5, 0.5/3, 0.5/3. 0.5/3]
                prob = self.action_randomizing_array[effect]
                for prior_state in range(S):
                    post_state = neighbours[prior_state, outcome]
                    post_state = int(post_state)
                    T[post_state,prior_state,action] = T[post_state,prior_state,action]+prob
                    
    
        # Build the reward matrix
        R = self.default_reward*np.ones((S,S,4))
        for i, sr in enumerate(self.special_rewards):
            post_state = self.loc_to_state(self.absorbing_locs[i],locations)
            R[post_state,:,:]= sr
        
        return S, T,R,absorbing,locations
    

    def get_topology(self):
        height = self.shape[0]
        width = self.shape[1]
        
        index = 1 
        locs = []
        neighbour_locs = []
        
        for i in range(height):
            for j in range(width):
                # Get the locaiton of each state
                loc = (i,j)
                
                #And append it to the valid state locations if it is a valid state (ie not absorbing)
                if(self.is_location(loc)):
                    locs.append(loc)
                    
                    # Get an array with the neighbours of each state, in terms of locations
                    local_neighbours = [self.get_neighbour(loc,direction) for direction in ['nr','ea','so', 'we']]
                    neighbour_locs.append(local_neighbours)
                
        # translate neighbour lists from locations to states
        num_states = len(locs)
        state_neighbours = np.zeros((num_states,4))
        
        for state in range(num_states):
            for direction in range(4):
                # Find neighbour location
                nloc = neighbour_locs[state][direction]
                
                # Turn location into a state number
                nstate = self.loc_to_state(nloc,locs)
      
                # Insert into neighbour matrix
                state_neighbours[state,direction] = nstate
                
    
        # Translate absorbing locations into absorbing state indices
        absorbing = np.zeros((1,num_states))
        for a in self.absorbing_locs:
            absorbing_state = self.loc_to_state(a,locs)
            absorbing[0,absorbing_state] =1
        
        return locs, state_neighbours, absorbing 


    def loc_to_state(self,loc,locs):
        #takes list of locations and gives index corresponding to input loc
        return locs.index(tuple(loc))


    def is_location(self, loc):
        # It is a valid location if it is in grid and not obstacle
        if(loc[0]<0 or loc[1]<0 or loc[0]>self.shape[0]-1 or loc[1]>self.shape[1]-1):
            return False
        elif(loc in self.obstacle_locs):
            return False
        else:
             return True
            
    def get_neighbour(self,loc,direction):
        #Find the valid neighbours (ie that are in the grif and not obstacle)
        i = loc[0]
        j = loc[1]
        
        nr = (i-1,j)
        ea = (i,j+1)
        so = (i+1,j)
        we = (i,j-1)
        
        # If the neighbour is a valid location, accept it, otherwise, stay put
        if(direction == 'nr' and self.is_location(nr)):
            return nr
        elif(direction == 'ea' and self.is_location(ea)):
            return ea
        elif(direction == 'so' and self.is_location(so)):
            return so
        elif(direction == 'we' and self.is_location(we)):
            return we
        else:
            #default is to return to the same location
            return loc
        
###########################################         
        
    def get_random_loc(self):
        return random.choice(list(filter(
            lambda x: x not in self.obstacle_locs and x not in self.absorbing_locs, 
            [(i,j) for i in range(6) for j in range(6)])))
    

    def generate_trace(self, policy):
        locations, neighbours, absorbing = self.get_topology()
        R = self.get_reward_matrix()

        # get random starting location
        self.starting_loc = self.get_random_loc()

        state_index = self.loc_to_state(self.starting_loc, self.locs)
        trace = []

        while True: 
            if state_index in absorbing[0].nonzero()[0]:
                return trace
            
            action = random.choices([0,1,2,3], policy[state_index])
            effect_probability = self.action_randomizing_transition_matrix[action][0]
            effect = random.choices([0,1,2,3], effect_probability)[0]
            
            destination_index = int(neighbours[state_index][effect])
            trace.append((state_index, ["N", "E", "S", "W"][effect], R[destination_index][state_index][effect]))
            state_index = destination_index

    def apply_e_greedy_Q(self, Q, epsilon):
        _, neighbours, _ = self.get_topology()
        new_policy = np.full((self.state_size, self.action_size),epsilon/self.action_size)
        immediate_reward=np.full(self.state_size, self.default_reward)
        
        for i in range(len(self.absorbing_locs)):
            immediate_reward[self.loc_to_state(self.absorbing_locs[i],self.locs)]=self.special_rewards[i]
            
        for state_idx in range(self.state_size): 
            immediate_rewards = [immediate_reward[int(neighbour)] for neighbour in neighbours[state_idx]]
            value_of_neighbours = list([Q[state_idx][j]+immediate_rewards[j] for j in range(len(immediate_rewards))])
            new_policy[state_idx, np.argmax(value_of_neighbours)] = 1 - epsilon + epsilon/self.action_size
        return new_policy
    
    def calculate_trace_reward(self,trace):
        np_trace = np.array(trace)
        rewards = np.asarray(np_trace[:,2], dtype=np.float32)
        total = rewards.sum()
        return total
        
    def Q2V(self, Q, epsilon=0):
        Vs=[]
        for s in range(self.state_size):
            V = (1-epsilon)*max(Q[s]) +(epsilon*np.sum(Q[s]))/self.action_size
            Vs.append(V)
        return np.array(Vs)
          
        
    ##########################################
    #################################
    ############################################
    ##############################################
 

    def SARSA(self, iterations, alpha, epsilon=0, update_epsilon=True, gamma=gamma_cid):
            R = self.get_reward_matrix()
            locations, neighbours, absorbing = self.get_topology()
            initial_epsilon = epsilon
            
            Q = np.random.rand(self.state_size, self.action_size)
            for loc in self.absorbing_locs:
                Q[self.loc_to_state(loc,self.locs)]=0

            total_rewards = []  
            Vs = []
            for i in range(iterations):
                self.init_state = self.get_random_loc()
                state_index = self.loc_to_state(self.starting_loc, self.locs)
                
                total_reward = 0
                policy = self.apply_e_greedy_Q(Q, epsilon=epsilon)
                action_index = random.choices([0,1,2,3], policy[state_index])[0]
                if update_epsilon:     
                    epsilon = initial_epsilon*(math.exp(-i/400))
                while True:
                    if state_index in absorbing[0].nonzero()[0]:
                        total_rewards.append(total_reward)
                        break
                        
                    effect_probability = self.action_randomizing_transition_matrix[action_index]
                    effect = random.choices([0,1,2,3], effect_probability)[0]
                
                    destination_index = int(neighbours[state_index][effect])
                    
                    reward = R[destination_index][state_index][effect]
                    total_reward = total_reward + reward
                    
                    policy = self.apply_e_greedy_Q(Q, epsilon=epsilon)
                    action_index_prime = random.choices([0,1,2,3], policy[destination_index])[0]
                    
                    Q[state_index][action_index] = Q[state_index][action_index] + alpha * (reward + gamma * Q[destination_index][action_index_prime] - Q[state_index][action_index])
                    V = self.Q2V(Q, epsilon)
                    state_index = destination_index
                    action_index= action_index_prime
                
                Vs.append(V)
                
            return Vs, Q, policy, np.asarray(total_rewards, dtype=np.float32)


print('start sarsa')
grid = GridWorld()
repetitions = 300
iterations = 1000
alphas = [0.1, 0.2]
for graph in range(len(alphas)):
    Values_noUpdate=[]
    Values_Update1=[]
    Values_Update2=[]
    with Pool(cores) as pool:
        Values_noUpdate = np.array(pool.starmap(grid.SARSA, [(iterations,alphas[graph],0.05,False) for rep in range(repetitions)] ))

    print(1)#

    with Pool(cores) as pool:
        Values_Update1 = np.array(pool.starmap(grid.SARSA, [(iterations,alphas[graph],0.05,True) for rep in range(repetitions)] ))#

    print(2)#

    with Pool(cores) as pool:
        Values_Update2 = np.array(pool.starmap(grid.SARSA, [(iterations,alphas[graph],0.1,True) for rep in range(repetitions)] ))#

    print(3)


    file = 'SARSA_{}'.format(graph)
    with open(file, 'wb') as f:
        pickle.dump([Values_noUpdate,Values_Update1,Values_Update2], f)
        f.close()
        print('saved')

print('ALL DONE SARSA')