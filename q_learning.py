import sys
import numpy as np
from create_board import create_board, plot_board
import networkx as nx
from random import randint, choice

###################################################################################################################
# HYPER-PARAMETERS
###################################################################################################################
N = 10
PLOT = True
RED_PROBABILITY = 0.3
RED_PENALTY = -100
BLUE_PENALTY = -1
PRIZE = 50
CHANGES_THRESHOLD = 0.001
MAX_EPISODES = 1000
EPISODE_COUNT = 1

ALPHA = 0.5 # learning rate
GAMMA = 0.1 # how much we weight future rewards as opposed to immediate rewards
EPSILON = 1.0 # the balance between exploration and exploitation (we choose exploration with probability "EPSILON")
og_parameters = [ALPHA,GAMMA,EPSILON]

def update_hyper_parameter(): # auxiliary function, updates the hyper-parameters

    global ALPHA
    global GAMMA
    global EPSILON
    
    if(EPSILON>0.1): EPSILON -= 0.001
    if(GAMMA<0.4): GAMMA += 0.001
    if(ALPHA>0.15): ALPHA -= 0.001

def find_optimal_path(board, q_table, rewards_table): # auxiliary function, finds the optimal path from start to finish with the knowledge obtained
    
    start_state = np.where(board==-PRIZE)[0][0]*board.shape[0] + np.where(board==-PRIZE)[1][0]
    end_state = np.where(board==PRIZE)[0][0]*board.shape[0] + np.where(board==PRIZE)[1][0]
    state_update_rule = {0: -1, 1: -N, 2: +1, 3: +N}
    
    path = [start_state+1]
    current_state = start_state
    score = 0
    while(True):

        # get the action that maximizes our rewards
        aux = -sys.maxsize
        actions = []
        for idx,i in enumerate(q_table[current_state]):
            if(not (i is None)):
                actions.append([idx,i])

        # special case if 2 or more actions are equally likely to lead to good outcomes
        actions.sort(key = lambda x : x[1], reverse = False)
    
        if(len(actions)==1 or (len(actions)>=2 and actions[-1][1]!=actions[-2][1])): action = actions[-1][0]
        else: action = choice(actions[-2:])[0]
        
        # carry out the chosen action
        new_state = current_state + state_update_rule[action]

        path.append(new_state+1)

        # update the score
        score += rewards_table[current_state+1][new_state+1][str(current_state+1) + "-" + str(new_state+1)]["weight"]

        # stop criterium
        if(new_state==end_state): break

        # move to the chosen state
        current_state = new_state
    
    print("Final score: " + str(score))
    print("\nFinal path: " + " -> ".join(list(map(lambda x : str(x),path))))
    plot_board(board,RED_PENALTY,BLUE_PENALTY,PRIZE,path,og_parameters,[ALPHA,GAMMA,EPSILON],score,EPISODE_COUNT)

def print_q_table(board, q_table): # auxiliary function, prints the Q table in a fancy way

    start_state = np.where(board==-PRIZE)[0][0]*board.shape[0] + np.where(board==-PRIZE)[1][0]
    end_state = np.where(board==PRIZE)[0][0]*board.shape[0] + np.where(board==PRIZE)[1][0]
    moves = ["left","up","right","down"]

    for idx,i in enumerate(q_table):
        
        # the beginning of the line
        if(idx==start_state or idx==end_state): line = ">" + str(idx+1) + ": " 
        else: line = str(idx+1) + ": "
        
        # the rest of the line
        maximum = max([j for j in i if(not (j is None))])
        for jdx,j in enumerate(i):
            if(j is None): line += "---   "
            elif(j!=maximum): line += moves[jdx] + " (" + str(round(float(j),3)) + ")   "
            elif(j==maximum): line += moves[jdx].upper() + " (" + str(round(float(j),3)) + ")   "
        
        print(line[:-3])

def stop_criterium(old_q_table, new_q_table): # auxiliary function, decides when to stop training

    changes = 0.0

    # calculate the changes made to the Q-table
    for i in range(old_q_table.shape[0]):
        for j in range(old_q_table.shape[1]):
            if(not old_q_table[i][j] is None):
                changes += abs(old_q_table[i][j]-new_q_table[i][j])

    if(changes>=CHANGES_THRESHOLD): return(False)
    else: return(True)

def q_learning(board, rewards_table): # main function, applies the Q-Learning algorithm on the given board

    global EPISODE_COUNT

    ##########################################################################
    # INITIALIZE THE Q-TABLE
    ##########################################################################
    q_table = np.full((N*N,4),None) # the 4 actions are: LEFT, UP, RIGHT, DOWN
    cell_counter = 0
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            
            # left cell
            if(j!=0): q_table[cell_counter][0] = 0

            # top cell
            if(i!=0): q_table[cell_counter][1] = 0

            # right cell
            if(j!=(N-1)): q_table[cell_counter][2] = 0

            # bottom cell
            if(i!=(N-1)): q_table[cell_counter][3] = 0

            cell_counter += 1

    # get the start and finish states
    start_state = np.where(board==-PRIZE)[0][0]*board.shape[0] + np.where(board==-PRIZE)[1][0]
    end_state = np.where(board==PRIZE)[0][0]*board.shape[0] + np.where(board==PRIZE)[1][0]

    state_update_rule = {0: -1, 1: -N, 2: +1, 3: +N}
    
    #############################################################################################################################################
    # APPLY THE Q-LEARNING ALGORITHM
    #############################################################################################################################################
    stop = False
    while(EPISODE_COUNT<MAX_EPISODES):
        current_state = start_state
        history = []
        #q_table_snapshot = q_table.copy()

        # get to the end of the board
        while(True):
            
            #print("EP " + str(EPISODE_COUNT) + ": " + str(current_state+1) + " (GOAL: " + str(end_state+1) + ")")
            
            # select every possible action
            possible_actions = [j for j in range(len(q_table[current_state])) if(not (q_table[current_state][j] is None))]
            
            # ---------------------------------------------------------------------------------------------------------------------------------
            # select an action using ε-greedy policy (we explore if the random number is less than or equal to "EPSILON", otherwise we exploit)
            # ---------------------------------------------------------------------------------------------------------------------------------
            random_number = randint(1,100)
            
            if(random_number<=(EPSILON*100)): # we explore
                action = choice(possible_actions)
            
            else: # we exploit

                # get the action that maximizes our rewards
                aux = -sys.maxsize
                action = 0
                for i in possible_actions:
                    # if better, save this action
                    if(q_table[current_state][i]>aux): 
                        if(len(history)!=0 and (current_state + state_update_rule[i])==history[-1]): continue
                        action = i
                        aux = q_table[current_state][i]

            # ---------------------------------------------------------------------------------------------------------------------
            # move to the next state, compute immediate and future rewards
            # ---------------------------------------------------------------------------------------------------------------------
            # carry out the chosen action
            new_state = current_state + state_update_rule[action]

            # we have reached the end state, we can stop
            if(new_state==end_state): break
            
            history.append(new_state)
            
            # compute the immediate reward
            immediate_reward = rewards_table[current_state+1][new_state+1][str(current_state+1) + "-" + str(new_state+1)]["weight"]

            # compute the approximation of future rewards
            future_reward = max([i for i in q_table[new_state] if(not (i is None))])

            # -----------------------------------------------------------------------------------------------------------------------------------
            # update the Q table
            # -----------------------------------------------------------------------------------------------------------------------------------
            q_table[current_state][action] += ALPHA * (immediate_reward + (GAMMA * future_reward) - q_table[current_state][action])
            #q_table[current_state][action] = ((1-ALPHA) * q_table[current_state][action]) + ALPHA * (immediate_reward + (GAMMA * future_reward))

            # move to the chosen state
            current_state = new_state

        # decay our hyper-parameters
        update_hyper_parameter()

        EPISODE_COUNT += 1

        # decide if we stop training
        #stop = stop_criterium(q_table_snapshot,q_table)
    
    return(q_table)

if __name__ == "__main__":

    # create a NxN board
    #create_board(10,RED_PROBABILITY,RED_PENALTY,BLUE_PENALTY,PRIZE)

    board = np.load("board.npy")
    rewards_table = nx.read_gpickle("rewards_table.gpickle")
    
    # build the Q table
    q_table = q_learning(board,rewards_table)

    # print the Q table
    print_q_table(board,q_table)

    print("\nTotal episodes: " + str(EPISODE_COUNT))

    # find the optimal path for the given board
    find_optimal_path(board,q_table,rewards_table)

    print("Final parameters: α = " + str(round(ALPHA,2)) + " γ = " + str(round(GAMMA,2)) + " ε = " + str(round(EPSILON,2)))