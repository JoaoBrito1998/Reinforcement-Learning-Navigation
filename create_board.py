import sys
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from matplotlib import colors as c
import networkx as nx

def discard_board(rewards_table, reds, start_state, end_state): # auxiliary function, checks if we can get from start to finish without using any red cell

    graph = rewards_table.copy()

    for i in reds: graph.remove_node(i)

    try:
        nx.astar_path(graph,start_state,end_state,heuristic=None)
    except nx.NetworkXNoPath as e:
        return(True)

    return(False)

def plot_board(board, red_penalty = -100, blue_penalty = -1, prize = 50, optimal_path = [], og_parameters = [], final_parameters = [], score = 0, episode_count = 0): # auxiliary function, plots the given board with fancy colors
    
    board_copy = board.copy()
    board_copy[board_copy==(-prize)] = 256
    board_copy[board_copy==prize] = 256
    board_copy[board_copy==red_penalty] = 100
    board_copy[board_copy==blue_penalty] = 1

    plt.gca().invert_yaxis()
    title = "Q-Learning " + "(" + str(board.shape[0]) + "x" + str(board.shape[1]) + ") - "
    
    # alpha part of the title
    if(og_parameters[0]==final_parameters[0]): title += "α = " + str(og_parameters[0]) + " | "
    else: title += "α = " +str(og_parameters[0]) + " ... " + str(round(final_parameters[0],2)) + " | "

    # gamma part of the title
    if(og_parameters[1]==final_parameters[1]): title += "γ = " +str(og_parameters[1]) + " | "
    else: title += "γ = " +str(og_parameters[1]) + " ... " + str(round(final_parameters[1],2)) + " | "

    # epsilon part of the title
    if(og_parameters[2]==final_parameters[2]): title += "ε = " +str(og_parameters[2])
    else: title += "ε = " +str(og_parameters[2]) + " ... " + str(round(final_parameters[2],2))

    plt.title(title)
    plt.gcf().text(0.5, 0.052, "Score: " + str(score) + "/" + str(prize) +  " | " + str(episode_count) + " Ep.", fontsize=14,ha='center', va='center')
    
    # label each cell 
    cell_counter = 1
    for y in range(board_copy.shape[0]):
        for x in range(board_copy.shape[1]):
            if(cell_counter not in optimal_path):
                plt.text(x + 0.5, y + 0.5, "%d" % cell_counter, horizontalalignment="center", verticalalignment="center")
            elif(cell_counter in optimal_path):
                plt.text(x + 0.5, y + 0.5, "%d" % cell_counter, horizontalalignment="center", verticalalignment="center", color = "white")
            cell_counter += 1
    
    plt.pcolor(board_copy, edgecolors="k", linewidth=1.0, cmap=c.ListedColormap(["steelblue","firebrick","gold"]))
    plt.axis("off")

    plt.savefig("board.png")

def create_board(N = 10, red_probability = 0.3, red_penalty = -100, blue_penalty = -1, prize = 50): # main function, creates a NxN board with blue (penalty = "blue_penalty"), red (penalty = "red_penalty") and yellow (start and finish) cells 

    while(True):
        try:
            board = np.ones((N,N),dtype=int)*blue_penalty
            rewards_table = nx.MultiDiGraph()
            auxiliary_graph = nx.Graph()
            reds = []

            # size of the matrix version of R
            #print(sys.getsizeof(board))

            '''board = np.zeros((10,10),int)
            board[0,:] = [-1,-1,-1,-1,-5,-50,-1,-1,-1,-1]
            board[1,:] = [-1,-1,-5,-5,-1,-1,-1,-1,-1,-1]
            board[2,:] = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
            board[3,:] = [-5,-1,-5,-5,-1,-5,-1,-5,-1,-1]
            board[4,:] = [-5,-1,-1,-1,-5,-1,-1,-5,-5,-1]
            board[5,:] = [-5,-1,-1,-5,-1,-5,-1,-1,-5,-5]
            board[6,:] = [-5,-1,-1,-1,-1,-1,-5,-1,-5,-1]
            board[7,:] = [-1,-5,-5,-5,-5,-1,-5,-1,-1,-1]
            board[8,:] = [-1,-1,-1,-1,-1,-1,-5,-1,-1,-1]
            board[9,:] = [-1,-5,50,-5,-1,-5,-1,-1,-1,-1]'''
            
            ##################################################
            # CREATE THE BOARD
            ##################################################
            # randomize the distribution of red cells
            for i in range(board.shape[0]):
                for j in range(board.shape[1]):
                    if(randint(1,100)<=(100*red_probability)):
                        board[i][j] = red_penalty
                        reds.append(((i*board.shape[0])+j)+1)

            # randomise the starting position
            board[0,randint(0,N-1)] = -prize

            # randomise the goal position
            board[N-1,randint(0,N-1)] = prize

            #######################################################################################################
            # CREATE THE REWARDS TABLE/GRAPH
            #######################################################################################################
            cell_counter = 0
            for i in range(board.shape[0]):
                for j in range(board.shape[1]):
                    
                    # top cell
                    if(i!=0):
                        a = (cell_counter)
                        b = (cell_counter)-N

                        # first edge
                        rewards_table.add_edge(a+1,b+1,key=(str(a+1) + "-" + str(b+1)),weight=board[int(b/N)][b%N])

                        # second edge
                        rewards_table.add_edge(b+1,a+1,key=(str(b+1) + "-" + str(a+1)),weight=board[int(a/N)][a%N])
                        
                        # update the auxiliary edge
                        auxiliary_graph.add_edge(a+1,b+1)

                    # bottom cell
                    if(i!=(N-1)): 
                        a = (cell_counter)
                        b = (cell_counter)+N

                        # first edge
                        rewards_table.add_edge(a+1,b+1,key=(str(a+1) + "-" + str(b+1)),weight=board[int(b/N)][b%N])

                        # second edge
                        rewards_table.add_edge(b+1,a+1,key=(str(b+1) + "-" + str(a+1)),weight=board[int(a/N)][a%N])
                        
                        # update the auxiliary edge
                        auxiliary_graph.add_edge(a+1,b+1)

                    # left cell
                    if(j!=0): 
                        a = (cell_counter)
                        b = (cell_counter)-1

                        # first edge
                        rewards_table.add_edge(a+1,b+1,key=(str(a+1) + "-" + str(b+1)),weight=board[int(b/N)][b%N])

                        # second edge
                        rewards_table.add_edge(b+1,a+1,key=(str(b+1) + "-" + str(a+1)),weight=board[int(a/N)][a%N])
                        
                        # update the auxiliary edge
                        auxiliary_graph.add_edge(a+1,b+1)

                    # right cell
                    if(j!=(N-1)): 
                        a = (cell_counter)
                        b = (cell_counter)+1

                        # first edge
                        rewards_table.add_edge(a+1,b+1,key=(str(a+1) + "-" + str(b+1)),weight=board[int(b/N)][b%N])

                        # second edge
                        rewards_table.add_edge(b+1,a+1,key=(str(b+1) + "-" + str(a+1)),weight=board[int(a/N)][a%N])
                        
                        # update the auxiliary edge
                        auxiliary_graph.add_edge(a+1,b+1)

                    cell_counter += 1

            start_state = np.where(board==-prize)[0][0]*board.shape[0] + np.where(board==-prize)[1][0] + 1
            end_state = np.where(board==prize)[0][0]*board.shape[0] + np.where(board==prize)[1][0] + 1

            # decide if we keep this board
            if(discard_board(auxiliary_graph,reds,start_state,end_state)): continue
            break
        
        except Exception as e: pass

    # dump the rewards table/graph to a file
    nx.write_gpickle(rewards_table, "rewards_table.gpickle")

    # size of the graph version of R
    #print(sys.getsizeof(rewards_table.edges) + sys.getsizeof(rewards_table.nodes) + sys.getsizeof(rewards_table))
    
    # save the board to a file
    np.save("board.npy",board)

    print(board)

if __name__ == "__main__":

    create_board(10)