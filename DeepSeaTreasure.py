# The Deep Sea Treasure Environment : In this game, the
# environment consists of a grid of 10 rows and 11 columns
# as shown in Figure 1. The diving vessel is controlled by the
# agent to search for undersea treasure. Treasure with varying
# values is located at multiple locations. One objective is to
# minimize the time taken to reach the treasure and another
# objective is to maximize the value of the treasure retrieved.
# As this is the episodic task, it starts with the vessel in the
# top row of the first column, and terminates when a treasure
# location is reached or after 1000 actions have been executed.
# Agent can move either left, right, up or down per square in
# each time step.
# The reward received by agent is 2-element vector for each
# turn. The first element in the vector is the penalty for time
# consumed, which equals -1 on all turns while second element is 
# the value of the treasure retrieved which varies according to 
# the treasure location and it is 0 for other locations
# without any treasure.
# An interesting property of this environment is that only
# 2 of the 10 Pareto-optimal policies (one for a minimumlength path to each treasure) are on the convex hull of the
# Pareto front; this means that only those two can be the
# globally-optimal policies for linear scalarization functions.
# This poses problems for learning algorithms that try to use
# different linear scalarizations to learn new Pareto-optimal
# policies; though they may discover non-convex-hull policies incidentally, they will only converge on the convex-hull
# ones.

# The solution gives pareto front with rewards related to time and treasure respectively.

import time
import copy
import random
import numpy as np
from tabulate import tabulate

# 2-d array to represent game
game=[]
for i in range(11):
    g=[]
    for j in range(10):
        g.append(' ')
    game.append(g)

game[1][0]=1
for i in range(2,11):
    game[i][0]='**'
game[2][1]=2
for i in range(3,11):
    game[i][1]='**'

game[3][2]=3
for i in range(4,11):
    game[i][2]='**'

game[4][3]=5
for i in range(5,11):
    game[i][3]='**'

game[4][4]=8
for i in range(5,11):
    game[i][4]='**'

game[4][5]=16
for i in range(5,11):
    game[i][5]='**'

game[7][6]=24
for i in range(8,11):
    game[i][6]='**'

game[7][7]=50
for i in range(8,11):
    game[i][7]='**'

game[9][8]=74
for i in range(10,11):
    game[i][8]='**'

game[10][9]=124
game[0][0]='@'

print (tabulate(game, tablefmt="fancy_grid"))



game1=copy.deepcopy(game)

r = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
              [1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
              [0, 2, -1, -1, -1, -1, -1, -1, -1, -1],
              [0, 0, 3, -1, -1, -1, -1, -1, -1, -1],
              [0, 0, 0, 5, 8, 16, -1, -1, -1, -1],
              [0, 0, 0, 0, 0, 0, -1, -1, -1, -1],
              [0, 0, 0, 0, 0, 0, -1, -1, -1, -1],
              [0, 0, 0, 0, 0, 0, 24, 50, -1, -1],
              [0, 0, 0, 0, 0, 0, 0, 0, -1, -1],
              [0, 0, 0, 0, 0, 0, 0, 0, 74, -1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 124]])


#creating states
k=0
states=[]
for i in range(11):
    for j in range(10):
        if r[i][j]!=0:
            states.append([k,i,j])
            k=k+1
            
#Defining actions            

def goUp():
    global states
    negativeReward=0
    positiveReward=0
    for i in range(11):
        if '@' in game[i]:
            j=game[i].index('@')
            if i!=0 and game[i-1][j]==' ':
                game[i][j]=' '
                game[i-1][j]='@'
                negativeReward=1
                break
            elif i!=0 and type(game[i-1][j])==int:
                positiveReward=game[i-1][j]
                game[i][j]=' '
                game[i-1][j]='@'
                break
    m=0
    n=0
    for i in range(11):
        if '@' in game[i]:
            m=i
            n=game[i].index('@')
    h=0
    for each in states:
        if each[1]==m and each[2]==n:
            h=each[0]
    if negativeReward>0:
        return (h,negativeReward,0)
    if positiveReward>0:
        return (h,1,positiveReward)
    return (h,1,0)
                
def goDown():
    global states
    negativeReward=0
    positiveReward=0
    for i in range(11):
        if '@' in game[i]:
            j=game[i].index('@')
            if i!=10 and game[i+1][j]==' ':
                game[i][j]=' '
                game[i+1][j]='@'
                negativeReward=1
                break
            elif i!=10 and type(game[i+1][j])==int:
                positiveReward=game[i+1][j]
                game[i][j]=' '
                game[i+1][j]='@'
                break

    m=0
    n=0
    for i in range(11):
        if '@' in game[i]:
            m=i
            n=game[i].index('@')
    h=0
    for each in states:
        if each[1]==m and each[2]==n:
            h=each[0]
    if negativeReward>0:
        return (h,negativeReward,0)
    if positiveReward>0:
        return (h,1,positiveReward)
    return (h,1,0)

def goLeft():
    global states
    negativeReward=0
    positiveReward=0
    for i in range(11):
        if '@' in game[i]:
            j=game[i].index('@')
            if j!=0 and game[i][j-1]==' ':
                game[i][j]=' '
                game[i][j-1]='@'
                negativeReward=1
                break
            elif j!=0 and type(game[i][j-1])==int:
                positiveReward=game[i][j-1]
                game[i][j]=' '
                game[i][j-1]='@'
                break

    m=0
    n=0
    for i in range(11):
        if '@' in game[i]:
            m=i
            n=game[i].index('@')
    h=0
    for each in states:
        if each[1]==m and each[2]==n:
            h=each[0]
    if negativeReward>0:
        return (h,negativeReward,0)
    if positiveReward>0:
        return (h,1,positiveReward)
    return (h,1,0)


def goRight():
    global states
    negativeReward=0
    positiveReward=0
    for i in range(11):
        if '@' in game[i]:
            j=game[i].index('@')
            if j!=9 and game[i][j+1]==' ':
                game[i][j]=' '
                game[i][j+1]='@'
                negativeReward=1
                break
            elif j!=9 and type(game[i][j+1])==int:
                positiveReward=game[i][j+1]
                game[i][j]=' '
                game[i][j+1]='@'
                break
    m=0
    n=0
    for i in range(11):
        if '@' in game[i]:
            m=i
            n=game[i].index('@')
    h=0
    for each in states:
        if each[1]==m and each[2]==n:
            h=each[0]
    if negativeReward>0:
        return (h,negativeReward,0)
    if positiveReward>0:
        return (h,1,positiveReward)
    return (h,1,0)

            
def reset():
    global game
    game=copy.deepcopy(game1)
       
paretoList=[]
def qlearning(timeW,treasureW):
    #Initialize table with all zeros
    Q1 = np.zeros([61,4])
    Q2 = np.zeros([61,4])
    Q = np.zeros([61,4])
    # Set learning parameters
    lr = 0.9
    y = 0.95
    num_episodes = 15000
    paretoList=[]
    for i in range(num_episodes):
        #Reset environment and get first new observation
        reset()
        s = 0    
        rAll = 0
        totalTime=0
        treasure=0
        j=0
        #The Q-Table learning algorithm
        while totalTime<500:
            j=j+1
            #Choose an action by greedily (with noise) picking from Q table
            a = np.argmax(Q[s,:] + np.random.randn(1,4)*(20))   
            
            #Get new state and reward from environment
            if a==0:
                s1,time,treasure = goRight()
            if a==1:
                s1,time,treasure = goLeft()

            if a==2:
                s1,time,treasure = goUp()

            if a==3:
                s1,time,treasure = goDown()
                
            totalTime=totalTime+time

            #Update Q-Table with new knowledge
            Q1[s,a] = Q1[s,a] + lr*((-time) + y*np.max(Q1[s1,:]) - Q1[s,a])
            Q2[s,a] = Q2[s,a] + lr*(treasure + y*np.max(Q2[s1,:]) - Q2[s,a])
            Q[s,a]= timeW*Q1[s,a] + treasureW*Q2[s,a]
            s = s1
            if treasure>=1:
                break
        paretoList.append([-totalTime,treasure])
    
    print('Pareto Front')
    paretoFront=[]

    for each in paretoList:
        paretoFront.append(each)
        for e in paretoList:
            if (e[1]>each[1] and e[0]>=each[0]) or (e[1]>=each[1] and e[0]>each[0]) :
                paretoFront.remove(each)
                paretoList.remove(each)
                break
                
    paretoFront1=[]
    for each in paretoFront:
        if each not in paretoFront1:
            paretoFront1.append(each)
            
    print('The solutions in pareto front has rewards related to time and treasure respectively.')
    print(paretoFront1)
    return paretoFront1


r1_goal0=[]
r1_goal1=[]

#User intereaction for getting weights from user.

print('For scalarization function, 2 weights are needed, one for minimizing time consumption as an objective and another for treasure as an objective:')
timeW=float(input('Enter weight in the range of 0 to 1 as an emphasis you want to put on time minimization objective:'))
treasureW=float(input('Enter weight in the range of 0 to 1 as an emphasis you want to put on treasure objective:'))
result1=qlearning(timeW,treasureW)
for each in result1:
    r1_goal0.append(each[0])
    r1_goal1.append(each[1])
    

import matplotlib.pyplot as plt
plt.xlabel('Reward (Positive): Treasure')
plt.ylabel('Reward (Negative): Time')
plt.scatter(r1_goal1,r1_goal0,color='blue')
plt.show()
    
