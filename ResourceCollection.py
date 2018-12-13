# The solutions gives pareto front which has rewards related to enemy,gold and gems respectively.

import copy
import random
import time
import numpy as np
from tabulate import tabulate

#Representing game in 2-d array
game=[]
for i in range(5):
    g=[]
    for j in range(5):
        g.append(' ')
    game.append(g)

game[0][2]='GOLD'
game[0][3]='THIEVES'
game[1][2]='THIEVES'
game[1][4]='GEMS'
game[4][2]='@'

#creating states
k=0
states=[]
for i in range(5):
    for j in range(5):
        states.append([k,i,j])
        k=k+1

game1=copy.deepcopy(game)

print (tabulate(game, tablefmt="fancy_grid"))

#Defining actions

def goUp():
    enemy=0
    gold=0
    gem=0
    global game
    for i in range(5):
        if '@' in game[i]:
            j=game[i].index('@')
            if i!=0 and game[i-1][j]==' ':
                game[i][j]=' '
                game[i-1][j]='@'
                break
            elif i!=0 and game[i-1][j]=='GOLD':
                gold=1
                game[i][j]=' '
                game[i-1][j]='@'
                break
            elif i!=0 and game[i-1][j]=='GEMS':
                gems=1
                game[i][j]=' '
                game[i-1][j]='@'
                break
            elif i!=0 and game[i-1][j]=='THIEVES':
                if random.random()<0.1:
                    enemy=-1
                    game[i][j]=' '
                    game[4][2]='@'
                    break
                else:
                    game[i][j]=' '
                    game[i-1][j]='@ THIEVES'
                    break
        elif '@ THIEVES' in game[i]:
            j=game[i].index('@ THIEVES')
            if i!=0 and game[i-1][j]==' ':
                game[i][j]='THIEVES'
                game[i-1][j]='@'
                break
            elif i!=0 and game[i-1][j]=='GOLD':
                gold=1
                game[i][j]='THIEVES'
                game[i-1][j]='@'
                break
            elif i!=0 and game[i-1][j]=='GEMS':
                gems=1
                game[i][j]='THIEVES'
                game[i-1][j]='@'
                break
            elif i!=0 and game[i-1][j]=='THIEVES':
                if random.random()<0.1:
                    enemy=-1
                    game[i][j]='THIEVES'
                    game[4][2]='@'
                    break
                else:
                    game[i][j]='THIEVES'
                    game[i-1][j]='@ THIEVES'
                    break
    m=0
    n=0
    for i in range(5):
        if '@' in game[i]:
            m=i
            n=game[i].index('@')
    h=0
    for each in states:
        if each[1]==m and each[2]==n:
            h=each[0]
    return (h,enemy,gold,gem)
                    
                
def goDown():
    enemy=0
    gold=0
    gem=0
    global game
    for i in range(5):
        if '@' in game[i]:
            j=game[i].index('@')
            if i!=4 and game[i+1][j]==' ':
                game[i][j]=' '
                game[i+1][j]='@'
                break
            elif i!=4 and game[i+1][j]=='GOLD':
                gold=1
                game[i][j]=' '
                game[i+1][j]='@'
                break
            elif i!=4 and game[i+1][j]=='GEMS':
                gems=1
                game[i][j]=' '
                game[i+1][j]='@'
                break
            elif i!=4 and game[i+1][j]=='THIEVES':
                if random.random()<0.1:
                    enemy=-1
                    game[i][j]=' '
                    game[4][2]='@'
                    break
                else:
                    game[i][j]=' '
                    game[i+1][j]='@ THIEVES'
                    break
        elif '@ THIEVES' in game[i]:
            j=game[i].index('@ THIEVES')
            if i!=4 and game[i+1][j]==' ':
                game[i][j]='THIEVES'
                game[i+1][j]='@'
                break
            elif i!=4 and game[i+1][j]=='GOLD':
                gold=1
                game[i][j]='THIEVES'
                game[i+1][j]='@'
                break
            elif i!=4 and game[i+1][j]=='GEMS':
                gems=1
                game[i][j]='THIEVES'
                game[i+1][j]='@'
                break
            elif i!=4 and game[i+1][j]=='THIEVES':
                if random.random()<0.1:
                    enemy=-1
                    game[i][j]='THIEVES'
                    game[4][2]='@'
                    break
                else:
                    game[i][j]='THIEVES'
                    game[i+1][j]='@ THIEVES'
                    break
    m=0
    n=0
    for i in range(5):
        if '@' in game[i]:
            m=i
            n=game[i].index('@')
    h=0
    for each in states:
        if each[1]==m and each[2]==n:
            h=each[0]
    return (h,enemy,gold,gem)


def goLeft():
    enemy=0
    gold=0
    gem=0
    global game
    for i in range(5):
        if '@' in game[i]:
            j=game[i].index('@')
            if j!=0 and game[i][j-1]==' ':
                game[i][j]=' '
                game[i][j-1]='@'
                break
            elif j!=0 and game[i][j-1]=='GOLD':
                gold=1
                game[i][j]=' '
                game[i][j-1]='@'
                break
            elif j!=0 and game[i][j-1]=='GEMS':
                gem=1
                game[i][j]=' '
                game[i][j-1]='@'
                break
            elif j!=0 and game[i][j-1]=='THIEVES':
                if random.random()<0.1:
                    enemy=-1
                    game[i][j]=' '
                    game[4][2]='@'
                    break
                else:
                    game[i][j]=' '
                    game[i][j-1]='@ THIEVES'
                    break
        elif '@ THIEVES' in game[i]:
            j=game[i].index('@ THIEVES')
            if j!=0 and game[i][j-1]==' ':
                game[i][j]='THIEVES'
                game[i][j-1]='@'
                break
            elif j!=0 and game[i][j-1]=='GOLD':
                gold=1
                game[i][j]='THIEVES'
                game[i][j-1]='@'
                break
            elif j!=0 and game[i][j-1]=='GEMS':
                gem=1
                game[i][j]='THIEVES'
                game[i][j-1]='@'
                break
            elif j!=0 and game[i][j-1]=='THIEVES':
                if random.random()<0.1:
                    enemy=-1
                    game[i][j]='THIEVES'
                    game[4][2]='@'
                    break
                else:
                    game[i][j]='THIEVES'
                    game[i][j-1]='@ THIEVES'
                    break
    m=0
    n=0
    for i in range(5):
        if '@' in game[i]:
            m=i
            n=game[i].index('@')
    h=0
    for each in states:
        if each[1]==m and each[2]==n:
            h=each[0]
    return (h,enemy,gold,gem)
    

def goRight():
    enemy=0
    gold=0
    gem=0
    global game
    for i in range(5):
        if '@' in game[i]:
            j=game[i].index('@')
            if j!=4 and game[i][j+1]==' ':
                game[i][j]=' '
                game[i][j+1]='@'
                break
            elif j!=4 and game[i][j+1]=='GOLD':
                gold=1
                game[i][j]=' '
                game[i][j+1]='@'
                break
            elif j!=4 and game[i][j+1]=='GEMS':
                gem=1
                game[i][j]=' '
                game[i][j+1]='@'
                break
            elif j!=4 and game[i][j+1]=='THIEVES':
                if random.random()<0.1:
                    enemy=-1
                    game[i][j]=' '
                    game[4][2]='@'
                    break
                else:
                    game[i][j]=' '
                    game[i][j+1]='@ THIEVES'
                    break
        elif '@ THIEVES' in game[i]:
            j=game[i].index('@ THIEVES')
            if j!=4 and game[i][j+1]==' ':
                game[i][j]='THIEVES'
                game[i][j+1]='@'
                break
            elif j!=4 and game[i][j+1]=='GOLD':
                gold=1
                game[i][j]='THIEVES'
                game[i][j+1]='@'
                break
            elif j!=4 and game[i][j+1]=='GEMS':
                gem=1
                game[i][j]='THIEVES'
                game[i][j+1]='@'
                break
            elif j!=4 and game[i][j+1]=='THIEVES':
                if random.random()<0.1:
                    enemy=-1
                    game[i][j]='THIEVES'
                    game[4][2]='@'
                else:
                    game[i][j]='THIEVES'
                    game[i][j+1]='@ THIEVES'
                    break

    m=0
    n=0
    for i in range(5):
        if '@' in game[i]:
            m=i
            n=game[i].index('@')
    h=0
    for each in states:
        if each[1]==m and each[2]==n:
            h=each[0]
    return (h,enemy,gold,gem)
            

def reset():
    global game
    game=copy.deepcopy(game1)
    
paretoList=[]
def qlearning(enemyW,goldW,gemW):
    #Initialize table with all zeros
    Q1 = np.zeros([25,4])
    Q2 = np.zeros([25,4])
    Q3 = np.zeros([25,4])
    Q = np.zeros([25,4])
    # Set learning parameters
    lr = 0.9
    y = 0.95
    num_episodes = 1000
    paretoList=[]
    for i in range(num_episodes):
        #Reset environment and get first new observation
        reset()
        s = 0    
        rAll = 0
        #The Q-Table learning algorithm
        while True:
            #Choose an action by greedily (with noise) picking from Q table
            a = np.argmax(Q[s,:] + np.random.randn(1,4)*(5./(1)))
            #Get new state and rewards from environment
            if a==0:
                s1,enemy,gold,gem = goRight()
            if a==1:
                s1,enemy,gold,gem = goLeft()

            if a==2:
                s1,enemy,gold,gem = goUp()

            if a==3:
                s1,enemy,gold,gem = goDown()
                

            #Update Q-Table with new knowledge
            Q1[s,a] = Q1[s,a] + lr*((enemy) + y*np.max(Q1[s1,:]) - Q1[s,a])
            Q2[s,a] = Q2[s,a] + lr*(gold + y*np.max(Q2[s1,:]) - Q2[s,a])
            Q3[s,a] = Q3[s,a] + lr*(gem + y*np.max(Q3[s1,:]) - Q3[s,a])
            Q[s,a]= enemyW*Q1[s,a] + goldW*Q2[s,a] + gemW*Q3[s,a]
            s = s1
            if gold>=1:
                break
            if gem>=1:
                break

        paretoList.append([enemy,gold,gem])
    
    print('Pareto Front')
    paretoFront=[]

    for each in paretoList:
        paretoFront.append(each)
        for e in paretoList:
            if (e[0]>each[0] and e[1]>=each[1] and e[2]>=each[2]) or (e[0]>=each[0] and e[1]>each[1] and e[2]>=each[2]) or (e[0]>=each[0] and e[1]>=each[1] and e[2]>each[2]):
                paretoFront.remove(each)
                paretoList.remove(each)
                break
                
    paretoFront1=[]
    for each in paretoFront:
        if each not in paretoFront1:
            paretoFront1.append(each)
    print('The solutions in pareto front has rewards related to enemy,gold and gems respectively.')
    print(paretoFront1)
    
#Use interaction for getting weights

print('For scalarization function, 3 weights are needed, for avoiding enemy attack as an objective, gold as an objective and gems as an objective:')
enemyW=float(input('Enter weight in the range of 0 to 1 as an emphasis you want put on avoiding enemy attack objective:'))
goldW=float(input('Enter weight in the range of 0 to 1 as an emphasis you want put on gold objective:'))
gemW=float(input('Enter weight in the range of 0 to 1 as an emphasis you want put on gems objective:'))

qlearning(enemyW,goldW,gemW)


