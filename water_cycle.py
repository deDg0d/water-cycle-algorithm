#danial ramezani
#https://github.com/deDg0d/water-cycle-algorithm/blob/main/WCA.py
#water cycle algorithm python code for ROP problem
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
#--------------------------------------------------------problem data----------------------------------------------------------------------------------
lambdaValue=[[0],
[0.00532 ,0.00818 ,0.0133 ,0.00741 ,0.00619, 0.00436 ,0.0105, 0.015, 0.00268, 0.0141, 0.00394, 0.00236, 0.00215, 0.011],
[0.000726 ,0.000619, 0.011, 0.0124, 0.00431, 0.00567, 0.00466, 0.00105, 0.000101, 0.00683, 0.00355, 0.00769, 0.00436, 0.00834],
[0.00499 ,0.00431, 0.0124, 0.00683, 0.00818, 0.00268, 0.00394, 0.0105, 0.000408, 0.00105, 0.00314, 0.0133, 0.00665, 0.00355],
[0.00818 ,0.0000, 0.00466, 0.0000, 0.0000, 0.000408 ,0.0000, 0.0000, 0.000943, 0.0000, 0.0000, 0.0110, 0.0000, 0.00436]]
kValue=[[0],
[2 ,3, 3, 2, 1, 3 ,3, 3, 2, 3, 2, 1, 2, 3],
[1 ,1 ,3 ,3 ,2 ,3 ,2 ,1 ,1 ,2 ,2 ,2 ,3, 1],
[2, 2, 3, 2, 3, 2, 2, 3, 1, 1, 2, 3, 3, 2],
[3, 0, 2, 0, 0, 1, 0 ,0 ,1 ,0 ,0 ,3 ,0 ,3]]
cValue=[[0],
[1 ,2 ,2 ,3 ,2 ,3 ,4 ,3 ,2 ,4 ,3 ,2 ,2 ,4],
[1 ,1 ,3 ,4 ,2 ,3 ,4 ,5 ,3 ,4 ,4 ,3 ,3 ,4],
[2 ,1 ,1 ,5 ,3, 2 ,5 ,6 ,4 ,5 ,5 ,4 ,2 ,5],
[2 ,0 ,4 ,0 ,0 ,2 ,0 ,0 ,3 ,0 ,0 ,5 ,0 ,6]]
wValue=[[0],
[3 ,8 ,7 ,5 ,4 ,5 ,7 ,4 ,8 ,6 ,5 ,4 ,5, 6],
[4 ,10 ,5 ,6 ,3 ,4 ,8 ,7 ,9 ,5 ,6 ,5 ,5,7],
[2 ,9 ,6 ,4 ,5 ,5 ,9 ,6 ,7 ,6 ,6 ,6 ,6, 6],
[5 ,0 ,4 ,0 ,0 ,4 ,0 ,0 ,8 ,0 ,0 ,7 ,0 ,9]]
Wmax = 170
Cmax = 130
alfa = 1.1
beta = 1.05
tValue = 100
nfe = 0
subsystems = 14
LB_typePeice = 1
UB_typePeice = 4
LB_numberPeice = 1
UB_numberPeice = 4
#--------------------------------------------------------code variables-------------------------------------------------------------------------------------------------------------------------------------------------------
rainDrops = []
bestFitInitialy = 0
all_best_fitness = []
#--------------------------------------------------------parameters-------------------------------------------------------------------------------------------------------------------------------------------------------------
nRivers = 2
nsr = nRivers+1
n_pop = 1
dmax = 0.01
max_it = 100
rainRate = 1000
#/////////////////////////////////////////////////////////////////////functions\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#--------------------------------------------------------(produce solution)------------------------------------------------------------------------------------------------------------------------------------------------------------------
def create_rainDrops(number=1): #create new rainDrop
    new_solutions = []
    for _ in range(number):
        for _ in range(subsystems):#type
            new_solutions.append(random.choice(np.arange(LB_typePeice,UB_typePeice)))
        for _ in range(subsystems):#number
            new_solutions.append(random.choice(np.arange(LB_numberPeice,UB_numberPeice)))
    new_solutions = np.array(new_solutions)
    new_solutions = new_solutions.reshape(number, int(len(new_solutions)/(subsystems*number)) , subsystems)#3d array
    return new_solutions
#-------------------------------------------------------------(objective function)-------------------------------------------------------------------------------------------------------------------------------------------------------------------
def objective(landa, k, t):
    # nfe+=1
    sigma = 0
    if k==0:
        return 0
    else:
        for i in range(k):
            sigma+=(landa*t)**i/math.factorial(i)
        return math.exp(-landa*t)*sigma
#--------------------------------------------------------(fitness function)------------------------------------------------------------------------------------------------------------------------------------------------------------------
def fitness(rainDrop):
    reliabilityForEach = []
    reliabilityTotal = []
    for i in range(len(rainDrop)):
        reliabilityForEach.clear()
        k=0
        violateCost = 0
        violateWeight = 0
        for typePeice,n in zip(rainDrop[i,0],rainDrop[i,1]):
            r = objective(lambdaValue[int(typePeice)][k], kValue[int(typePeice)][k], tValue)
            reliabilityForEach.append(1-(1-r)**n)
            violateCost += cValue[int(typePeice)][k]*n
            violateWeight += wValue[int(typePeice)][k]*n
            penalty1 = max(violateCost-Cmax,0)#penalty cost
            penalty2 = max(violateWeight-Wmax,0)#penalty weight
            k+=1
        reliabilityTotal.append(np.prod(reliabilityForEach)-(alfa*penalty1)-(beta*penalty2))
    return (reliabilityTotal)
#--------------------------------------------------------(changing position stream and rivers related to sea(crossover))------------------------------------------------------------------------------------------------------------------------------------------------------------------
def changePositionSea(sea, streamRiver):
    evaporation = False
    #streamRiver[number][][0=peice1=number][element]
    #sea[0][0=peice1=number][element]
    newStreamFit = []
    seaFit = fitness(sea)
    for i in range(len(streamRiver)): #modified: crossover for all stream|original: crossover on ns[0] number of streams
        rand = np.random.choice(np.arange(1,subsystems))
        for j in range(rand):
            streamRiver[i][0][0][j] = sea[0][0][j]  #randth first element from sea #peice
            streamRiver[i][0][1][j] = sea[0][1][j]  #randth first element from sea #number
    for i in range(len(streamRiver)): #if stream was better than sea=> swap river and sea
        newStreamFit.append(fitness(streamRiver[i]))
        if newStreamFit[i] >seaFit:
            sea , streamRiver[i] = streamRiver[i] , sea
    if len(streamRiver) == nRivers:#if it was river not stream
        for j in range(len(streamRiver)):
            if abs(seaFit[0] - newStreamFit[j][0])<dmax:
                evaporation = True # rainning process
    return sea , streamRiver, evaporation
#--------------------------------------------------------------(changing position stream related to rivers(crossover))--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def change_position_river_stream(river,stream,ns):
    ns = ns[1:nRivers+1]
    newStreamFit = []
    random_j = []
    for i in range(nRivers): #change pos related to river i
        for _ in range(ns[i]): #change pos of ns[i]th stream related to river i|they will be chosen randomly
            rand = np.random.choice(np.arange(1,subsystems))
            j = np.random.choice(np.arange(1,len(stream))) #pick random stream to change position related to river 1
            random_j.append(j)
            for k in range(rand):
                stream[j][0][0][k] = rivers[i][0][0][k]
                stream[j][0][1][k] = rivers[i][0][1][k]
    k=0 #to compare new streams fitness versus river
    for i in range(len(river)): #compare all streams' fit with river ith
        for j in random_j:
            newStreamFit.append(fitness(stream[j]))
            if newStreamFit[k]>fitness(river[i]):
                stream[j] , river[i] = river[i] , stream[j]
            k+=1
    return stream, river
#--------------------------------------------------------(intesnsity of flow),modification of nsn------------------------------------------------------------------------------------------------------------------------------------------------------------------
def intesnsityOfFlow(cs,n_pop):
    cn = cs - min(cs)
    ns = []
    for i in range(len(cn)):
        ns.append(round((cn[i]/sum(cn)*(n_pop-nsr)))) #*number of streams
    nStream = n_pop-nsr
    while sum(ns)>nStream:
        ns = [i-1 for i in ns]
    while sum(ns)<nStream:
        ns = [i+1 for i in ns]
    if 0 in ns:
        index = ns.index(0)
        while ns[index]==0:
            ns[index]+=round(ns[0]/6)
            ns[0]-=round(ns[0]/6)
    return (ns)
#--------------------------------------------------------------(changing dmax value)--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def change_dmax():
    new_dmax = dmax - dmax/max_it
    return new_dmax
#--------------------------------------------------------------(rain,evaporation(mutation))--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def rain(): #creating new points
    print('rain')
    initial_rain = []
    toSortFitness = []
    sortedRain = []
    for _ in range(rainRate):
        d = create_rainDrops()
        if fitness(d)[0] > 0.5:
            initial_rain.append(d)
            toSortFitness.append(fitness(d)[0])
    n_pop = len(initial_rain)
    toSortFitness = np.array(toSortFitness)
    sortedFitness = np.argsort(toSortFitness)[::-1]#ARGSORTING
    for idx in sortedFitness:
        sortedRain.append(initial_rain[idx])
    toSortFitness = np.sort(toSortFitness)[::-1]
    nsn = intesnsityOfFlow(toSortFitness[:nRivers+2],n_pop)#calculating intensity of flow
    return sortedRain , toSortFitness , nsn
#--------------------------------------------------------initializing------------------------------------------------------------------------------------------------------------------------------------------------------------------
rainDrops ,fit ,ns = rain()
bestFitInitialy = fit[0]
sea = rainDrops[0]
seaFit = fit[0]
rivers = rainDrops[1:nRivers+1]
riverFit = fit[1:nRivers+1]
stream = rainDrops[nRivers+1:] 
streamFit = fit[nRivers+1:]
#--------------------------------------------------------Main loop-------------------------------------------------------------------------------------------------------------------------------------------------------------------
for i in range(max_it):
    sea , stream, evaporation = changePositionSea(sea, stream ) #move streams toward sea #ns[0] is number of streams
    sea , rivers, evaporation = changePositionSea(sea, rivers ) #move rivers toward sea #ns[0] is number of streams
    stream , rivers = change_position_river_stream(rivers,stream,ns) #move streams toward rivers
    if evaporation == True: #check for evaporation condition
        stream, streamFit, ns= rain() #rainning process
    seaFit = fitness(sea)
    print(f' iteration {i} sea {seaFit} best initial point is {bestFitInitialy} dmax {dmax}  max find in ran{max(streamFit)}') #run status
    all_best_fitness.append(seaFit[0])
    dmax = change_dmax()#changing dmax value
#--------------------------------------------------------Main loop-------------------------------------------------------------------------------------------------------------------------------------------------------------------
iteration = [i for i in range(max_it)]
plt.plot(iteration,all_best_fitness)
plt.show()