#danial ramezani
#water cycle algorithm python code for ROP problem

import numpy as np
import matplotlib.pyplot
import random
import math
import time
t=time.time()
time.time()-t
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
UB_numberPeice = 6
#--------------------------------------------------------code variables-------------------------------------------------------------------------------------------------------------------------------------------------------
rainDrops = []
#--------------------------------------------------------parameters-------------------------------------------------------------------------------------------------------------------------------------------------------------
nRivers = 2
nsr = nRivers+1
n_pop = 1
dmax = 1e-16
max_it = 2
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
#--------------------------------------------------------(changing position(crossover))------------------------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------(evaporation(mutation))------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
#--------------------------------------------------------(intesnsity of flow)------------------------------------------------------------------------------------------------------------------------------------------------------------------
def intesnsityOfFlow(fit):
    nsn = []
    for i in range(len(fit)):
        nsn.append(math.ceil((fit[i]/sum(fit)*n_pop)))
    return (nsn)
#--------------------------------------------------------(rain)--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def rain():
    initial_rain = []
    toSortFitness = []
    sortedRain = []
    for _ in range(rainRate):
        d = create_rainDrops()
        if fitness(d)[0] > 0:
            initial_rain.append(d)
            toSortFitness.append(fitness(d)[0])
    n_pop = len(initial_rain)
    toSortFitness = np.array(toSortFitness)
    sortedFitness = np.argsort(toSortFitness)[::-1]#ARGSORTING
    for idx in sortedFitness:
        sortedRain.append(initial_rain[idx])
    toSortFitness = np.sort(toSortFitness)[::-1]
    nsn = intesnsityOfFlow(toSortFitness[:nsr])#calculating intensity of flow
    return sortedRain , nsn
#--------------------------------------------------------initializing------------------------------------------------------------------------------------------------------------------------------------------------------------------
rainDrops, nsn = rain()
print(rainDrops)
#--------------------------------------------------------Main loop-------------------------------------------------------------------------------------------------------------------------------------------------------------------
for i in range(max_it):
    pass

