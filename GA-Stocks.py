# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 12:30:29 2020
mutation = cupy
@author: Wai
windows is optimize with GA : use slope and sharpe  ; for stocks w/o numba ; with SPDN and slope = sharpe_cum
#https://github.com/nopaixx/TensorFlow-GeneticsAlgo/blob/master/train.ipynb
#https://stackoverflow.com/questions/6148207/linear-regression-with-matplotlib-numpy
#https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.mstats.theilslopes.html?fbclid=IwAR1GjMv5J3W5u5Dzd3aENxqXgwi9SpNw_f2GRyG0i4jEzRGGDZWLB4u89yI
"""

#import cupy as cp
import numpy as np
import random
import pandas as pd
#from sklearn.covariance import LedoitWolf
import nonlinshrink as nls
from scipy import stats
from multiprocessing import Pool
import matplotlib.pyplot as plt
#from numba import double
#from numba import jit
#from numba.decorators import jit, autojit
#from pypfopt import risk_models
#pool = Pool(4)
#import os
#from itertools import product
#import timeit

import time
        
def check_duplicates(population,pop):
    
   stocks1 = get_stock(population,pop)
   if len(stocks1) != len(set(stocks1)):
       
        
        return 1
    
def get_num_stock(population,pop): # number of stocks in portfolio
    a = population[pop,:6] #starting allele of number of stocks in portfolio
        #print('a',a)
        # binary_stock = [1,2,4,8,16]

    b = int(s_floor + ((np.dot(np.asarray(a),np.asarray(binary_stock)))/62.0) * (s_ceiling-s_floor))
    #b = int(s_floor + ((np.dot(np.asarray(a),np.asarray(binary_stock)))/50.0) * (s_ceiling-s_floor))
    #print('b',b)
    return b
    
def get_stock(population,pop): # get stock code/ticker
  #print('p1',population[5:15,0])
   #stock number / stock name
   stocks = []
   s = get_num_stock(population,pop)
   #s = s+1 # add SPDN
   #print('s',s)
  #print('p1',population[:5,i])
   t = 6  #starting allele of stocks code to number of stocks derive from s
   for i in range(1,s+1):
       a = population[pop,t:t+10]
      #print('a',a)
       t +=10

       c = int(minS + ((np.dot(np.asarray(a),np.asarray(binary)))/1022.0) * (maxS - minS))
      
       stocks.insert(i, c)
      #print('Stocks',stocks)
   stocks.insert(i,0)
   return stocks

def get_weights(population,pop):  # number of weights depend on number of stocks in portfolio
            weights = []
            s = get_num_stock(population,pop)
            s = s + 1
        #print('s',s)
            t = 206 + 10 #starting allele of weights
            for i in range(1,s+1):

            #a = population[t:(t+10)]
                a = population[pop,t:t+10]
   
                t +=10

                c = minW + ((np.dot(np.asarray(a),np.asarray(binary)))/1022.0) * (maxW - minW)
            #c = minW + numa

            
                weights.insert(i, c)
        #weights = tf.reshape(weights,[1,s])
        #print('w',weights)

            if minW == -2:
                    new_weights = map(lambda x: abs(x),weights)
                    sumW = np.sum(new_weights)
            else:
                sumW = sum(weights)
            #print('ss',sumW)

            weights_list = map(lambda x: x/sumW, weights)
            weights_list = list(weights_list)
        #print('w1',tf.reduce_sum(weights_list))
            return weights_list
        


def get_windows(population,pop): # number of stocks in portfolio
    #a = population[pop,406:412] #starting allele of number of stocks in portfolio
    a = population[pop,416:422] #starting allele of number of stocks in portfolio
        #print('a',a)
        # binary_stock = [1,2,4,8,16]

    b = int(w_floor + ((np.dot(np.asarray(a),np.asarray(binary_stock)))/62.0) * (w_ceiling-w_floor))
    #b = int(s_floor + ((np.dot(np.asarray(a),np.asarray(binary_stock)))/50.0) * (s_ceiling-s_floor))
    #print('b',b)
    return b

def get_sharpe(population,pop):
        a = population[pop,422:]
        b = sharpe_min + ((np.dot(np.asarray(a),np.asarray(binary_stock)))/62.0) * (sharpe_max-sharpe_min)
        
        #print('Perfect',b)
        return b

def get_num_stockb(chromosome):
        a = best[:6]

        num = np.dot(np.asarray(a),np.asarray(binary_stock))
        numa = num / 62.0
    #b = 5+((numa)*(20-5))
        c = numa*(s_ceiling-s_floor)
        b = s_floor+c
        b = int(b)
        return b

def get_windowsb(chromosome):
        a = best[416:422]

        num = np.dot(np.asarray(a),np.asarray(binary_stock))
        numa = num / 62.0
    #b = 5+((numa)*(20-5))
        c = numa*(w_ceiling-w_floor)
        b = w_floor+c
        b = int(b)
        return b
    
def get_sharpeb(chromosome):
        a = best[422:]

        b = sharpe_min + ((np.dot(np.asarray(a),np.asarray(binary_stock)))/62.0) * (sharpe_max-sharpe_min)
        #b = int(b)
        print('Perfect Sharpe: ',b)
        return b

def get_stockb(chromosome):
    
    
     stocks = []
     s = get_num_stockb(chromosome)
     
     #print('s',s)
     t = 6  #starting allele of stocks
     for i in range(1,s+1):
         

         a = best[t:(t+10)]

         t +=10
         num = np.dot(np.asarray(a),np.asarray(binary))
         numa = num / 1022.0
         numa = numa * (maxS - minS)
         c = minS + numa
         c = int(c)
        #c = int(minS + ((num / 1022) * (maxS - minS)))
         stocks.insert(i, c)
        #print(stocks)
     stocks.insert(i,0)
     return stocks
    
def get_weightsb(chromosome):
        weights = []
        s = get_num_stockb(chromosome)
        s = s + 1
        t = 206 + 10 #starting allele of weights
        for i in range(1,s+1):

            a = best[t:(t+10)]
        #print('alle',a)
            t +=10
            num = np.dot(np.asarray(a),np.asarray(binary))
            numa = num / 1022.0
            numa = numa * (maxW - minW)
            c = minW + numa

        #c = minW + ((num / 1022) * (maxW - minW))
            weights.insert(i, c)

        if minW == -2:
                new_weights = map(lambda x: abs(x),weights)
                sumW = sum(new_weights)
        else:
            sumW = sum(weights)
            #print('w',weights)

        #weights_list = map(lambda x: x/sumW, weights)
        return weights #weights_list       

def create_starting_population(individuals, chromosome_length):
    # Set up an initial array of all zeros
        population = np.zeros((individuals, chromosome_length))
    # Loop through each row (individual)
        for i in range(individuals):
        # Choose a random number of ones to create
            ones = random.randint(0, chromosome_length)
        # Change the required number of zeros to ones
            population[i, 0:ones] = 1
        # Sfuffle row
            np.random.shuffle(population[i])
    
        return population

def select_individual_by_tournament1(population, scores):
    
    # Get population size
        population_size = len(scores)
        new_population1 = []
    #print(scores)
      # Pick individuals for tournament
        fighter_1 = random.randint(0, population_size-1)
        fighter_2 = random.randint(0, population_size-1)
   
    # Get fitness score for each
        fighter_1_fitness = scores[fighter_1]
    #print('f1',fighter_1_fitness)
        fighter_2_fitness = scores[fighter_2]
    
        if fighter_1_fitness >= fighter_2_fitness:
            winner = fighter_1
        else:
            winner = fighter_2
        
        parent_1 =  population[winner, :]
        
        fighter_1 = random.randint(0, population_size-1)
        fighter_2 = random.randint(0, population_size-1)
   
    # Get fitness score for each
        fighter_1_fitness = scores[fighter_1]
    #print('f1',fighter_1_fitness)
        fighter_2_fitness = scores[fighter_2]
    
        if fighter_1_fitness >= fighter_2_fitness:
            winner = fighter_1
        else:
            winner = fighter_2
    
        parent_2 =  population[winner, :]
    
        chromosome_length = len(parent_1)
   
        crossover_point = random.randint(1,chromosome_length-1)
    
    # Create children. np.hstack joins two arrays
        #t1 = [parent_1[0:crossover_point]]
        #t2 = [parent_2[crossover_point:]]
   
        child_1 = np.hstack((parent_1[0:crossover_point],
                        parent_2[crossover_point:]))
    
        child_2 = np.hstack((parent_2[0:crossover_point],
                        parent_1[crossover_point:]))
    
        new_population1.append(child_1)
        new_population1.append(child_2)
    
        return new_population1

#@jit(nopython=True)
def randomly_mutate_population(population,mutation_probability):
#def randomly_mutate_population(population):
    
        #mutation_probability = 0.005
    # Apply random mutation
        population = np.array(population)
        random_mutation_array = np.random.random(
            size=(population.shape))
        
        random_mutation_boolean = \
                random_mutation_array <= mutation_probability

        population[random_mutation_boolean] = \
        np.logical_not(population[random_mutation_boolean])
        population = np.array(population)
        # Return mutation population
        return population
    
#@jit(nopython=True)
def mutation(population,pop):
    mutation_rate = 0.005
    population_nextgen = []
    
        
    chromosome = population[pop]
    #print('c',chromosome)
    for j in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[j]= not chromosome[j]
    population_nextgen.append(chromosome)
    #print(population_nextgen)
    return population_nextgen

def get_mean_cov():
    
    return


#############################################################################
def eval(population,pop):
    
      score = [] #new_population = tf.zeros([1, chromosome_length], tf.int32)
      
      compute = check_duplicates(population,pop)
      if compute == 1:
      #aa = 1
          score3 = 0.0
          score.append(score3)
      #print('score',score2)
      else:
            
          #num_of_stock = get_num_stock(population,pop)
          try:
              
              
              stocks_list = get_stock(population,pop)
              weights_list = get_weights(population,pop)
          #print('wl',weights_list)
              weights_list1 = np.asarray(weights_list)
              score2 = 0
              score3 = 0
              windows = get_windows(population,pop)
              windows_forward = window_size - windows
              perfect_sharpe = get_sharpe(population,pop)
              k = 0
              slope = []
              sharpe_cum = []
              for j in range(windows_forward):
              
                  returns1 = returns.iloc[k:windows,stocks_list]
              
                  mean_return = returns[k:windows].mean()

              #cov = returns[k:windows].cov()
                  cov = nls.shrink_cov(returns1)
              #cov = returns1.cov()
                  covMatrix = cov * annualize
              #cov = risk_models.CovarianceShrinkage(returns[k:windows]).ledoit_wolf()
              
              
              
                  mr = mean_return.iloc[stocks_list]
              #a = cov.loc[:,cov.columns[stocks_list]]
              #covMatrix = (a.iloc[stocks_list]) * annualize
                  meanDailyReturns = mr * annualize
                  meanDailyReturns = np.array(meanDailyReturns)
              #covMatrix = np.array(covMatrix)
      
                  portReturn = np.sum(meanDailyReturns.T*weights_list1)
                  print('Port Ret: ',portReturn)
      
                  portStdDev = np.sqrt(np.dot(weights_list1.T, np.dot(covMatrix, weights_list1)))
     

                  sharpe = (portReturn - risklessRate) / portStdDev  #Sharpe ratio
                  sharpe_cum.append(sharpe)
                  #print('Inside Sharpe: ',sharpe)
                  if sharpe > perfect_sharpe:
                  
                      sharpe = perfect_sharpe/sharpe
                  else:
                      sharpe = sharpe/perfect_sharpe
              #sharpe1 = (portReturn - risklessRate) / portStdDev  #Sharpe ratio
              #if sharpe > sharpe_max:
                  #sharpe = sharpe_max
                  
                  score2 = score2 + sharpe
                  k = k + 1
                  windows = windows + 1
                  slope.append(sharpe)
              X = np.arange(1,k+1,1)
              
              #slope1 = stats.theilslopes(slope, X, 0.90)[0] #stats.linregress(X,slope)[0]
              slope1 = stats.theilslopes(sharpe_cum, X, 0.90)[0] #stats.linregress(X,slope)[0]
              #print('slope',slope1)
             
              score3 = score2 / windows_forward
          #score3 = score3 + (1+ slope1)**2
              #print('Loop Sharpe ',score3)
          #score.append(score2)
              if score3 > perfect_sharpe:
                  
                  score3 = (perfect_sharpe/score3) + slope1
                  #score3 = perfect_sharpe/(score3 - slope1)
              else:
                  score3 = (score3/perfect_sharpe) + slope1 
                  #score3 = (score3+slope1)/perfect_sharpe 
          
             # if score3 > perfect_sharpe:
                  
                  #score3 = (perfect_sharpe/score3) + slope1 #(1+slope1^2)
              #else:
                  #score3 = (score3/perfect_sharpe) + slope1 #(1+slope1^2)
          except:
              score3 = -1.0
              pass
 
      return score3 

#############################################################################
def eval_test(population,pop):
    
      score = [] #new_population = tf.zeros([1, chromosome_length], tf.int32)
      
      compute = check_duplicates(population,pop)
      if compute == 1:
      #aa = 1
          score3 = 0.0
          score.append(score3)
      #print('score',score2)
      else:
            
          #num_of_stock = get_num_stock(population,pop)
          try:
              
              
              stocks_list = get_stock(population,pop)
              weights_list = get_weights(population,pop)
              stocks_name = prices.columns[stocks_list]
              stocks_name = list(stocks_name)
              print('Stocks: ',stocks_name)
              print('Weights: ',weights_list)
              weights_list1 = np.asarray(weights_list)
              score2 = 0
              score3 = 0
              windows = get_windows(population,pop)
              windows_forward = window_size - windows
              perfect_sharpe = get_sharpe(population,pop)
              k = 0
              slope = []
              sharpe_cum = []
              mean_price = []
              for j in range(windows_forward):
              
                  returns1 = returns.iloc[k:windows,stocks_list]
              
                  mean_return = returns[k:windows].mean()
                  #prices1 = prices.iloc[k:windows,stocks_list].mean() # stock prices
                  prices2 = prices.iloc[k:windows,stocks_list]
                  prices1 = prices2.mean()

              #cov = returns[k:windows].cov()
                  cov = nls.shrink_cov(returns1)
              #cov = returns1.cov()
                  covMatrix = cov * annualize
              #cov = risk_models.CovarianceShrinkage(returns[k:windows]).ledoit_wolf()
              
                  mr = mean_return.iloc[stocks_list]
                  #mean_price = prices1.iloc[stocks_list]
                  port_price = np.dot(weights_list1,prices1)
                  #print('Mean Price: ', port_price)
              #a = cov.loc[:,cov.columns[stocks_list]]
              #covMatrix = (a.iloc[stocks_list]) * annualize
                  meanDailyReturns = mr * annualize
                  meanDailyReturns = np.array(meanDailyReturns)
              #covMatrix = np.array(covMatrix)
      
                  portReturn = np.sum(meanDailyReturns.T*weights_list1)
                  print('Port Ret: ',portReturn)
      
                  portStdDev = np.sqrt(np.dot(weights_list1.T, np.dot(covMatrix, weights_list1)))
     

                  sharpe = (portReturn - risklessRate) / portStdDev  #Sharpe ratio
                  sharpe_cum.append(sharpe)
                  mean_price.append(port_price)
                  #mean_price.append(portReturn)
                  #print('Inside Sharpe: ',sharpe)
                  if sharpe > perfect_sharpe:
                  
                      sharpe = perfect_sharpe/sharpe
                  else:
                      sharpe = sharpe/perfect_sharpe
              #sharpe1 = (portReturn - risklessRate) / portStdDev  #Sharpe ratio
              #if sharpe > sharpe_max:
                  #sharpe = sharpe_max
                  
                  score2 = score2 + sharpe
                  k = k + 1
                  windows = windows + 1
                  slope.append(sharpe)
              #kk = windows - 1
              # #np.arange(1,k+1,1)
              #kk = k+1
              #print(prices.index[-kk:])
              #print(X)
              #print(X)
              #kk = len(sharpe_cum) + 1
              X = np.arange(1,k+1,1)
              #X = prices.index[-kk:]
              #print(X)
              
              #slope1 = stats.theilslopes(slope, X, 0.90)[0] #stats.linregress(X,slope)[0]
              slope1 = stats.theilslopes(sharpe_cum, X, 0.90)[0] #stats.linregress(X,slope)[0]
              #slope1 = stats.theilslopes(port_price, X, 0.90)[0] #stats.linregress(X,slope)[0]
              print('slope',slope1)
              w = windows_forward - 10
              poly1d_fn = np.poly1d(slope1) 
              #plt.plot(X,slope, 'yo', X, poly1d_fn(X), '--k')
              plt.plot(X,slope, X, poly1d_fn(X), '--k',linewidth=0.4)
              plt.plot(X,sharpe_cum, linewidth=0.6,linestyle='--',color='black')
              plt.ylim((-1,5.5))
              
              
              #plt.grid(linestyle=':',linewidth='0.3', color='black')
              #plt.grid(which='major', linestyle='-', linewidth='0.3', color='black')
# Customize the minor grid
              #plt.grid(which='minor', linestyle=':', linewidth='0.3', color='green')
              plt.axvline(w,color='red',linewidth=0.4,linestyle='--')
              plt.axhline(perfect_sharpe,color='brown',linewidth=0.4,linestyle='--')
              plt.title('Slope = ' + str(round(slope1,4)))# + 'No: ', + str(i))
              #plt.title(round(slope1,4))
              plt.show()
              plt.plot(X,mean_price, linewidth=0.8,linestyle='--',color='black')
              plt.axvline(w,color='red',linewidth=0.4,linestyle='--')
              plt.title(str(stocks_name))# + 'No: ', + str(i))
              plt.show()
              score3 = score2 / windows_forward
          #score3 = score3 + (1+ slope1)**2
              print('Loop Sharpe ',score3)
          #score.append(score2)
          
              if score3 > perfect_sharpe:
                  
                  score3 = (perfect_sharpe/score3) + slope1
                  #score3 = perfect_sharpe/(score3 - slope1)
              else:
                  score3 = (score3/perfect_sharpe) + slope1 
                  #score3 = (score3+slope1)/perfect_sharpe 
          except:
              score3 = -1.0
              pass
              
      return score3, stocks_name

def sort_by_fitness(score, population):
    #print('S4',score)
    #a = len(score)
    #b = population.shape[1]
        score = np.array(score)
        #fitness = score[::-1].argsort()
        fitness = score.argsort()
        #fitness = fitness[::-1]
        fit = population[fitness]
    #print('fit',fit)
        return np.array(fit)
#############################################################################
filepath = 'C:/Users/Wai/Downloads/'
#def main_loop():

#returns = pd.read_csv(filepath+"sp500return.csv",encoding='latin1')  #SP500
#df = pd.read_csv(filepath+"crypto_ret_Oct21.csv") 
df = pd.read_csv("stocks_rets.csv",index_col=0)
prices = pd.read_csv("stocks.csv",index_col=0)
#df = pd.read_csv(filepath+"sp500return.csv")
#df = df.iloc[:160,:]
#df = pd.read_csv(filepath+"crypto_ret_hour.csv")
#price_data = pd.read_csv(filepath+"crypto_prices.csv")
#returns1 = np.log(price_data/price_data.shift(1))
#returns1 = price_data/price_data.shift(1)
#returns.head()
#df = df.dropna()
#df = np.log(df)
#df = df.drop(['BTCDOWNUSDT','ETHDOWNUSDT'],axis=1)
df['SPDN'] = -(df['SPDN'])
df1 = df.std()
drop_list = list(df1[df1 > 0.05].index) # 0.08
df = df.drop(drop_list,axis=1)
#spdn = df['SPDN']
#df = df.drop('SPDN',axis=1)
#df = df.drop(['THETAUSDT','BTCDOWNUSDT'],axis=1)
#df = df.drop(['THETAUSDT'],axis=1)

#returns1 = returns1.dropna()
returns = df.copy() #df[80:]
train = returns[:-50]
test = returns[-50:]

counter_size = train.shape[1]
window_size = train.shape[0]

start = time.time()

binary = [1,2,4,8,16,32,64,128,256,512]
binary_stock = [1,2,4,8,16,32]
#binary_stock = [1,2,4,8]
s_ceiling = 5.0
s_floor = 3.0
w_ceiling = 150.0 #90.0 
w_floor = 100.0 # 70.0  
sharpe_max = 4.0 #2.618
sharpe_min = 2.0 # 1.618 # 1.618
maxW = 2.0
minW = 1.0
minS = 2.0
maxS = counter_size - 1 # 33.0
risklessRate = 0.0 #0.03
annualize = 252
chromosome_length = 428 #418
population_size = 100
maximum_generation = 20 #250
best_score_progress = [] # Tracks progress
mutation_probability = 0.005
population = create_starting_population(population_size, chromosome_length)
    
if __name__ == '__main__':
    
    #with Pool(multiprocessing.cpu_count()) as pool:
    with Pool(4) as pool:
        
        for gen in range(maximum_generation):
  
            print('Gen: ',gen+1)
      #print(population)
      #scores = eval(population,population_size)
      #ii = [for i in range(population_size)]
      #scores = pool.map(eval,[row for row in population])
      #scores = [pool.apply(eval,args=(population,pop)) for pop in range(population_size)]
      
      
            scores = pool.starmap(eval,[(population,pop) for pop in range(population_size)])
            score1 = max(scores)
            
            print('Sharpe Ratio : ', score1)
            population_best = population
            
            new_population2 = pool.starmap(select_individual_by_tournament1,[(population, scores) for i in range(len(scores[:50]))])

            population = np.array(new_population2) # new_population2[:-1,:]
            population = np.reshape(population,(population_size,chromosome_length))
           
            
            population = randomly_mutate_population(population, mutation_probability)
           
    
    pool.close()
    
    df['SPDN'] = -(df['SPDN'])
    returns = df.copy() #df[80:]
    train = returns[:-50]
    test = returns[-50:]
    fittest = np.argmax(scores,axis=0)
    j = np.random.randint(2,99)
    #print('f',fittest)
    best = population_best[fittest]
    best2 = population_best[j]
    best1 = np.vstack((best2,best))
    
    
    #for i in range(2):
         #j = np.random.randint(2,99)
         #best2 = population_best[j]
         #best1 = np.vstack((best1,best2))
    
    for i in range(2):
        print('Final Loop ', i+1)
        scores1 = eval_test(best1,i)
        print('Loop Sharpe',score1)
        
       
    
    #best = population_best[-1]
    #print(best)
    win = get_windowsb(best)
    
    mean_return = train.tail(win).mean()
    
    cov = train.tail(win).cov()
    #cov = risk_models.CovarianceShrinkage(train.tail(win)).ledoit_wolf()
    
    # #(10,)
    #mean_return = returns.mean()
    #cov = returns.cov()
    ns = get_num_stockb(best)
    print('Num Of Stocks: ',ns + 1)
    print('Window: ',win)
    #cov = nls.shrink_cov(returns1)
    sl = get_stockb(best)
    #print('Stocks',sl)
    mr = mean_return.iloc[sl]
    a = cov.loc[:,cov.columns[sl]]
    covMatrix = (a.iloc[sl]) * annualize

    meanDailyReturns = mr * annualize
    print(np.exp(meanDailyReturns))
    
    sll = list(cov.columns[sl])
    print('Stocks: ',sll)
    w = get_weightsb(best)
    #print('w',w)
    w = np.array(w)
    sumW = np.sum(w)
    weights_list1 = np.asarray(w)/sumW
    print('Weights: ',weights_list1)
    #sumW = sum(map(lambda x: abs(x),w))
    
    

    portReturn = sum(meanDailyReturns.T*weights_list1)
        #portReturn = sum(portReturn1)
    portStdDev = np.sqrt(np.dot(weights_list1.T, np.dot(covMatrix, weights_list1)))

    sharpe = (portReturn - risklessRate) / portStdDev  #Sharpe ratio
    print('Expected Return: ',portReturn)
    print('Std Dev: ',portStdDev)
    print('Sharpe Ratio: ',sharpe)
    #sha = get_sharpeb(best),
    #print('Perfect',sha)
    
# Test
    print('Test ................................................................')
    #print('................................................................')
    window_size = returns.shape[0]
    
    for i in range(2):
        print('Test Loop..................................................... ', i+1)
        
        scores1,stocks_name = eval_test(best1,i)
        print('Test Sharpe:' ,score1)
    plt.rc('legend',**{'fontsize':7})
    #plt.rc('xlabel',**{'fontsize':8})
    #prices.loc[:,sll].plot(subplots=True,fontsize=7)
    prices.loc[:,stocks_name].plot(subplots=True,fontsize=7)
    #ax.legend(prop={'size':8})
    print('End: ', time.time() - start)
    #stocks_name = prices.columns[stocks_list]
    #stocks_name = list(stocks_name)
              #print('Stocks: ',stocks_name)
              #print('Weights: ',weights_list)
   
    

