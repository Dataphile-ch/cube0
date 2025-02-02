#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 18:19:36 2025

@author: william

Make a lot of sample cubes and evaluate solving performance.
"""
#%% Setup
from cube import Cube
import mcts
from ast import literal_eval
from csv import writer
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import time

#%% Functions

def plot_it() :
    
    fig, ax = plt.subplots()
    
    eval = pd.read_csv('evaluate.csv')
    bars = eval.groupby(by='Scrambles').mean()
    
    ax = bars.plot.bar(y='Success')

    plt.show()
    
    return True

#%% Main

def main() :
    iterations = 100000 # iterations per solve attempt
    explore_param = 0.05
    scrambles = range(4,4+1)
    
    samples = 10 # how many samples at each scramble level
    
    results = []
    for s in scrambles :
        print(f'Level: {s}')
        success = 0
        for n in tqdm(range(samples)) :
            rubiks = Cube()
            rubiks.move(rubiks.rand_move(s))
            root = mcts.TreeHorn(rubiks)
            tic = time.time()
            solved = mcts.mcts_search(root, iterations=iterations, explore_param=explore_param)
            toc = time.time()
            elapsed = toc-tic
            if solved : success += 1
            result = literal_eval(repr(root))
            result.append(s)
            results.append(result)
            results.append(int(elapsed))
        print(f'\nLevel: {s}, success rate: {success/samples:.1%}')
    
    headings = ['Success', 'Nodes', 'Depth', 'MaxReward', 'Scrambles', 'Elapsed']
    with open('evaluate.csv', 'w') as f:
        write = writer(f)
        write.writerow(headings)
        write.writerows(results)
    
    return True

if __name__ == '__main__'  :
    main()
    