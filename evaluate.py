#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 18:19:36 2025

@author: william

Make a lot of sample cubes and evaluate solving performance.
"""
#%% Setup
from cube import Cube
from mcts import TreeHorn
from ast import literal_eval
from csv import writer
from tqdm import tqdm

#%% Functions


#%% Main
iterations = 100000 # iterations per solve attempt
explore_param = 0.1
scrambles = range(5,15+1)
samples = 50 # how many samples at each scramble level

results = []
for s in scrambles :
    print(f'Level: {s}')
    for n in tqdm(range(samples)) :
        rubiks = Cube()
        rubiks.move(rubiks.rand_move(s))
        root = TreeHorn(rubiks, iterations=iterations, explore_param=explore_param)
        solved = root.mcts_search()
        result = literal_eval(repr(root))
        result.append(s)
        results.append(result)

headings = ['Success', 'Nodes', 'Depth', 'MaxReward', 'Scrambles']
with open('evaluate.csv', 'w') as f:
    write = writer(f)
    write.writerow(headings)
    write.writerows(results)


