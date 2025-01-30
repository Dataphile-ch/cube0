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
import matplotlib.pyplot as plt
import pandas as pd

#%% Functions


#%% Main
iterations = 100000 # iterations per solve attempt
explore_param = 0.05
scrambles = range(9,12+1)

samples = 10 # how many samples at each scramble level

results = []
for s in scrambles :
    print(f'Level: {s}')
    success = 0
    for n in tqdm(range(samples)) :
        rubiks = Cube()
        rubiks.move(rubiks.rand_move(s))
        root = TreeHorn(rubiks, iterations=iterations, explore_param=explore_param)
        solved = root.mcts_search()
        if solved : success += 1
        result = literal_eval(repr(root))
        result.append(s)
        results.append(result)
    print(f'Level: {s}, success rate: {success/samples:.1%}\n')

headings = ['Success', 'Nodes', 'Depth', 'MaxReward', 'Scrambles']
with open('evaluate.csv', 'w') as f:
    write = writer(f)
    write.writerow(headings)
    write.writerows(results)

#%% Plotting

fig, ax = plt.subplots()

eval = pd.read_csv('evaluate.csv')
bars = eval.groupby(by='Scrambles').mean()

ax = bars.plot.bar(y='Success')
