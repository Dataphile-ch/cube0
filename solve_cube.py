#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:46:34 2025

@author: william
"""
#%% Setup
from cube import Cube
from mcts import TreeHorn
import time

#%% Functions

def inverse_move(moves) :
    rev_index = [3,2,1]
    inverse_rotates = []
    for m in moves :
        face = m[0]
        rotate = int(m[1])
        rev_rotate = rev_index[rotate-1]
        inverse_rotates.append(face+str(rev_rotate))
    inverse_rotates = inverse_rotates[::-1]
    return inverse_rotates


#%% Solve
iterations = 100,000
explore_param = 1
scrambles = 8

rubiks = Cube()
rubiks.move(rubiks.rand_move(scrambles))

root = TreeHorn(rubiks, iterations=iterations, explore_param=explore_param)

tic = time.time()
solved = root.mcts_search()
toc = time.time()
elapsed = toc-tic

if solved :
    print('Cube solved!')
else :
    print('Not solved!')
print('Scramble moves: \t', root.state.moves)
print('Reverse moves: \t', inverse_move(root.state.moves))
solve_moves = []
path = root
while path.is_fully_expanded() :
    path = path.best_child(explore_param=0.0)
    solve_moves.append(path.parent_action)
print('Solve moves: \t', solve_moves)
print(root)
print(f'Working time: \t {int(elapsed/60):02d}:{int(elapsed%60):02d}')

