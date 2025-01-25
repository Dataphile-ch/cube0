#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:46:34 2025

@author: william
"""
#%% Setup
from cube import Cube
from mcts import TreeHorn

rubiks = Cube()
rubiks.move(rubiks.rand_move(5))
root = TreeHorn(rubiks)


#%% Functions

def look_path(path : TreeHorn) :
    while not path.is_terminal_node() :
        path = path.best_child(explore_param=0.0)
        print (path.best_reward, path.reward, path.parent_action, path.is_terminal_node())

    return

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
iterations = 5000
solved = root.v_search(iterations)
if solved :
    print('Cube solved!')
    print('Scramble moves: \t', root.state.moves)
    print('Reverse moves: \t', inverse_move(root.state.moves))
    solve_moves = []
    path = root
    while not path.is_terminal_node() :
        path = path.best_child(explore_param=0.0)
        solve_moves.append(path.parent_action)
    print('Solve moves: \t', solve_moves)
else :
    print('Not solved!')
    print('Scramble moves: \t', root.state.moves)


