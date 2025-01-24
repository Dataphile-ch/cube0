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
rubiks.move(rubiks.rand_move(3))
root = TreeHorn(rubiks)

#%% Solve
best_child = root.v_search(50)
if root.best_reward == 0 :
    print('Cube solved!')
    print('Scramble moves: \t', root.state.moves)
    solve_moves = []
    path = root
    while not path.is_terminal_node() :
        path = path.best_child()
        solve_moves.append(path.parent_action)
    print('Solve moves: \t', solve_moves)


