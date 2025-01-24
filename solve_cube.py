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
rubiks.move(rubiks.rand_move(2))
root = TreeHorn(rubiks)

#%% Solve
best_child = root.v_search()
if root.best_reward == 0 :
    print('Cube solved!')
