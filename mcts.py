#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:12:02 2025

@author: william

Implement the MCTS algorithm for cube solving.

"""

#%% Packages
import numpy as np
from collections import defaultdict
from cube import Cube

#%% Functions

#%% Main

class TreeHorn :
    """
    Normally the main class is called TreeNode, but I couldn't resist...
    https://thebiglebowski.fandom.com/wiki/Jackie_Treehorn
    
    A round consists of 
    Select - get best child using exploration vs exploitation
    Expand - find all child nodes, up to level k
    Simulate - get rewards for child states
    Backpropogation - propogate best reward to all parents

    Reward function for the cube: estimated number of moves from solved state.
    For the Cube, there is only 1 solved state so we don't cumulate rewards from different policies (as in some MCTS examples)
    
    """
    
    def __init__(self, state, parent=None, parent_action=None) :
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        
        # Some cube-specific stuff.  Probably should be in a separate class...
        self.valid_rotates = ['R1', 'R2', 'R3', 'L1', 'L2', 'L3', \
                            'U1', 'U2', 'U3', 'D1', 'D2', 'D3', \
                            'F1', 'F2', 'F3', 'B1', 'B2', 'B3']


        return

    def get_possible_actions(self, parent_action) :
        """
        Return possible next rotations, without repeating a rotation on the same face
        as the parent.
        """
        last_face = parent_action[0]
        possible_actions = []
        for v in self.valid_rotates :
            if v[0] != last_face :
                possible_actions.append(v)
        return possible_actions

    def untried_actions(self) :
        return