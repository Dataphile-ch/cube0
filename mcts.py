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
from random import randint
from copy import deepcopy

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

    State representation for Cube: the 6x3x3 matrix from Cube.cube

    Reward function for Cube: estimated number of moves from solved state.
    For the Cube, there is only 1 solved state so we don't cumulate rewards from different policies (as in some MCTS examples)
    
    """
    
    def __init__(self, state : Cube, parent=None, parent_action=None) :
        self.state = state 
        self.parent = parent
        self.parent_action = parent_action
        self.reward = 20
        self.best_reward = 20 # I think I need this ...
        self.best_action = None
        self.children = [] # should be node + action
        self._number_of_visits = 0
        
        # Some cube-specific stuff.  Probably should be in a separate class...
        self._untried_actions = None
        self._untried_actions = self.get_possible_actions(parent_action)

        return

    def get_possible_actions(self, parent_action=[]) :
        """
        Return possible next rotations, without repeating a rotation on the same face
        as the parent.
        """
        if parent_action :
            last_face = parent_action[0]
            possible_actions = []
            for v in self.state.valid_rotates :
                if v[0] != last_face :
                    possible_actions.append(v)
            return possible_actions
        else :
            return deepcopy(self.state.valid_rotates)
    
    def is_root_node(self):
        if self.parent is None :
            return True
        else:
            return False
    
    def is_terminal_node(self):
        return (self.state.cube == self.state.solved_cube).all()
    
    def expand(self):
        """
        Get an untried action, move to new state and return new child node
        """
        rand_action_idx = randint(0, len(self._untried_actions) - 1)
        action = self._untried_actions.pop(rand_action_idx)
        next_state = deepcopy(self.state)
        next_state.rotate(action) 
        child_node = TreeHorn(next_state, parent=self, parent_action=action)
        
        self.children.append(child_node)
        return child_node         

    def backpropagate(self, action, reward):
        """
        Backpropogate the best reward from this node to parent(s)
        And the action that got us here, so we can come back if interested
                
        """
        self._number_of_visits += 1.
        if reward < self.best_reward :
            self.best_reward = reward
            self.best_action = action
        if self.parent:
            self.parent.backpropagate(self.best_reward, self.parent_action)
            
