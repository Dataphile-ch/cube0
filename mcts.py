#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:12:02 2025

@author: william

Implement the MCTS algorithm for cube solving.

"""

#%% Packages
import numpy as np
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
        self.reward = self.state.get_reward()
        self.best_reward = 20 # I think I need this ...
        self.best_action = None
        self.children = [] # should be node + action
        self.num_visits = 1 # initialize to 1 to stop div0 errors
        
        # Some cube-specific stuff.  Probably should be in a separate class...
        self.possible_actions = self.state.get_possible_actions(parent_action)

        return
    
    def is_root_node(self):
        if self.parent is None :
            return True
        else:
            return False
    
    def is_terminal_node(self):
        return (self.state.cube == self.state.solved_cube).all()
    
    def is_fully_expanded(self):
        return ( len(self.children) == len(self.possible_actions) )
    
    def best_child(self, explore_param=0.1):
        """
        Find a child node to explore, using exploitation vs exploration
        """
        if not self.children :
            raise Exception("Attempt to find children before expanding")
        
#        choices_weights = [((20-c.reward) / c.num_visits) + \
#                            explore_param * np.sqrt(2 * np.log(self.num_visits / c.num_visits)) \
#                                                    for c in self.children]
        choices_weights = [(20-c.reward) for c in self.children]
        
        return self.children[np.argmax(choices_weights)]
    
    def expand(self):
        """
        Create child nodes with new state from all possible actions.
        """
        if self.is_fully_expanded() :
            raise Exception("Attempt to expand fully-expanded node")
        if self.is_terminal_node() :
            raise Exception("Attempt to expand a solved cube")

        for a in self.possible_actions :
            next_state = deepcopy(self.state)
            next_state.move([a]) # ??
            child_node = TreeHorn(next_state, parent=self, parent_action=a)
            self.children.append(child_node)
        return True

    def tree_policy(self):
        """
        select the next node for rollout.  Search recursively down the tree using "best child".
        when we get to an unexpanded node, expand it and return.
        NB. Doesn't return best child if this node is terminal ?!
        """
    
        current_node = self
        while current_node.is_fully_expanded() :  # go recursively down the tree
            current_node = current_node.best_child()

        if not current_node.is_terminal_node() :
            current_node.expand()

        return current_node

    def backpropagate(self, reward, action):
        """
        Backpropogate the best reward from this node to parent(s)
        And the action that got us here, so we can come back if interested
                
        """
        self.num_visits += 1.
        if reward < self.best_reward :
            self.best_reward = reward
            self.best_action = action
        if self.parent:
            self.parent.backpropagate(self.best_reward, self.parent_action)

    def rollout(self, max_depth=3):
        """
        This will normally simulate the game until it finds a solved state.
        However, for the cube problem, we will only simulate the next leve.
        i.e. we just need to get the reward and action for the best child.
        
        May be improved by searching down 2-3 layers for best reward, 
        if this can be done cheaply.
        """
        
        best = self.best_child(explore_param=0.0)
        
        return best.reward, best.parent_action
            
    def v_search(self, trials = 100):
        """
        pick a node, find a possible reward, backpropogate.
        repeat n times, and then return the best child.
        PROBLEM: we never recursively search the best child.
        """
        for i in range(trials):
            v = self.tree_policy() # get a node to try
            reward, action = v.rollout() # possible reward from that node
            v.backpropagate(reward, action)
            if reward == 0 :
                break
	
        return self.best_child(explore_param=0.0)
