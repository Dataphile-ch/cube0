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
        self.num_visits = 0
        
        # Some cube-specific stuff.  Probably should be in a separate class...
        self._untried_actions = None
        self._untried_actions = self.state.get_possible_actions(parent_action)

        return
    
    def is_root_node(self):
        if self.parent is None :
            return True
        else:
            return False
    
    def is_terminal_node(self):
        return (self.state.cube == self.state.solved_cube).all()
    
    def is_fully_expanded(self):
        return len(self._untried_actions) == 0
    
    def best_child(self, explore_param=0.1):
        """
        Find a child node to explore, using exploitation vs exploration
        """
        if not self.children :
            raise Exception("Attempt to find children before expanding")
        
        choices_weights = [(c.reward / c.num_visits) + \
                            explore_param * np.sqrt(2 * np.log(self.num_visits / c.num_visits)) \
                                                    for c in self.children]
        return self.children[np.argmax(choices_weights)]
    
    def expand(self):
        """
        Get an untried action, create child node with new state from that action.
        """
        if len(self._untried_actions) == 0 :
            raise Exception("Attempt to expand fully-expanded node")
        rand_action_idx = randint(0, len(self._untried_actions) - 1)
        action = self._untried_actions.pop(rand_action_idx)
        next_state = deepcopy(self.state)
        next_state.move([action]) 
        child_node = TreeHorn(next_state, parent=self, parent_action=action)
        
        self.children.append(child_node)
        return child_node         

    def tree_policy(self):
        """
        select the next node for rollout.  Either an untried node, or the best child
        from the current node.
        NB. Doesn't return best child if that node is terminal ?!
        """
    
        current_node = self
        while not current_node.is_terminal_node():
            
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
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
        This does the business of searching down a random path from the current state.
        Stop search at max_depth because we are almost certain not to find a solved cube
        PROBLEM : 1 random search is unlikely to find the right path
        """
        
        current_rollout_state = deepcopy(self.state) 
        depth = 0
        first_action = []
        while not current_rollout_state.is_solved() and depth < max_depth:
            
            possible_moves = current_rollout_state.get_possible_actions()
            
            rand_action_idx = randint(0, len(possible_moves) - 1)
            action = possible_moves[rand_action_idx]
            if depth == 0 :
                first_action = action

            current_rollout_state.move([action])
            depth += 1
        return (current_rollout_state.get_reward(), first_action)
    
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
	
        return self.best_child(explore_param=0.)
