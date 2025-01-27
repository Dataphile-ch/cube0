#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:12:02 2025

@author: william

Implement the MCTS algorithm for cube solving.
#TO DO:
    Consider using num_visits to de-weight nodes that have been visited too often.

"""

#%% Packages
import numpy as np
from cube import Cube
from copy import deepcopy

#%% Functions

#%% Main

class TreeHorn :
    """
    Normally the main class is called TreeNode, but I couldn't resist...
    https://thebiglebowski.fandom.com/wiki/Jackie_Treehorn
    
    A round consists of 
    Select - get best unexpanded node using exploration vs exploitation
    Expand - find all child nodes, up to level k
    Simulate - get rewards for child states
    Backpropogation - propogate best reward to all parents

    State representation for Cube: the 6x3x3 matrix from Cube.cube

    Reward function for Cube: estimated distance from solved state.
    For the Cube, there is only 1 solved state so we don't cumulate rewards from different policies (as in some MCTS examples)
    
    """
    
    def __init__(self, state : Cube, parent=None, parent_action=None, iterations=100, explore_param=0.5) :
        self.state = state 
        self.parent = parent
        self.parent_action = parent_action
        self.iterations = iterations
        self.explore_param = explore_param

        self.reward = self.state.get_reward()
        self.best_reward = self.reward
        self.children = [] 
        self.num_visits = 1 # initialize to 1 to stop div0 errors
        self.possible_actions = self.state.get_possible_actions(parent_action)

        return

    def __str__(self) :
        d,c = self.traverse()
        r = self.best_reward
        s = f'Total children:\t {c:,}\n' + \
            f'Max depth:\t\t {d}\n' + \
                f'Max reward:\t\t {r:.3f}'
        return s
    
    def __repr__(self) :
        d,c = self.traverse()
        r = self.best_reward
        s = r == 1
        return f'[{s},{c},{d},{r:.3f}]'
    
    def traverse(self) :
        """
        go down the tree and collect statistics
        """
        childs = len(self.children)
        max_depth = 1
        for child_node in self.children :
            d, c = child_node.traverse()
            childs += c
            max_depth = max(d+1, max_depth)
        return max_depth, childs
        
    def softmax(self, X : np.array, theta=1) :
        X = X / theta
        return(np.exp(X - np.max(X)) / np.exp(X - np.max(X)).sum())

    def is_root_node(self):
        if self.parent is None :
            return True
        else:
            return False
    
    def is_terminal_node(self):
        return self.state.is_solved()
    
    def is_fully_expanded(self):
        return ( len(self.children) == len(self.possible_actions) )
    
    def best_child(self, explore_param=None):
        """
        Find a child node to explore, using exploitation vs exploration

        # explore paramter is either 0 (no exploration, just best reward) or used as Theta in softmax
        # low values of theta will generate less randomness (exploit vs explore)
        # higher values will select more random nodes (explore vs exploit)
        # No further parameters needed to control exploit vs explore !
        # TO DO: consider weighing by number of visits.
        # Testing : adjust the softmax temp to favour exploration 
        for cubes with higher entropy.
        """
        if not self.children :
            raise Exception("Attempt to find best child of unexpanded node")

        child_rewards = np.array( [(c.best_reward) for c in self.children] )

        if explore_param == 0 :
            # full exploit
            return self.children[np.argmax(child_rewards)]
        else :
            # explore
            if not explore_param : theta=self.explore_param        
#            entropy = self.state.estimate_distance()
#            theta = entropy/20
            weights = self.softmax(child_rewards,theta=theta)
            rand_child = np.random.choice(len(self.children), p=weights)
            return self.children[rand_child]
    
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
            next_state.move([a]) 
            child_node = TreeHorn(next_state, parent=self, parent_action=a, explore_param=self.explore_param)
            self.children.append(child_node)
        return True

    def tree_policy(self):
        """
        select the next node for rollout.  Search recursively down the tree using "best child".
        when we get to an unexpanded node, expand it and return.
        """
    
        current_node = self
        while current_node.is_fully_expanded() :  # go recursively down the tree
            current_node = current_node.best_child()

        current_node.expand()

        return current_node

    def backpropagate(self, reward):
        """
        Backpropogate the best reward from this node to parent(s)
                
        """
        self.num_visits += 1.
        if reward > self.best_reward :
            self.best_reward = reward
        if self.parent:
            self.parent.backpropagate(self.best_reward)

    def rollout(self):
        """
        This will normally simulate the game until it finds a solved state.
        However, for the cube problem, we will only simulate the next leve.
        i.e. we just need to get the reward for the best child.
        
        May be improved by searching down 2-3 layers for best reward, 
        if this can be done cheaply.
        """
        if self.is_terminal_node() or (not self.children):
            raise Exception('Attempt to rollout from solved cube')
            
        for c in self.children :
            c.best_reward = c.deep_rollout()

        best = self.best_child(explore_param=0.0)

        return best.reward

    def deep_rollout(self) :
        """
        Explore moves from current cube state and return best reward.
        TO DO: eliminate redundant rotations
        """
        start_state = deepcopy(self.state.cube)
        rollout_cube = Cube()
        possible_actions = rollout_cube.get_possible_actions(self.parent_action)
        all_actions = rollout_cube.get_possible_actions()
#        moves = [(r1,r2) for r1 in possible_actions for r2 in all_actions]
        moves = possible_actions

        for m in moves :
            rollout_cube.cube = start_state
            rollout_cube.move([m])
            reward = rollout_cube.get_reward()
            best_reward = max(self.best_reward, reward)
        return best_reward

    def mcts_search(self):
        """
        pick a node, find a possible reward, backpropogate.
        repeat n times, and then return the best child.
        """
        for i in range(self.iterations):
            v = self.tree_policy() # get a node to try
            reward = v.rollout() # possible reward from that node
            v.backpropagate(reward)
            if reward==1 :
                break
	
        return (reward==1)
