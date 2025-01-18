# -*- coding: utf-8 -*-

from cube import Cube
from copy import deepcopy
import random

edge_triangle = ['R1', 'U3', 'R1', 'U1', 'R1', 'U1', 'R1', 'U3', 'R3', 'U3', 'R2']

def best_move(cube, level=1, max_depth=1) :
    """
    Parameters
    ----------
    Cube : TYPE
        DESCRIPTION.
    moves : algorithm to get to the current iteration
    level : int, current recursion level
    max_depth : int, max recursions or iterations

    Returns
    -------
    List of moves to best solution

    fn: best_move -> find the best move by searching to max_depth from current state
    fn: solve_cube -> find best move and do it, iterate up to max_moves or until solved


    """
    cube0 = deepcopy(cube)
    max_entropy = 48
    best_r = ''
    entropy_list = []

#    for r in random.sample(cube0.valid_rotates, 9) :
    for (i,r) in enumerate(cube0.valid_rotates) :
        cube0.rotate(r)
        if cube0.entropy == 0 :
            cube0.rotate(cube0.inverse_rotates[cube0.valid_rotates.index(r)])
            return 0, r
        
        if level < max_depth :
            r_entropy, next_r = best_move(cube0, level=level+1, max_depth=max_depth)
        else :
            r_entropy = cube0.entropy + level 
            
        cube0.rotate(cube0.inverse_rotates[cube0.valid_rotates.index(r)])
    
        if r_entropy < max_entropy :
            max_entropy = r_entropy
            best_r = r
            
        entropy_list.append(r_entropy)

    ## if there are several rotates with the minimum reward, pick one at random
    r_list = []
    for (i, e) in enumerate(entropy_list) :
        if e == min(entropy_list) :
            r_list.append(cube.valid_rotates[i])
    best_r = random.choice(r_list)

    return max_entropy, best_r


def solve_cube(cube, max_depth=1, max_moves=1) :
    """
    fn: solve_cube -> find best move and do it, iterate up to max_moves or until solved
    """
    moves=[] 
    for m in range(max_moves) :
        r_entropy, best_r = best_move(cube, max_depth=max_depth)
        moves.append(best_r)
        cube.rotate(best_r)
        if cube.entropy == 0 :
            return cube.entropy, moves
    return cube.entropy, moves


def test_inverses(cube) :
    for r in cube.valid_rotates :
        cube.rotate(r)
        cube.rotate(cube.inverse_rotates[cube.valid_rotates.index(r)])

### Main
# CONST:
rand_moves=5
max_depth=3
max_moves=16

# Instantiate:
mycube = Cube()
mycube.reset()

scramble = mycube.compress_moves(mycube.rand_move(rand_moves))
mycube.move(scramble)
rev = mycube.reverse_moves
mycube.show_cube()
print("Scramble Moves: ", scramble)
print(" Reverse Moves: ", rev)

solve_entropy, solve_moves = solve_cube(mycube, max_depth=max_depth, max_moves=max_moves)
mycube.show_cube()
print("   Solve Moves: ", mycube.compress_moves(solve_moves))
