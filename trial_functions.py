#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 18:36:05 2025

@author: william
"""

def vector_cube(cube) :
    """ returns some kind of vector representation of cube
    with each row = a face and each col = colour 1 or 0
    """
    flat = np.reshape(cube, (-1))
    out = np.zeros((len(flat),max(flat)+1))
    for i, f in enumerate(flat) :
        out[i, f] = 1
    return out

def matrix_dist(cube) :    
    cube0=np.zeros((6,3,3), dtype=int)
    for f in range(6):
        cube0[f] = f
    cube0 = vector_cube(cube0)
    cube1 = vector_cube(cube)
    dist = np.linalg.norm(cube1-cube0)
    return dist

def tensor_dist(cube) :
    cube0=np.zeros((6,3,3), dtype=int)
    for f in range(6):
        cube0[f] = f
    dist = tf.norm(tf.convert_to_tensor(cube - cube0, dtype=float))
    return dist.numpy()

def compress_moves(in_move) :
    """
    Helper function to remove redundant moves from a seq.
    e.g. B1, B1 -> B2
    e.g. B2, B2 -> null
    """
    in_move = list(in_move)
    out_move = []
    while len(in_move) > 0 :
        r = in_move.pop(0)
        face = r[0]
        rotates = int(r[1])
        while len(in_move) != 0 and face == in_move[0][0] :
            next_r = in_move.pop(0)
            rotates += int(next_r[1])

        if rotates % 4 != 0 :
            out_move.append(face + str(rotates % 4))

    return out_move