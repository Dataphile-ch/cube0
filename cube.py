# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import tensorflow.keras as keras

class Cube :
    """
    Terminology:
        Rotate: rotation of a single face/layer by 90, 180 or 270 degrees.
        Move: sequence of rotations from one state to another.
        Entropy: measure of how scrambled the cube is.  
            Ideally this would be the shortest number of rotations away from the solved state.
        Cublet: one of the 8 corners, 12 edges, 6 centres.  Lower case notation.
        Cubicle: one of the locations that can hold a Cubelet.  Upper case notation.
    """
    
    def __init__(self) :
        # map colours to faces on the cube
        # cube is oriented white up, green front
        # example:
        # u_colour = colours[faces.index('U')]

        self.colours = ['white', 'orange', 'green', 'red', 'blue', 'yellow']
        self.faces = ['U', 'L', 'F', 'R', 'B', 'D']
            
        # The cube is represented as 6 faces with 3x3 stickers or facelets
        self.solved_cube=np.zeros((6,3,3), dtype=int)
        # initialize the colours on each face
        for f in range(6):
            self.solved_cube[f] = f
        
        self.cube = np.copy(self.solved_cube)

        self.entropy = 0
        self.valid_rotates = ['R1', 'R2', 'R3', 'L1', 'L2', 'L3', \
                            'U1', 'U2', 'U3', 'D1', 'D2', 'D3', \
                            'F1', 'F2', 'F3', 'B1', 'B2', 'B3']
        self.inverse_rotates = ['R3', 'R2', 'R1', 'L3', 'L2', 'L1', \
                            'U3', 'U2', 'U1', 'D3', 'D2', 'D1', \
                            'F3', 'F2', 'F1', 'B3', 'B2', 'B1']
        self.moves = []
        # to  get reverse moves, use: self.moves[::-1]
        
        self.pred_model = keras.models.load_model("nn_entropy.keras")
        
    def reset(self) :
        # initialize the colours on each face
        self.cube = np.copy(self.solved_cube)
        self.entropy = 0

    def is_solved(self) :
        return (self.cube == self.solved_cube).all()
        
    def vector_cube(self) :
        """ returns some kind of vector representation of cube
        with each row = a face and each col = colour 1 or 0
        """
        out = self.cube.flatten()
        return out

    def matrix_dist(self) :    
        cube0 = self.solved_cube.flatten()
        cube1 = self.vector_cube()
        dist = np.linalg.norm(cube1-cube0)
        return dist

    def nn_entropy(self) :
        pred = self.pred_model.predict(np.reshape(self.cube, (1,6,3,3)))
        return pred.argmax()

    def naive_entropy(self) :
        # For now, this is just counting the number of misplaced or misoriented cubelets.
        entropy = (self.cube != self.solved_cube).sum()
        return entropy
                    
    def align_entropy(self) :
        # A measure based on alignment of corners and edges, which is always a step towards solved state.
        max_entropy = 8*4
        align = 0
                
        UFL = (self.cube[0,2,0], self.cube[2,0,0], self.cube[1,0,2]) 
        URF = (self.cube[0,2,2], self.cube[3,0,0], self.cube[2,0,2]) 
        ULB = (self.cube[0,0,0], self.cube[1,0,0], self.cube[4,0,2])
        UBR = (self.cube[0,0,2], self.cube[4,0,0], self.cube[3,0,2])
        DLF = (self.cube[5,0,0], self.cube[1,2,2], self.cube[2,2,0])
        DFR = (self.cube[5,0,2], self.cube[2,2,2], self.cube[3,2,0])
        DLB = (self.cube[5,2,0], self.cube[1,2,0], self.cube[4,2,2])
        DBR = (self.cube[5,2,2], self.cube[4,2,0], self.cube[3,2,2])
        
        UF = (self.cube[0,2,1], self.cube[2,0,1]) 
        UL = (self.cube[0,1,0], self.cube[1,0,1]) 
        UB = (self.cube[0,0,1], self.cube[4,0,1]) 
        UR = (self.cube[0,1,2], self.cube[3,0,1]) 

        LF = (self.cube[1,1,2], self.cube[2,1,0]) 
        LB = (self.cube[1,1,0], self.cube[4,1,2]) 
        RF = (self.cube[3,1,0], self.cube[2,1,2]) 
        BR = (self.cube[4,1,0], self.cube[3,1,2]) 

        DF = (self.cube[5,0,1], self.cube[2,2,1]) 
        DL = (self.cube[5,1,0], self.cube[1,2,1]) 
        DB = (self.cube[5,2,1], self.cube[4,2,1]) 
        DR = (self.cube[5,1,2], self.cube[3,2,1]) 

        U = self.cube[0,1,1]
        L = self.cube[1,1,1]
        F = self.cube[2,1,1]
        R = self.cube[3,1,1]
        B = self.cube[4,1,1]
        D = self.cube[5,1,1]
        
        # UFL should align with UF, LF, UL
        # UFL should align with U, F, L
        if UFL[0] == UF[0] and UFL[1] == UF[1] :
            align += 1
        if UFL[2] == LF[0] and UFL[1] == LF[1] :
            align += 1
        if UFL[0] == UL[0] and UFL[2] == UL[1] :
            align += 1
        if UFL[0] == U and UFL[1] == F :
            align += 1
        
        # URF should align with UR, RF, UF
        # URF should align with U, R, L
        if URF[0] == UR[0] and URF[1] == UR[1] :
            align += 1
        if URF[1] == RF[0] and URF[2] == RF[1] :
            align += 1
        if URF[0] == UF[0] and URF[2] == UF[1] :
            align += 1
        if URF[0] == U and URF[1] == R :
            align += 1
        
        # ULB should align with UL, LB, UB
        # ULB should align with U, L, B
        if ULB[0] == UL[0] and ULB[1] == UL[1] :
            align += 1
        if ULB[1] == LB[0] and ULB[2] == LB[1] :
            align += 1
        if ULB[0] == UB[0] and ULB[2] == UB[1] :
            align += 1
        if ULB[0] == U and ULB[1] == L :
            align += 1
        
        # UBR should align with UB, BR, UR
        # UBR should align with U, B, R
        if UBR[0] == UB[0] and UBR[1] == UB[1] :
            align += 1
        if UBR[1] == BR[0] and UBR[2] == BR[1] :
            align += 1
        if UBR[0] == UR[0] and UBR[2] == UR[1] :
            align += 1
        if UBR[0] == U and UBR[1] == B :
            align += 1
        
        # DLF should align with DL, LF, DF
        if DLF[0] == DL[0] and DLF[1] == DL[1] :
            align += 1
        if DLF[1] == LF[0] and DLF[2] == LF[1] :
            align += 1
        if DLF[0] == DF[0] and DLF[2] == DF[1] :
            align += 1
        if DLF[0] == D and DLF[1] == L :
            align += 1
        
        # DFR should align with DF, RF, DR
        if DFR[0] == DF[0] and DFR[1] == DF[1] :
            align += 1
        if DFR[1] == RF[1] and DFR[2] == RF[0] :
            align += 1
        if DFR[0] == DR[0] and DFR[2] == DR[1] :
            align += 1
        if DFR[0] == D and DFR[1] == F :
            align += 1
        
        # DLB should align with DL, LB, DB
        if DLB[0] == DL[0] and DLB[1] == DL[1] :
            align += 1
        if DLB[1] == LB[0] and DLB[2] == LB[1] :
            align += 1
        if DLB[0] == DB[0] and DLB[2] == DB[1] :
            align += 1
        if DLB[0] == D and DLB[1] == L :
            align += 1
        
        # DBR should align with DB, BR, DR
        if DBR[0] == DB[0] and DBR[1] == DB[1] :
            align += 1
        if DBR[1] == BR[0] and DBR[2] == BR[1] :
            align += 1
        if DBR[0] == DR[0] and DBR[2] == DR[1] :
            align += 1
        if DBR[0] == D and DBR[1] == B :
            align += 1
        
        return max_entropy - align
    
    def update_entropy(self, style='naive') :
        if style == 'off' :
            pass
        elif style == 'align' :
            self.entropy = self.align_entropy() 
        elif style == 'naive' :
            self.entropy = self.naive_entropy()
        elif style == 'nn' :
            self.entropy = self.nn_entropy()
        elif style == 'matrix' :
            self.entropy = self.matrix_dist()
        else :
            print("Invalid Entropy Style")
        return self.entropy
        
    def rotate(self, r, level=1) :
        
        def rotate_face(f) :
                save = np.copy(self.cube[f,0,:])
                self.cube[f,0,:] = np.flip(self.cube[f,:,0])
                self.cube[f,:,0] = self.cube[f,2,:]
                self.cube[f,2,:] = np.flip(self.cube[f,:,2])
                self.cube[f,:,2] = save

        match r :
            case 'R1' :
                rotate_face(3)
                save = np.copy(self.cube[0,:,2])
                self.cube[0,:,2] = self.cube[2,:,2]
                self.cube[2,:,2] = self.cube[5,:,2]                
                self.cube[5,:,2] = np.flip(self.cube[4,:,0])
                self.cube[4,:,0] = np.flip(save)

            case 'R2' :
                self.rotate('R1', level+1)
                self.rotate('R1', level+1)
            case 'R3' :
                self.rotate('R1', level+1)
                self.rotate('R1', level+1)
                self.rotate('R1', level+1)
                
            case 'L1' :
                rotate_face(1)
                save = np.copy(self.cube[0,:,0])
                self.cube[0,:,0] = np.flip(self.cube[4,:,2])
                self.cube[4,:,2] = np.flip(self.cube[5,:,0])
                self.cube[5,:,0] = self.cube[2,:,0]
                self.cube[2,:,0] = save
            case 'L2' :
                self.rotate('L1', level+1)
                self.rotate('L1', level+1)
            case 'L3' :
                self.rotate('L1', level+1)
                self.rotate('L1', level+1)
                self.rotate('L1', level+1)

            case 'U1' :
                rotate_face(0)
                save = np.copy(self.cube[1,0,:])
                self.cube[1,0,:] = self.cube[2,0,:]
                self.cube[2,0,:] = self.cube[3,0,:]
                self.cube[3,0,:] = self.cube[4,0,:]
                self.cube[4,0,:] = save
            case 'U2' :
                self.rotate('U1', level+1)
                self.rotate('U1', level+1)
            case 'U3' :
                self.rotate('U1', level+1)
                self.rotate('U1', level+1)
                self.rotate('U1', level+1)
                
            case 'D1' :
                rotate_face(5)
                save = np.copy(self.cube[2,2,:])
                self.cube[2,2,:] = self.cube[1,2,:]
                self.cube[1,2,:] = self.cube[4,2,:]
                self.cube[4,2,:] = self.cube[3,2,:]
                self.cube[3,2,:] = save
            case 'D2' :
                self.rotate('D1', level+1)
                self.rotate('D1', level+1)
            case 'D3' :
                self.rotate('D1', level+1)
                self.rotate('D1', level+1)
                self.rotate('D1', level+1)
                
            case 'F1' :
                rotate_face(2)
                save = np.copy(self.cube[0,2,:])
                self.cube[0,2,:] = np.flip(self.cube[1,:,2])
                self.cube[1,:,2] = self.cube[5,0,:]
                self.cube[5,0,:] = np.flip(self.cube[3,:,0])
                self.cube[3,:,0] = save
            case 'F2' :
                self.rotate('F1', level+1)
                self.rotate('F1', level+1)
            case 'F3' :
                self.rotate('F1', level+1)
                self.rotate('F1', level+1)
                self.rotate('F1', level+1)
                
            case 'B1' :
                rotate_face(4)
                save = np.copy(self.cube[0,0,:])
                self.cube[0,0,:] = self.cube[3,:,2]
                self.cube[3,:,2] = np.flip(self.cube[5,2,:])
                self.cube[5,2,:] = self.cube[1,:,0]
                self.cube[1,:,0] = np.flip(save)
                
            case 'B2' :
                self.rotate('B1', level+1)
                self.rotate('B1', level+1)
            case 'B3' :
                self.rotate('B1', level+1)
                self.rotate('B1', level+1)
                self.rotate('B1', level+1)
                
            case _ :
                print(f"Invalid rotation {r}")
                return 1
            
        if level ==1 :
            self.moves.append(r)

    def move(self, m) :
        # m is an array of valid rotations
        for r in m :
            self.rotate(r)
        self.update_entropy()

    def rand_move(self, k) :
        """
        Generate a move with rotations to depth k
        """
        
        move = []
        while len(move) < k :
            if len(move) > 0 :
                # avoid valid rotates on the same face to stop redundancy
                last_r = move[-1]
                next_valids = []
                for v in self.valid_rotates :
                    if v[0] != last_r[0] :
                        next_valids.append(v)
            else :
                next_valids = self.valid_rotates
        
            move.append(np.random.choice(next_valids).tolist())
        

        return move
    
    def get_reward(self) :
        """
        Returns: estimate of the distance from this state to solved state.
        i.e. number of rotations needed
        -------
        To do: estimate reward from different entropy functions.
        Maybe combine using weighted voting.
        Or : implement as NN
        
        Initial implementation is just a dumb re-scaling of naive entropy score.
        """
        e = self.update_entropy(style='naive')
        if e == 0 :
            reward = e
        else:
            reward = (e-9)*20/54
        return reward

        
    def show_cube(self) :
    
        # NB fliping around the x-axis to plot with origin in top-left
        U = np.flip(self.cube[0], axis=0)
        L = np.flip(self.cube[1], axis=0)
        F = np.flip(self.cube[2], axis=0)
        R = np.flip(self.cube[3], axis=0)
        B = np.flip(self.cube[4], axis=0)
        D = np.flip(self.cube[5], axis=0)
        cmap = colors.ListedColormap(self.colours)
        bounds = [-0.5, 0.5,1.5,2.5,3.5,4.5,5.5] 
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        
        fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(800*px, 600*px), subplot_kw={'xticks': [], 'yticks': []})
    
        axs[0,0].axis("off")
        axs[0,2].axis("off")
        axs[0,3].axis("off")
        axs[2,0].axis("off")
        axs[2,2].axis("off")
        axs[2,3].axis("off")
        
        axs[0,1].pcolor(U, cmap=cmap, norm=norm, edgecolor='k', linewidth=2)
        axs[1,0].pcolor(L, cmap=cmap, norm=norm, edgecolor='k', linewidth=2)
        axs[1,1].pcolor(F, cmap=cmap, norm=norm, edgecolor='k', linewidth=2)
        axs[1,2].pcolor(R, cmap=cmap, norm=norm, edgecolor='k', linewidth=2)
        axs[1,3].pcolor(B, cmap=cmap, norm=norm, edgecolor='k', linewidth=2)
        axs[2,1].pcolor(D, cmap=cmap, norm=norm, edgecolor='k', linewidth=2)
        plt.tight_layout(pad=0.5)
