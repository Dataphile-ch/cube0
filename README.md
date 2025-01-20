# Cube Solver
Explore empirical methods for solving the 3x3x3 Rubik's Cube.

# Why?
The cube can be scrambled in a few moves.  The number called "God's number" for the cube is proved to be 20 https://cube20.org/#:~:text=At%20long%20last%2C%20God's%20Number,moves%20suffice%20for%20all%20positions.
So, no cube can be more that 20 moves from the solved state.

Current solving strategies rely on pattern recognition, step-by-step solving, and memorizing (or programming) up to 100 "algorithms" to move from 1 step to the next.  The most common strategy in speed cubing is CFOP https://jperm.net/3x3/cfop
These strategies typically take at least 40-50 moves to solve a cube, at least 2x as many as necessary.

> Is it possible to quickly find a minimal set of moves to solve any cube?
# Concepts
The Cube() class creates an object with an internal representation (6x3x3 matrix, but it's not very important for most purposes).  It allows the cube to be manipulated and for various attributes to be calculated.  Some methods are obvious, e.g. reset().  Core concepts are:
* Face: A cube has 6 faces: Right, Left, Up, Down, Front, Back [R,L,U,D,F,B]
* Sticker: each face has 3x3=9 squares or stickers.
* Cubelet: what actually gets moved are the cubelets, either an edge cubelet (with 2 stickers) or a corner cubelet (with 3 stickers)
* Rotation: Each face can be rotated 1, 2, or 3 times.  4 times returns to the original position, so is a non-move.
* Move: A move is a sequence of rotations.  e.g. R2, U1, D3
# Solving Method
The cube is solved using the MCTS algorithm (Monte-Carlo Tree Search).  This is the same basic algorithm used for most game solvers.
Because the space of possible moves is so vast, the algorithm has to be optimized - it can't try all possible paths until it finds the solved state.

The approach is to randomly explore possible moves down to a pre-defined depth, and then back-propogate a reward function.  Then the best move (or a random good move) is selected and we continue exploring until (hopefully) a solved state is found.
# Reward Function
The reward function is implemented inside the Cube() class as "entropy".  This term is almost certainly mis-used since all states of the cube are equally unlikely, but it is intended to represent "how far is this cube from the solved state".

Several definitions of cube entropy are evaluated:
* Naive entropy: count the number of "stickers" that are on the wrong face.
* Alignment entropy: based on the observation that corner and edge cubelets need to be aligned before they can be moved into place, how many alignments are there in the cube?
* Matrix entropy: use a linear algebra distance norm to calculate the distance between the current cube and the solved cube in matrix/vector form.
* Neural Network entropy: train a neural network on different cubes at different numbers of moves from solved.  The NN can then try to fit any given cube and estimate the number of moves to solve.

# Some Thoughts
Is it actually possible to estimate entropy (the distance from a given cube to solved state)?

Perhaps there is insufficient information in the cube to determine it's state.  What would this mean?  If you try to solve a cube by examining and un-doing moves, you can find the solution for up to 4-5 rotations, but beyond that it becomes very difficult to sed the reverse moves.  Does this mean that the cube really does have something like entropy and tends to a random state where information about the original structure (solved state) is no longer available??

# To do
* Re-implement MCTS with approximate policy iteration
* Re-do matrix distance entropy
* Collect performance statistics on different strategies
* Re-do NN training

# References
https://paperswithcode.com/paper/solving-the-rubiks-cube-with-approximate
https://courses.cs.washington.edu/courses/cse599i/18wi/resources/lecture19/lecture19.pdf
