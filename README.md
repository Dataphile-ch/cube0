# Cube Solver
Explore empirical methods for solving the 3x3x3 Rubik's Cube.

# Why?
The cube can be scrambled in a few rotations.  The upper bound on the number of rotations to solve from any state is known as "God's number" and is proved to be 20 https://cube20.org/#:~:text=At%20long%20last%2C%20God's%20Number,moves%20suffice%20for%20all%20positions.
So, no cube can be more that 20 rotations from the solved state.

Current human solving strategies rely on pattern recognition, step-by-step solving, and memorizing (or programming) up to 100 "algorithms" to move from 1 step to the next.  The most common strategy in speed cubing is CFOP https://jperm.net/3x3/cfop
These strategies typically take at least 40-50 moves to solve a cube, at least 2-3x as many as necessary.

AI researchers have had fun training models to find optimal solutions, but often these take longer than the human methods and consume large resources (CPU and memory).

> Is it possible to quickly find a minimal set of moves to solve any cube?
# Concepts
The Cube() class creates an object with an internal representation (6x3x3 matrix, but it's not very important for most purposes).  It allows the cube to be manipulated and for various attributes to be calculated.  Some methods are obvious, e.g. reset().  Core concepts are:
* Face: A cube has 6 faces: Right, Left, Up, Down, Front, Back [R,L,U,D,F,B], which have colours red, orange, white, yellow, green, blue.
* Facelet/Sticker: each face has 3x3=9 squares which have coloured stickers in the original Rubik's cube.
* Cubelet: what actually gets moved are the cubelets, either an edge cubelet (with 2 stickers) or a corner cubelet (with 3 stickers).  In case it's not obvious, the centre cubelets on each face don't move relative to eachother in a 3x3x3 cube, so they define the target colour for each face.
* Rotation: Each face can be rotated 1, 2, or 3 times.  4 times returns to the original position, so is a non-move.  Some solvers consider each 90° separately, so a 180° rotation is 2 rotations, but this approach makes it more difficult to elminate redundant rotations and therefore introduces a source of error.
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
The entropy is then used to estimate the reward function, which is the distance to the solved state.  In the fully developed solver, the reward function is estimated using a NN trained on the entropy measures as features.
# Some Thoughts
Is it actually possible to estimate entropy (the distance from a given cube to solved state)?

Perhaps there is insufficient information in the cube to determine it's state.  What would this mean?  If you try to solve a cube by examining and un-doing moves, you can find the solution for up to 4-5 rotations, but beyond that it becomes very difficult to sed the reverse moves.  Does this mean that the cube really does have something like entropy and tends to a random state where information about the original structure (solved state) is no longer available??

# To do
* Re-do NN training.  Using the approaoch from Brunetto & Trunda (2017), train the network using the 3 entropy measures as features (naive, align, matrix).  The network can be simple DNN with 3-4 layers.  It can then be used to esimate "distance to solved" from any input.
* Re-implement MCTS using distance estimated from NN.

# References
https://paperswithcode.com/paper/solving-the-rubiks-cube-with-approximate
https://courses.cs.washington.edu/courses/cse599i/18wi/resources/lecture19/lecture19.pdf
