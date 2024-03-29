# ML and Control Systems
My PSL432 Projects
These are some control theory and ML projects I have created on as part of my PSL432 class.
I list each of them and their descriptions:

- Two_Joint_System.m
  
The provided code is a simulation of an agent-controlled robotic arm with the goal of reaching a dynamic target position. The simulation employs a control strategy that estimates joint states and calculates actions to minimize the distance from the target. The agent utilizes a third-order Hurwitz polynomial to determine the desired change in acceleration, translating it into torque to drive the arm's movement. The simulation iterates over time steps, updating the agent's estimates and actual joint states based on the calculated actions. The overarching goal is to investigate the control dynamics of a two-joint robotic arm, emphasizing the interaction between estimation, control actions, and the achievement of target positions. The resulting plots illustrate the evolution of joint positions, target positions, and control actions throughout the simulation.

- mnist_neural_mapnet.m

In this assignment, I implemented a neural network using backpropagation with Adam optimization to recognize handwritten digits from the MNIST dataset. The network architecture is a 4-layer mapnet with two middle layers of 28-by-28 maps with a field-fraction of 0.25. Additionally, layer 2 directly projects to layer 4 through weight matrix W{4}. The activation function used in the middle layers is a modified version of the ReLU function called relog. The final layer of the network is affine, followed by a softmax function.

I wrote four MATLAB files:

init_surname_create_mapnet.m: Initializes the mapnet architecture, setting biases to zero and initializing neuron weights from a normal distribution with specific standard deviation based on the number of neuron inputs.
init_surname_forward_relog.m: Implements the forward pass of the network using the relog activation function.
init_surname_backprop_relog.m: Implements the backpropagation algorithm for the relog activation function, considering the direct projection from layer 2 to layer 4.
INIT_SURNAME_A2.m: Main script file responsible for running the training loop over 10 epochs, shuffling the data at the start of each epoch, and evaluating the network's performance on the test set after each epoch.
I utilized provided Adam optimization function with specific hyperparameters and fed the images to the network in minibatches of 100. I aimed to ensure concise and efficient code, avoiding loops and lists for minibatch processing. The goal was to observe a decreasing number of incorrect guesses on the test set as training progressed.
