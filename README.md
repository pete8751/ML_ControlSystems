# PSL432ControlSystems
My PSL432 Projects
These are some control projects I have worked on as part of my PSL432 class.
I list each of them and their descriptions:

- Two_Joint_System.m
  
The provided code is a simulation of an agent-controlled robotic arm with the goal of reaching a dynamic target position. The simulation employs a control strategy that estimates joint states and calculates actions to minimize the distance from the target. The agent utilizes a third-order Hurwitz polynomial to determine the desired change in acceleration, translating it into torque to drive the arm's movement. The simulation iterates over time steps, updating the agent's estimates and actual joint states based on the calculated actions. The overarching goal is to investigate the control dynamics of a two-joint robotic arm, emphasizing the interaction between estimation, control actions, and the achievement of target positions. The resulting plots illustrate the evolution of joint positions, target positions, and control actions throughout the simulation.
