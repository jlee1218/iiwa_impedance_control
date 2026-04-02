# ROS2 Cartesian Impedance Controller for the KUKA LBR iiwa via FRI

This code is a ROS 2 Cartesian impedance controller for the KUKA LBR iiwa. It uses torque control via the lbr_fri_ros2_stack (https://github.com/lbr-stack/lbr_fri_ros2_stack) and computes the robot dynamics with the help of the pinocchio rigid body dynamics library (https://github.com/stack-of-tasks/pinocchio)

## Code Structure
The following table summarizes the structure of the code that exists in the src folder.

File | Description
---- | ----------
dynamics_utilities.cpp dynamics_utilities.h | Impelementation of rigid body dynamics calculations using pinocchio are in this file. This includes the functions to calculate the torque commands for impedance control.
trajectory_generator.cpp | The ROS 2 node used to generate the trajectories used for the scenarios required for ME 780 HW 2.
impedance_control.cpp  | The ROS 2 node used to interface with the lbr_fri_ros2_stack for receiving robot measurements and sending torque commands. This node utilizes dynamics_utilities to compute the impedance control torques.
data_plotter.py | The code used for plotting the data recorded during experiments using ros bags. This code was generated with the help of GitHub Copilot.
