#include <pinocchio/fwd.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <cmath>
#include <iostream>
#include "dynamics_utilities.h"

using namespace pinocchio;

// Initialize the dynamics utilities class by loading the robot model from the URDF file, and setting up necessary data structures for computing dynamics and kinematics using pinocchio.
Dynamics_Utilities::Dynamics_Utilities() {
  Model full_model;
  pinocchio::urdf::buildModel(URDF_FILE_PATH, full_model);

  auto names = full_model.names;

  // Lock Camera and Gripper Joints in Model
  Eigen::VectorXd q = neutral(full_model);

  std::vector<JointIndex> list_of_joints_to_keep_unlocked_by_id = {1, 2, 3, 4, 5, 6, 7};

  std::vector<JointIndex> list_of_joints_to_lock_by_id;

  for (JointIndex joint_id = 1; joint_id < full_model.joints.size(); ++joint_id)
  {
    if (std::find(list_of_joints_to_keep_unlocked_by_id.begin(), list_of_joints_to_keep_unlocked_by_id.end(), joint_id) != list_of_joints_to_keep_unlocked_by_id.end())
      continue;
    else
    {
      list_of_joints_to_lock_by_id.push_back(joint_id);
    }
  }

  this->robot_model = buildReducedModel(full_model, list_of_joints_to_lock_by_id, q);
  this->data.reset(new pinocchio::Data(this->robot_model));

  this->set_cartesian_impedance_parameters(default_Kp_cart(0), default_Kp_cart(1), default_Kp_cart(2), default_Kp_cart(3), default_Kp_cart(4), default_Kp_cart(5));
}

Dynamics_Utilities::~Dynamics_Utilities() {}

// Helper function to calculate gravity compensation torques based on the current joint configuration using pinocchio's computeGeneralizedGravity function.
Eigen::VectorXd Dynamics_Utilities::get_tau_g(Eigen::VectorXd q) {
  Eigen::VectorXd tau_g = pinocchio::computeGeneralizedGravity(robot_model, *data, q);

  return tau_g;
}

// Helper function to calculate the end-effector pose based on the current joint configuration using pinocchio's forwardKinematics function.
void Dynamics_Utilities::forward_kinematics(Eigen::VectorXd q) {

  pinocchio::forwardKinematics(this->robot_model, *data, q);

  pinocchio::updateFramePlacements(this->robot_model, *data);

  current_pose_SE3 = data->oMf[this->robot_model.getFrameId(PLANNING_FRAME)];

  Eigen::VectorXd ee_xyz = current_pose_SE3.translation();
  Eigen::VectorXd ee_rpy = current_pose_SE3.rotation().eulerAngles(2, 1, 0);

  current_pose.head(3) = ee_xyz;
  current_pose.tail(3) = ee_rpy;

}

// Helper function to calculate the end-effector pose delta based on the current end-effector pose and the desired end-effector pose, which is used for the stiffness in the impedance control law.
void Dynamics_Utilities::calculate_ee_pose_delta(Eigen::VectorXd x_des) {
  Eigen::VectorXd ee_xyz = current_pose.head(3);
  Eigen::Quaterniond ee_quat(current_pose_SE3.rotation());

  Eigen::AngleAxisd rollAngle(x_des(5), Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd pitchAngle(x_des(4), Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd yawAngle(x_des(3), Eigen::Vector3d::UnitZ());
  Eigen::Quaterniond ee_quat_des = yawAngle * pitchAngle * rollAngle;

  Eigen::VectorXd rot_diff = this->calculateOrientationError(ee_quat_des, ee_quat);
  Eigen::VectorXd trans_diff = x_des.head(3)-ee_xyz;

  Eigen::VectorXd pose_delta(trans_diff.size()+rot_diff.size());
  pose_delta << trans_diff, -rot_diff;

  current_pose_delta = pose_delta;
}

// A low-pass filter function to filter the commanded torques for smoother control. This is also necessary as the measurements from the robot is noisy.
Eigen::VectorXd Dynamics_Utilities::low_pass_filter(Eigen::VectorXd desired_signal, Eigen::VectorXd prev_signal, double cutoff_freq, double process_freq) {
  double beta = exp(-cutoff_freq*(1/process_freq));
  return beta*prev_signal+(1-beta)*desired_signal;
}

// Helper function to calculate Coriolis and centrifugal torques based on the current joint configuration and velocities using pinocchio's computeCoriolisMatrix function.
Eigen::MatrixXd Dynamics_Utilities::get_C(Eigen::VectorXd q, Eigen::VectorXd q_dot) {
  Eigen::MatrixXd cor_mat = pinocchio::computeCoriolisMatrix(robot_model, *data, q, q_dot*(M_PI/180.0));

  return cor_mat;
}

// Helper function to calculate the mass matrix based on the current joint configuration using pinocchio's crba function.
Eigen::MatrixXd Dynamics_Utilities::get_M(Eigen::VectorXd q) {
  Eigen::MatrixXd m_mat = pinocchio::crba(robot_model,*data, q);
  
  return m_mat;
}

// Helper function to calculate the Jacobian matrix based on the current joint configuration using pinocchio's computeJointJacobians and getFrameJacobian functions.
Eigen::MatrixXd Dynamics_Utilities::get_J(Eigen::VectorXd q) {
  // FK needs to be called before computeJointJacobians()
  pinocchio::forwardKinematics(this->robot_model, *data, q);
  // computeJointJacobians() needs to be called before getFrameJacobian()
  Eigen::MatrixXd fullJacobian = pinocchio::computeJointJacobians(robot_model, *data, q);
  Eigen::MatrixXd eeJacobian = pinocchio::getFrameJacobian(this->robot_model, *data, this->robot_model.getFrameId(PLANNING_FRAME), WORLD);
  return fullJacobian;
}

// Function to calculate the desired Cartesian impedance control torques with Coriolis compensation.
Eigen::VectorXd Dynamics_Utilities::cartesian_impedance_no_g(Eigen::VectorXd x_des, Eigen::VectorXd q, Eigen::VectorXd v_des, Eigen::VectorXd q_dot) {

  Eigen::MatrixXd C = this->get_C(q, q_dot);
  Eigen::MatrixXd M = this->get_M(q);
  Eigen::MatrixXd J = this->get_J(q);

  this->forward_kinematics(q);
  this->calculate_ee_pose_delta(x_des);

  // Calculate the Cartesian impedance wrench
  Eigen::VectorXd cartesian_impedance_wrench = Kp_cart*current_pose_delta-Kd_cart*(v_des-J*q_dot);
  // Convert the Cartesian impedance wrench to joint torques using the Jacobian transpose
  Eigen::VectorXd cartesian_impedance_torques = J.transpose()*cartesian_impedance_wrench;
  // Add Coriolis compensation to the impedance control torques
  Eigen::VectorXd commanded_torque = cartesian_impedance_torques+C*q_dot;
  
  if(prev_commanded_torque.isZero() || prev_commanded_torque.hasNaN()) {
    this->prev_commanded_torque = commanded_torque;
  }
  // Filter the commanded torques for smoother control and to mitigate noise in the measurements.
  Eigen::VectorXd filtered_torque = low_pass_filter(commanded_torque, this->prev_commanded_torque);

  this->prev_commanded_torque = filtered_torque;

  return filtered_torque;
}

// Function to set the Cartesian impedance control parameters (stiffness and damping) for the impedance control law. 
// The stiffness is set based on the input parameters, and the damping is set to be critically damped based on the stiffness values.
void Dynamics_Utilities::set_cartesian_impedance_parameters(double Kp_x, double Kp_y, double Kp_z, double Kp_roll, double Kp_pitch, double Kp_yaw) {
  Eigen::VectorXd Kp(6);

  Kp << Kp_x, Kp_y, Kp_z, Kp_roll, Kp_pitch, Kp_yaw;
   
  // Critically damp
  int damping_factor = 1;
  Eigen::VectorXd Kd = 2*damping_factor*Kp.array().sqrt();

  Kp_cart.diagonal() = Kp;
  Kd_cart.diagonal() = Kd;
}

// Helper function to calculate the orientation error between the current end-effector orientation and the desired end-effector orientation, which is used for the rotational stiffness in the impedance control law.
Eigen::Vector3d Dynamics_Utilities::calculateOrientationError(Eigen::Quaterniond orientation_d, Eigen::Quaterniond orientation)
{
  // Orientation error
  if (orientation_d.coeffs().dot(orientation.coeffs()) < 0.0)
  {
    orientation.coeffs() << -orientation.coeffs();
  }
  // "difference" quaternion
  const Eigen::Quaterniond error_quaternion(orientation * orientation_d.inverse());
  // convert to axis angle
  Eigen::AngleAxisd error_quaternion_angle_axis(error_quaternion);
  return error_quaternion_angle_axis.axis() * error_quaternion_angle_axis.angle();
}

// Helper function to convert joint torques to end-effector wrenches for visualization and analysis purposes, using the Jacobian transpose.
Eigen::VectorXd Dynamics_Utilities::convertTorqueToWrench(Eigen::VectorXd torque, Eigen::VectorXd q) {
  Eigen::MatrixXd J = this->get_J(q);
  // Compute the pseudo-inverse of the Jacobian
  Eigen::MatrixXd J_pseudo_inverse = J.completeOrthogonalDecomposition().pseudoInverse();

  // Convert torque to wrench
  Eigen::VectorXd wrench = J_pseudo_inverse.transpose() * torque;

  return wrench;
}