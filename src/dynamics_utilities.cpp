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


Dynamics_Utilities::Dynamics_Utilities() {
  Model full_model;
  pinocchio::urdf::buildModel(URDF_FILE_PATH, full_model);

  auto names = full_model.names;

  // Add frame to plan at the center of the gripper tip
  // SE3 ee_to_gripper_tip_transform = SE3::Identity();
  // ee_to_gripper_tip_transform.translation() = SE3::LinearType(0.0,0.0,0.0);
  // Eigen::AngleAxisd ee_joint7_rot(M_PI, Eigen::Vector3d::UnitX());
  // ee_to_gripper_tip_transform.rotation() =  ee_joint7_rot.toRotationMatrix();
  // full_model.addBodyFrame("gripper_tip_center",full_model.getJointId("joint_7"),ee_to_gripper_tip_transform);  

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

Eigen::VectorXd Dynamics_Utilities::get_tau_g(Eigen::VectorXd q) {
  Eigen::VectorXd tau_g = pinocchio::computeGeneralizedGravity(robot_model, *data, q);

  return tau_g;
}

Eigen::VectorXd Dynamics_Utilities::get_tau_cor(Eigen::VectorXd q, Eigen::VectorXd q_dot) {
  Eigen::MatrixXd cor_mat = pinocchio::computeCoriolisMatrix(robot_model, *data, q, q_dot*(M_PI/180.0));
  Eigen::VectorXd tau_cor = cor_mat*q_dot*(M_PI/180);

  return tau_cor;
}

void Dynamics_Utilities::forward_kinematics(Eigen::VectorXd q) {

  pinocchio::forwardKinematics(this->robot_model, *data, q);

  pinocchio::updateFramePlacements(this->robot_model, *data);

  current_pose_SE3 = data->oMf[this->robot_model.getFrameId(PLANNING_FRAME)];

  Eigen::VectorXd ee_xyz = current_pose_SE3.translation();
  Eigen::VectorXd ee_rpy = current_pose_SE3.rotation().eulerAngles(2, 1, 0);

  current_pose.head(3) = ee_xyz;
  current_pose.tail(3) = ee_rpy;

}

void Dynamics_Utilities::calculate_ee_pose_delta(Eigen::VectorXd x_des) {
  Eigen::VectorXd ee_xyz = current_pose.head(3);
  Eigen::Quaterniond ee_quat(current_pose_SE3.rotation());

  Eigen::AngleAxisd rollAngle(x_des(5), Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd pitchAngle(x_des(4), Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd yawAngle(x_des(3), Eigen::Vector3d::UnitZ());
  Eigen::Quaterniond ee_quat_des = yawAngle * pitchAngle * rollAngle;

  // std::cout << "Desired EE RPY: " << x_des.tail(3).transpose() * (180.0/M_PI) << std::endl;
  // std::cout << "Current EE RPY: " << current_pose.tail(3).transpose() * (180.0/M_PI) << std::endl;

  // std::cout << "Desired EE Quaternion: " << ee_quat_des.w() << " " << ee_quat_des.x() << " " << ee_quat_des.y() << " " << ee_quat_des.z() << std::endl;
  // std::cout << "Current EE Quaternion: " << ee_quat.w() << " " << ee_quat.x() << " " << ee_quat.y() << " " << ee_quat.z() << std::endl;


  Eigen::VectorXd rot_diff = this->calculateOrientationError(ee_quat_des, ee_quat);
  Eigen::VectorXd trans_diff = x_des.head(3)-ee_xyz;

  // std::cout << "Rot Diff: " << rot_diff.transpose() << std::endl;

  Eigen::VectorXd pose_delta(trans_diff.size()+rot_diff.size());
  pose_delta << trans_diff, -rot_diff;

  current_pose_delta = pose_delta;
}

Eigen::VectorXd Dynamics_Utilities::low_pass_filter(Eigen::VectorXd desired_signal, Eigen::VectorXd prev_signal, double cutoff_freq, double process_freq) {
  double beta = exp(-cutoff_freq*(1/process_freq));
  return beta*prev_signal+(1-beta)*desired_signal;
}

Eigen::MatrixXd Dynamics_Utilities::get_C(Eigen::VectorXd q, Eigen::VectorXd q_dot) {
  Eigen::MatrixXd cor_mat = pinocchio::computeCoriolisMatrix(robot_model, *data, q, q_dot*(M_PI/180.0));

  return cor_mat;
}

Eigen::MatrixXd Dynamics_Utilities::get_M(Eigen::VectorXd q) {
  Eigen::MatrixXd m_mat = pinocchio::crba(robot_model,*data, q);
  
  return m_mat;
}

Eigen::MatrixXd Dynamics_Utilities::get_J(Eigen::VectorXd q) {
  // FK needs to be called before computeJointJacobians()
  pinocchio::forwardKinematics(this->robot_model, *data, q);
  // computeJointJacobians() needs to be called before getFrameJacobian()
  Eigen::MatrixXd fullJacobian = pinocchio::computeJointJacobians(robot_model, *data, q);
  Eigen::MatrixXd eeJacobian = pinocchio::getFrameJacobian(this->robot_model, *data, this->robot_model.getFrameId(PLANNING_FRAME), WORLD);
  return fullJacobian;
}

// Eigen::VectorXd Dynamics_Utilities::joint_impedance(Eigen::VectorXd q_des, Eigen::VectorXd q_dot_des, Eigen::VectorXd q, Eigen::VectorXd q_dot, double K_p, double K_d) {
//   Eigen::VectorXd g = this->get_tau_g(q);
//   Eigen::MatrixXd C = this->get_C(q, q_dot);
//   Eigen::MatrixXd M = this->get_M(q);
//   Eigen::MatrixXd q_delta = this->calculated_q_delta_for_continuous_joints(q, q_des);

//   Eigen::VectorXd computed_torque = M*(K_p*q_delta+K_d*(q_dot_des-q_dot*(M_PI/180)))+C*q_dot*(M_PI/180)+g;

//   return computed_torque;
// }

Eigen::VectorXd Dynamics_Utilities::cartesian_impedance(Eigen::VectorXd x_des, Eigen::VectorXd q, Eigen::VectorXd q_dot) {
  Eigen::VectorXd g = this->get_tau_g(q);
  Eigen::MatrixXd C = this->get_C(q, q_dot);
  Eigen::MatrixXd M = this->get_M(q);
  Eigen::MatrixXd J = this->get_J(q);

  this->forward_kinematics(q);
  this->calculate_ee_pose_delta(x_des);

  Eigen::VectorXd cartesian_impedance_wrench = Kp_cart*current_pose_delta-Kd_cart*J*q_dot*(M_PI/180);
  Eigen::VectorXd cartesian_impedance_torques = J.transpose()*cartesian_impedance_wrench;

  current_applied_wrench = cartesian_impedance_wrench;
  current_applied_torque = cartesian_impedance_torques;

  Eigen::VectorXd computed_torque = cartesian_impedance_torques+C*q_dot*(M_PI/180)+g;

  Eigen::VectorXd filtered_torque = low_pass_filter(computed_torque, this->prev_commanded_torque);

  this->prev_commanded_torque = filtered_torque;

  return filtered_torque;
}

Eigen::VectorXd Dynamics_Utilities::cartesian_impedance_no_g(Eigen::VectorXd x_des, Eigen::VectorXd q, Eigen::VectorXd q_dot) {

  Eigen::MatrixXd C = this->get_C(q, q_dot);
  Eigen::MatrixXd M = this->get_M(q);
  Eigen::MatrixXd J = this->get_J(q);

  this->forward_kinematics(q);
  this->calculate_ee_pose_delta(x_des);
  // std::cout << "Current Pose: " << current_pose.transpose() << std::endl;
  // std::cout << "Desired Pose: " << x_des.transpose() << std::endl;
  // std::cout << "Current Pose Delta: " << current_pose_delta.transpose() << std::endl;
  // std::cout << "C" << std::endl << C << std::endl;
  // std::cout << "M" << std::endl << M << std::endl;
  // std::cout << "J" << std::endl << J << std::endl;

  Eigen::VectorXd cartesian_impedance_wrench = Kp_cart*current_pose_delta-Kd_cart*J*q_dot;
  Eigen::VectorXd cartesian_impedance_torques = J.transpose()*cartesian_impedance_wrench;

  current_applied_wrench = cartesian_impedance_wrench;
  current_applied_torque = cartesian_impedance_torques;

  // std::cout << "Cartesian Impedance Wrench: " << cartesian_impedance_wrench.transpose() << std::endl;
  // std::cout << "Cartesian Impedance Torques: " << cartesian_impedance_torques.transpose() << std::endl;
  // std::cout << "Coriolis Torques: " << (C*q_dot).transpose() << std::endl;

  Eigen::VectorXd computed_torque = cartesian_impedance_torques+C*q_dot;
  
  if(prev_commanded_torque.isZero() || prev_commanded_torque.hasNaN()) {
    this->prev_commanded_torque = computed_torque;
  }

  // std::cout << "Computed Torque: " << computed_torque.transpose() << std::endl;
  Eigen::VectorXd filtered_torque = low_pass_filter(computed_torque, this->prev_commanded_torque);

  this->prev_commanded_torque = filtered_torque;

  return filtered_torque;
}

void Dynamics_Utilities::set_cartesian_impedance_parameters(double Kp_x, double Kp_y, double Kp_z, double Kp_roll, double Kp_pitch, double Kp_yaw) {
  Eigen::VectorXd Kp(6);

  Kp << Kp_x, Kp_y, Kp_z, Kp_roll, Kp_pitch, Kp_yaw;
   
  // Critically damp
  int damping_factor = 1;
  Eigen::VectorXd Kd = 2*damping_factor*Kp.array().sqrt();

  Kp_cart.diagonal() = Kp;
  Kd_cart.diagonal() = Kd;
}

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