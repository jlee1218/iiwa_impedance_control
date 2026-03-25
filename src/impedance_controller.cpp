#include <chrono>
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/wrench.hpp"

#include "lbr_fri_idl/msg/lbr_state.hpp"
#include "lbr_fri_idl/msg/lbr_torque_command.hpp"

#include "dynamics_utilities.h"

using namespace std::chrono_literals;

/* This example creates a subclass of Node and uses std::bind() to register a
* member function as a callback from the timer. */

class ImpedanceControllerNode : public rclcpp::Node {
  public:
    ImpedanceControllerNode() : Node("impedance_controller"), count_(0)
    {
      torque_publisher_ = this->create_publisher<lbr_fri_idl::msg::LBRTorqueCommand>("lbr/command/lbr_torque_command", 1);
      state_subscription_ = this->create_subscription<lbr_fri_idl::msg::LBRState>(
        "lbr/lbr_state", 1, std::bind(&ImpedanceControllerNode::state_callback, this, std::placeholders::_1));

      pose_delta_publisher_ = this->create_publisher<geometry_msgs::msg::Pose>("pose_delta", 1);
      wrench_publisher_ = this->create_publisher<geometry_msgs::msg::Wrench>("wrench", 1);

      RCLCPP_INFO(logger, "Impedance Controller Node has been started.");
    }

  private:
    void state_callback(const lbr_fri_idl::msg::LBRState::SharedPtr current_state)
    {
      lbr_fri_idl::msg::LBRTorqueCommand torque_command;
      current_joint_positions_ = current_state->measured_joint_position;
      current_joint_torques_ = current_state->measured_torque;

      if (!first_callback_) {
        first_callback_ = true;
        this->set_desired_ee_pose();

        // RCLCPP_INFO(this->get_logger(), "Current Pose: [%f, %f, %f, %f, %f, %f]", 
        //   this->dynamics_utilities.current_pose(0), 
        //   this->dynamics_utilities.current_pose(1), 
        //   this->dynamics_utilities.current_pose(2), 
        //   this->dynamics_utilities.current_pose(3), 
        //   this->dynamics_utilities.current_pose(4), 
        //   this->dynamics_utilities.current_pose(5));

      } else {
        const double dt = static_cast<double>((current_state->time_stamp_nano_sec - prev_time_stamp_) / 1e9);
        
        for (size_t i = 0; i < 7; ++i) {
          current_joint_velocities_[i] = (current_joint_positions_[i] - prev_joint_positions_[i]) / dt;
        }

        // std::cout << "dt" << dt << std::endl;
        // std::cout << "Current Joint Velocities: " << std::endl;
        // for (size_t i = 0; i < 7; ++i) {
        //   std::cout << "Joint " << i+1 << ": " << current_joint_velocities_[i] << " rad/s" << std::endl;
        // }

        torque_command.joint_position = current_state->measured_joint_position;

        Eigen::Map<const Eigen::VectorXd> desired_ee_pose_eigen(desired_ee_pose_.data(), desired_ee_pose_.size());
        Eigen::Map<const Eigen::VectorXd> current_joint_positions_eigen(current_joint_positions_.data(), current_joint_positions_.size());
        Eigen::Map<const Eigen::VectorXd> current_joint_velocities_eigen(current_joint_velocities_.data(), current_joint_velocities_.size());

        Eigen::VectorXd impedance_control_torques = dynamics_utilities.cartesian_impedance_no_g(
          desired_ee_pose_eigen,
          current_joint_positions_eigen,
          current_joint_velocities_eigen);
        
        // std::cout << "Impedance Control Torques: " << impedance_control_torques.transpose() << std::endl;

        for (size_t i = 0; i < 7; ++i) {
          torque_command.torque[i] = impedance_control_torques(i);
        }
        
        // Publish pose_delta as geometry_msgs::Pose
        geometry_msgs::msg::Pose pose_delta_msg;
        pose_delta_msg.position.x = dynamics_utilities.current_pose_delta(0);
        pose_delta_msg.position.y = dynamics_utilities.current_pose_delta(1);
        pose_delta_msg.position.z = dynamics_utilities.current_pose_delta(2);
        
        // Convert RPY to quaternion for pose delta orientation
        double roll = dynamics_utilities.current_pose_delta(3);
        double pitch = dynamics_utilities.current_pose_delta(4);
        double yaw = dynamics_utilities.current_pose_delta(5);
        
        double cy = std::cos(yaw * 0.5);
        double sy = std::sin(yaw * 0.5);
        double cp = std::cos(pitch * 0.5);
        double sp = std::sin(pitch * 0.5);
        double cr = std::cos(roll * 0.5);
        double sr = std::sin(roll * 0.5);
        
        pose_delta_msg.orientation.w = cr * cp * cy + sr * sp * sy;
        pose_delta_msg.orientation.x = sr * cp * cy - cr * sp * sy;
        pose_delta_msg.orientation.y = cr * sp * cy + sr * cp * sy;
        pose_delta_msg.orientation.z = cr * cp * sy - sr * sp * cy;
        
        // Publish wrench as geometry_msgs::Wrench
        geometry_msgs::msg::Wrench wrench_msg;
        wrench_msg.force.x = dynamics_utilities.current_applied_wrench(0);
        wrench_msg.force.y = dynamics_utilities.current_applied_wrench(1);
        wrench_msg.force.z = dynamics_utilities.current_applied_wrench(2);
        wrench_msg.torque.x = dynamics_utilities.current_applied_wrench(3);
        wrench_msg.torque.y = dynamics_utilities.current_applied_wrench(4);
        wrench_msg.torque.z = dynamics_utilities.current_applied_wrench(5);
        
        pose_delta_publisher_->publish(pose_delta_msg);
        wrench_publisher_->publish(wrench_msg);
 

        torque_publisher_->publish(torque_command);
        
      }

      prev_time_stamp_ = current_state->time_stamp_nano_sec;
      prev_joint_positions_ = current_joint_positions_;

    }

    void set_desired_ee_pose() {
      this->dynamics_utilities.forward_kinematics(Eigen::Map<const Eigen::VectorXd>(current_joint_positions_.data(), current_joint_positions_.size()));
      Eigen::VectorXd desired_ee_pose_eigen = dynamics_utilities.current_pose;
      
      desired_ee_pose_ = {desired_ee_pose_eigen(0), 
                          desired_ee_pose_eigen(1), 
                          desired_ee_pose_eigen(2), 
                          desired_ee_pose_eigen(3), 
                          desired_ee_pose_eigen(4), 
                          desired_ee_pose_eigen(5)};

      std::cout << "Desired EE Pose set to current pose: " << std::endl;
      std::cout << "Position (x, y, z): " << desired_ee_pose_[0] << ", " << desired_ee_pose_[1] << ", " << desired_ee_pose_[2] << std::endl;
      std::cout << "Orientation (roll, pitch, yaw): " << desired_ee_pose_[3] << ", " << desired_ee_pose_[4] << ", " << desired_ee_pose_[5] << std::endl;

    }

    std::array<double, 6> desired_ee_pose_{};
    std::array<double, 7> prev_joint_positions_{};
    std::array<double, 7> current_joint_positions_{};
    std::array<double, 7> current_joint_velocities_{};
    std::array<double, 7> current_joint_torques_{};

    uint32_t prev_time_stamp_ = 0;

    rclcpp::Logger logger = rclcpp::get_logger("ImpedanceControllerNode");

    bool first_callback_ = false;

    Dynamics_Utilities dynamics_utilities;

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<lbr_fri_idl::msg::LBRTorqueCommand>::SharedPtr torque_publisher_;
    rclcpp::Subscription<lbr_fri_idl::msg::LBRState>::SharedPtr state_subscription_;

    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pose_delta_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::Wrench>::SharedPtr wrench_publisher_;
    size_t count_;
};
  
int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImpedanceControllerNode>());
  rclcpp::shutdown();

  return 0;
};
