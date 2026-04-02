#include <chrono>
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/wrench.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "lbr_fri_idl/msg/lbr_state.hpp"
#include "lbr_fri_idl/msg/lbr_torque_command.hpp"

#include "dynamics_utilities.h"

using namespace std::chrono_literals;


class ImpedanceControllerNode : public rclcpp::Node {
  public:
    ImpedanceControllerNode() : Node("impedance_controller"), count_(0)
    {
      torque_publisher_ = this->create_publisher<lbr_fri_idl::msg::LBRTorqueCommand>("lbr/command/lbr_torque_command", 1);
      state_subscription_ = this->create_subscription<lbr_fri_idl::msg::LBRState>(
        "lbr/lbr_state", 1, std::bind(&ImpedanceControllerNode::state_callback, this, std::placeholders::_1));
      impedance_parameters_subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
        "impedance_controller/parameters", 1, std::bind(&ImpedanceControllerNode::impedance_parameters_callback, this, std::placeholders::_1));
      desired_pose_subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
        "impedance_controller/desired_pose", 1, std::bind(&ImpedanceControllerNode::desired_pose_callback, this, std::placeholders::_1));

      measured_pose_publisher_ = this->create_publisher<geometry_msgs::msg::Pose>("impedance_controller/measured_pose", 1);
      commanded_pose_publisher_ = this->create_publisher<geometry_msgs::msg::Pose>("impedance_controller/commanded_pose", 1);
      commanded_wrench_publisher_ = this->create_publisher<geometry_msgs::msg::Wrench>("impedance_controller/commanded_wrench", 1);
      measured_wrench_publisher_ = this->create_publisher<geometry_msgs::msg::Wrench>("impedance_controller/measured_wrench", 1);

      RCLCPP_INFO(logger, "Impedance Controller Node has been started.");
    }

  private:
    // Callback function to receive desired end-effector pose updates from trajectory generator node
    void desired_pose_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
      if (msg->data.size() != 6) {
        RCLCPP_WARN(logger, "Received desired pose of incorrect size. Expected 6, got %zu", msg->data.size());
        return;
      }
      for (size_t i = 0; i < 6; ++i) {
        desired_ee_pose_[i] = msg->data[i];
      }
      RCLCPP_INFO(logger, "Updated desired end-effector pose. [%f, %f, %f, %f, %f, %f]", 
        desired_ee_pose_[0], desired_ee_pose_[1], desired_ee_pose_[2], 
        desired_ee_pose_[3], desired_ee_pose_[4], desired_ee_pose_[5]);
    }

    // Callback function to set impedance parameters for parameter tuning 
    void impedance_parameters_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
      if (msg->data.size() != 6) {
        RCLCPP_WARN(logger, "Received impedance parameters of incorrect size. Expected 6, got %zu", msg->data.size());
        return;
      }
      dynamics_utilities.set_cartesian_impedance_parameters(
        msg->data[0], msg->data[1], msg->data[2], msg->data[3], msg->data[4], msg->data[5]);
      RCLCPP_INFO(logger, "Updated Cartesian impedance parameters.");
    }

    // Callback function to receive robot state updates from FRI and compute impedance control torques to be published to the robot via the "lbr/command/lbr_torque_command" topic to FRI.
    void state_callback(const lbr_fri_idl::msg::LBRState::SharedPtr current_state)
    {
      lbr_fri_idl::msg::LBRTorqueCommand torque_command;
      measured_joint_positions_ = current_state->measured_joint_position;
      measured_joint_torques_ = current_state->measured_torque;

      if (!first_callback_) {
        // Initialize the impedance controller on the robot's starting configuration.
        first_callback_ = true;
        this->initialize_ee_pose();

      } else {
        const double dt = static_cast<double>((current_state->time_stamp_nano_sec - prev_time_stamp_) / 1e9);
        
        // Set joint positions to measured joint positions for FRI interface. No effect on the torque controller.
        torque_command.joint_position = current_state->measured_joint_position; 

        // Calculate the current joint velocities based on the change in measured joint positions over time.
        for (size_t i = 0; i < 7; ++i) {
          current_joint_velocities_[i] = (measured_joint_positions_[i] - prev_joint_positions_[i]) / dt;
        }
        
        // Calculate the desired end-effector velocity based on the change in desired end-effector pose over time from trajectory generator.
        for (size_t i = 0; i < 6; ++i) {
          desired_ee_velocity_[i] = (desired_ee_pose_[i] - prev_desired_ee_pose_[i]) / dt;
        }

        Eigen::Map<const Eigen::VectorXd> desired_ee_pose_eigen(desired_ee_pose_.data(), desired_ee_pose_.size());
        Eigen::Map<const Eigen::VectorXd> desired_ee_velocity_eigen(desired_ee_velocity_.data(), desired_ee_velocity_.size());
        Eigen::Map<const Eigen::VectorXd> measured_joint_torques_eigen(measured_joint_torques_.data(), measured_joint_torques_.size());
        Eigen::Map<const Eigen::VectorXd> measured_joint_positions_eigen(measured_joint_positions_.data(), measured_joint_positions_.size());
        Eigen::Map<const Eigen::VectorXd> current_joint_velocities_eigen(current_joint_velocities_.data(), current_joint_velocities_.size());
        

        // Computer impedance control torques based on the current robot state and the desired end-effector pose using the dynamics utilities class, and publish the torque command to FRI.
        Eigen::VectorXd impedance_control_torques = dynamics_utilities.cartesian_impedance_no_g(
          desired_ee_pose_eigen,
          measured_joint_positions_eigen,
          desired_ee_velocity_eigen,
          current_joint_velocities_eigen);
        

        for (size_t i = 0; i < 7; ++i) {
          torque_command.torque[i] = impedance_control_torques(i);
        }

        torque_publisher_->publish(torque_command);
        


        // The following code in this function is for publishing the measured and commanded end-effector poses and wrenches for visualization and analysis. 
        geometry_msgs::msg::Pose measured_pose_msg;
        measured_pose_msg.position.x = dynamics_utilities.current_pose(0);
        measured_pose_msg.position.y = dynamics_utilities.current_pose(1);
        measured_pose_msg.position.z = dynamics_utilities.current_pose(2);
        
        tf2::Quaternion quaternion_current;
        quaternion_current.setRPY(dynamics_utilities.current_pose(3), 
                                  dynamics_utilities.current_pose(4), 
                                  dynamics_utilities.current_pose(5));

        measured_pose_msg.orientation = tf2::toMsg(quaternion_current);

        geometry_msgs::msg::Pose commanded_pose_msg;
        commanded_pose_msg.position.x = desired_ee_pose_[0];
        commanded_pose_msg.position.y = desired_ee_pose_[1];
        commanded_pose_msg.position.z = desired_ee_pose_[2];

        tf2::Quaternion quaternion_desired;
        quaternion_desired.setRPY(desired_ee_pose_[3], 
                                  desired_ee_pose_[4], 
                                  desired_ee_pose_[5]);

        commanded_pose_msg.orientation = tf2::toMsg(quaternion_desired);
        
        geometry_msgs::msg::Wrench commanded_wrench_msg;

        Eigen::VectorXd commanded_wrench = dynamics_utilities.convertTorqueToWrench(impedance_control_torques, measured_joint_positions_eigen);

        commanded_wrench_msg.force.x = commanded_wrench(0);
        commanded_wrench_msg.force.y = commanded_wrench(1);
        commanded_wrench_msg.force.z = commanded_wrench(2);
        commanded_wrench_msg.torque.x = commanded_wrench(3);
        commanded_wrench_msg.torque.y = commanded_wrench(4);
        commanded_wrench_msg.torque.z = commanded_wrench(5);
        
        geometry_msgs::msg::Wrench measured_wrench_msg;

        Eigen::VectorXd measured_wrench = dynamics_utilities.convertTorqueToWrench(measured_joint_torques_eigen, measured_joint_positions_eigen);

        measured_wrench_msg.force.x = measured_wrench(0);
        measured_wrench_msg.force.y = measured_wrench(1);
        measured_wrench_msg.force.z = measured_wrench(2);
        measured_wrench_msg.torque.x = measured_wrench(3);
        measured_wrench_msg.torque.y = measured_wrench(4);
        measured_wrench_msg.torque.z = measured_wrench(5);

        
        commanded_pose_publisher_->publish(commanded_pose_msg);
        measured_pose_publisher_->publish(measured_pose_msg);
        commanded_wrench_publisher_->publish(commanded_wrench_msg);
        measured_wrench_publisher_->publish(measured_wrench_msg);
      }

      prev_time_stamp_ = current_state->time_stamp_nano_sec;
      prev_joint_positions_ = measured_joint_positions_;
      prev_desired_ee_pose_ = desired_ee_pose_;

    }

    // Initialize the impedance controller on the robot's starting configuration.
    void initialize_ee_pose() {
      this->dynamics_utilities.forward_kinematics(Eigen::Map<const Eigen::VectorXd>(measured_joint_positions_.data(), measured_joint_positions_.size()));
      Eigen::VectorXd desired_ee_pose_eigen = dynamics_utilities.current_pose;
      
      desired_ee_pose_ = {desired_ee_pose_eigen(0), 
                          desired_ee_pose_eigen(1), 
                          desired_ee_pose_eigen(2), 
                          desired_ee_pose_eigen(3), 
                          desired_ee_pose_eigen(4), 
                          desired_ee_pose_eigen(5)};

      prev_desired_ee_pose_ = desired_ee_pose_;

      desired_ee_velocity_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

      std::cout << "Desired EE Pose set to current pose: " << std::endl;
      std::cout << "Position (x, y, z): " << desired_ee_pose_[0] << ", " << desired_ee_pose_[1] << ", " << desired_ee_pose_[2] << std::endl;
      std::cout << "Orientation (roll, pitch, yaw): " << desired_ee_pose_[3] << ", " << desired_ee_pose_[4] << ", " << desired_ee_pose_[5] << std::endl;

    }

    std::array<double, 6> desired_ee_velocity_{};
    std::array<double, 6> prev_desired_ee_pose_{};
    std::array<double, 6> desired_ee_pose_{};
    std::array<double, 7> prev_joint_positions_{};
    std::array<double, 7> measured_joint_positions_{};
    std::array<double, 7> current_joint_velocities_{};
    std::array<double, 7> measured_joint_torques_{};

    uint32_t prev_time_stamp_ = 0;

    rclcpp::Logger logger = rclcpp::get_logger("ImpedanceControllerNode");

    bool first_callback_ = false;

    Dynamics_Utilities dynamics_utilities;

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<lbr_fri_idl::msg::LBRTorqueCommand>::SharedPtr torque_publisher_;
    rclcpp::Subscription<lbr_fri_idl::msg::LBRState>::SharedPtr state_subscription_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr impedance_parameters_subscription_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr desired_pose_subscription_;

    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr measured_pose_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr commanded_pose_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::Wrench>::SharedPtr commanded_wrench_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::Wrench>::SharedPtr measured_wrench_publisher_;

    size_t count_;
};
  
int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImpedanceControllerNode>());
  rclcpp::shutdown();

  return 0;
};
