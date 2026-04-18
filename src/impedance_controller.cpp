#include <chrono>
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <cmath>
#include <algorithm>

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

/* This example creates a subclass of Node and uses std::bind() to register a
* member function as a callback from the timer. */

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

      current_impedance_torque_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("impedance_controller/current_impedance_torque", 1);
      current_coriolis_torque_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("impedance_controller/current_coriolis_torque", 1);
      current_pose_delta_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("impedance_controller/current_pose_delta", 1);
      current_stiffness_wrench_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("impedance_controller/current_stiffness_wrench", 1);
      current_damping_wrench_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("impedance_controller/current_damping_wrench", 1);
      current_joint_velocities_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("impedance_controller/current_joint_velocities", 1);

      current_joint_velocities_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

      RCLCPP_INFO(logger, "Impedance Controller Node has been started.");
    }

  private:
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

    void state_callback(const lbr_fri_idl::msg::LBRState::SharedPtr current_state)
    {
      lbr_fri_idl::msg::LBRTorqueCommand torque_command;
      measured_joint_positions_ = current_state->measured_joint_position;
      measured_joint_torques_ = current_state->measured_torque;

      if (!first_callback_) {
        first_callback_ = true;
        this->initialize_ee_pose();
        prev_joint_positions_ = measured_joint_positions_;

      } else {
        double dt = static_cast<double>((current_state->time_stamp_nano_sec - prev_time_stamp_) / 1e9);

        // const double dt = 0.002;
        
        for(size_t i = 0; i < 7; ++i) {
          measured_joint_positions_[i] = low_pass_filter(measured_joint_positions_[i], prev_joint_positions_[i], 0.9);
        }

        // for (size_t i = 0; i < 7; ++i) {
        //   current_joint_velocities_[i] = low_pass_filter(((measured_joint_positions_[i] - prev_joint_positions_[i]) / dt), current_joint_velocities_[i], 0.1);
        // }
        
        for (size_t i = 0; i < 7; ++i) {
          current_joint_velocities_[i] = (measured_joint_positions_[i] - prev_joint_positions_[i]) / dt;
        }

        torque_command.joint_position = current_state->measured_joint_position;

        Eigen::Map<const Eigen::VectorXd> desired_ee_pose_eigen(desired_ee_pose_.data(), desired_ee_pose_.size());
        Eigen::Map<const Eigen::VectorXd> measured_joint_torques_eigen(measured_joint_torques_.data(), measured_joint_torques_.size());
        Eigen::Map<const Eigen::VectorXd> measured_joint_positions_eigen(measured_joint_positions_.data(), measured_joint_positions_.size());
        Eigen::Map<const Eigen::VectorXd> current_joint_velocities_eigen(current_joint_velocities_.data(), current_joint_velocities_.size());

        Eigen::VectorXd impedance_control_torques = dynamics_utilities.cartesian_impedance_no_g(
          desired_ee_pose_eigen,
          measured_joint_positions_eigen,
          current_joint_velocities_eigen);
        
        // std::cout << "Impedance Control Torques: " << impedance_control_torques.transpose() << std::endl;

        for (size_t i = 0; i < 7; ++i) {
          torque_command.torque[i] = impedance_control_torques(i);
        }
        
        // Publish pose_delta as geometry_msgs::Pose
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
        
        // Publish wrench as geometry_msgs::Wrench
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

        std_msgs::msg::Float64MultiArray impedance_torque_msg;
        impedance_torque_msg.data.resize(7);
        for (size_t i = 0; i < 7; ++i) {
          impedance_torque_msg.data[i] = dynamics_utilities.current_impedance_torque(i);
        }

        std_msgs::msg::Float64MultiArray coriolis_torque_msg;
        coriolis_torque_msg.data.resize(7);
        for (size_t i = 0; i < 7; ++i) {
          coriolis_torque_msg.data[i] = dynamics_utilities.current_coriolis_torque(i);
        } 

        std_msgs::msg::Float64MultiArray current_pose_delta_msg;
        current_pose_delta_msg.data.resize(6);
        for (size_t i = 0; i < 6; ++i) {
          current_pose_delta_msg.data[i] = dynamics_utilities.current_pose_delta(i);
        }

        std_msgs::msg::Float64MultiArray stiffness_wrench_msg;
        stiffness_wrench_msg.data.resize(6);
        for (size_t i = 0; i < 6; ++i) {
          stiffness_wrench_msg.data[i] = dynamics_utilities.current_stiffness_wrench(i);
        }

        std_msgs::msg::Float64MultiArray damping_wrench_msg;
        damping_wrench_msg.data.resize(6);
        for (size_t i = 0; i < 6; ++i) {
          damping_wrench_msg.data[i] = dynamics_utilities.current_damping_wrench(i);
        }

        std_msgs::msg::Float64MultiArray current_joint_velocities_msg;
        current_joint_velocities_msg.data.resize(7);
        for (size_t i = 0; i < 7; ++i) {
          current_joint_velocities_msg.data[i] = current_joint_velocities_[i];
        }

        current_impedance_torque_publisher_->publish(impedance_torque_msg);
        current_coriolis_torque_publisher_->publish(coriolis_torque_msg);
        current_stiffness_wrench_publisher_->publish(stiffness_wrench_msg);
        current_damping_wrench_publisher_->publish(damping_wrench_msg);
        current_joint_velocities_publisher_->publish(current_joint_velocities_msg);

        commanded_pose_publisher_->publish(commanded_pose_msg);
        measured_pose_publisher_->publish(measured_pose_msg);
        commanded_wrench_publisher_->publish(commanded_wrench_msg);
        measured_wrench_publisher_->publish(measured_wrench_msg);
        current_pose_delta_publisher_->publish(current_pose_delta_msg);
        torque_publisher_->publish(torque_command);
        
      }

      prev_time_stamp_ = current_state->time_stamp_nano_sec;
      prev_joint_positions_ = measured_joint_positions_;

    }

    void initialize_ee_pose() {
      this->dynamics_utilities.forward_kinematics(Eigen::Map<const Eigen::VectorXd>(measured_joint_positions_.data(), measured_joint_positions_.size()));
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

    double low_pass_filter(const double new_value, const double prev_value, const double alpha) {
      const double alpha_clamped = std::clamp(alpha, 0.0, 1.0);
      return alpha_clamped * new_value + (1.0 - alpha_clamped) * prev_value;
    }

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

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr current_impedance_torque_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr current_coriolis_torque_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr current_stiffness_wrench_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr current_damping_wrench_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr current_joint_velocities_publisher_;

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr current_pose_delta_publisher_;

    size_t count_;
};
  
int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImpedanceControllerNode>());
  rclcpp::shutdown();

  return 0;
};
