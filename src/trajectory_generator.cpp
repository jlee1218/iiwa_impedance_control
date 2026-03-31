#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

#include "lbr_fri_idl/msg/lbr_state.hpp"

#include "dynamics_utilities.h"

using namespace std::chrono_literals;

class TrajectoryGeneratorNode : public rclcpp::Node {
  static constexpr double amplitude_ = 0.05; // 0.1
  static constexpr double frequency_ = 0.1;

  public:
    TrajectoryGeneratorNode() : Node("trajectory_generator")
    {

      desired_pose_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
        "impedance_controller/desired_pose", 1);

      state_subscription_ = this->create_subscription<lbr_fri_idl::msg::LBRState>(
        "lbr/lbr_state", 1,
        std::bind(&TrajectoryGeneratorNode::state_callback, this, std::placeholders::_1));

      phase_ = 0.0;
      dt_ = 1.0/500.0; 

      RCLCPP_INFO(logger_, "Trajectory Generator Node started.");
    }

  private:
    void state_callback(const lbr_fri_idl::msg::LBRState::SharedPtr current_state)
    {
      if(!initialized_) {
        dynamics_utilities_.forward_kinematics(
          Eigen::Map<const Eigen::VectorXd>(current_state->measured_joint_position.data(), current_state->measured_joint_position.size()));

        for (size_t i = 0; i < 6; ++i) {
          start_pose_[i] = dynamics_utilities_.current_pose(i);
        }
        initialized_ = true;
      }

      // sine_traj();
      down_shift_traj();

      std_msgs::msg::Float64MultiArray desired_pose_msg;
      desired_pose_msg.data.resize(6);

      for (size_t i = 0; i < 6; ++i) {
        desired_pose_msg.data[i] = desired_pose_[i];
      }
      
      desired_pose_publisher_->publish(desired_pose_msg);
    }

    void sine_traj() {
        desired_pose_[0] = start_pose_[0] + amplitude_ * sin(phase_);
        desired_pose_[1] = start_pose_[1] + amplitude_ * sin(phase_);  
        desired_pose_[2] = start_pose_[2] + amplitude_ * sin(phase_); 
        desired_pose_[3] = start_pose_[3]; 
        desired_pose_[4] = start_pose_[4]; 
        desired_pose_[5] = start_pose_[5];  
  
        phase_ += 2 * M_PI * frequency_ * dt_;
    }

    void down_shift_traj() {
        desired_pose_[0] = start_pose_[0];
        desired_pose_[1] = start_pose_[1];  
        desired_pose_[2] = start_pose_[2] + amplitude_ * sin(phase_ + M_PI/2 ) - amplitude_; 
        desired_pose_[3] = start_pose_[3]; 
        desired_pose_[4] = start_pose_[4]; 
        desired_pose_[5] = start_pose_[5]; 
  
        phase_ += 2 * M_PI * frequency_ * dt_;
    }


    std::array<double, 7> measured_joint_positions_{};
    std::array<double, 6> desired_pose_{};
    std::array<double, 6> start_pose_{};
    bool initialized_ = false;


    double dt_;
    double phase_;

    rclcpp::Logger logger_ = rclcpp::get_logger("TrajectoryGeneratorNode");

    Dynamics_Utilities dynamics_utilities_;

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr desired_pose_publisher_;
    rclcpp::Subscription<lbr_fri_idl::msg::LBRState>::SharedPtr state_subscription_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TrajectoryGeneratorNode>());
  rclcpp::shutdown();

  return 0;
}
