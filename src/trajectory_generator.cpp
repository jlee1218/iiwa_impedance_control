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
  public:
    TrajectoryGeneratorNode() : Node("trajectory_generator")
    {
      diameter_m_ = this->declare_parameter<double>("circle_diameter", 0.1);
      radius_m_ = diameter_m_ / 2.0;
      angular_speed_rad_s_ = this->declare_parameter<double>("circle_angular_speed", 0.4);
      start_time_sec_ = this->now().seconds();

      desired_pose_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
        "impedance_controller/desired_pose", 1);

      state_subscription_ = this->create_subscription<lbr_fri_idl::msg::LBRState>(
        "lbr/lbr_state", 1,
        std::bind(&TrajectoryGeneratorNode::state_callback, this, std::placeholders::_1));

      timer_ = this->create_wall_timer(10ms, std::bind(&TrajectoryGeneratorNode::publish_circle_pose, this));

      RCLCPP_INFO(logger_, "Trajectory Generator Node started.");
    }

  private:
    void state_callback(const lbr_fri_idl::msg::LBRState::SharedPtr current_state)
    {
      measured_joint_positions_ = current_state->measured_joint_position;

      if (!initialized_) {
        dynamics_utilities_.forward_kinematics(
          Eigen::Map<const Eigen::VectorXd>(measured_joint_positions_.data(), measured_joint_positions_.size()));

        for (size_t i = 0; i < 6; ++i) {
          start_pose_[i] = dynamics_utilities_.current_pose(i);
          desired_pose_[i] = start_pose_[i];
        }

        // Circle basis vectors define an inclined plane so x/y/z all vary.
        constexpr double inv_sqrt_2 = 0.7071067811865475;
        circle_u_ = {inv_sqrt_2, -inv_sqrt_2, 0.0};
        circle_v_ = {0.4082482904638631, 0.4082482904638631, -0.8164965809277261};

        circle_center_[0] = start_pose_[0] - radius_m_ * circle_u_[0];
        circle_center_[1] = start_pose_[1] - radius_m_ * circle_u_[1];
        circle_center_[2] = start_pose_[2] - radius_m_ * circle_u_[2];

        initialized_ = true;
        RCLCPP_INFO(
          logger_,
          "Initialized circle from current pose. start=[%.4f %.4f %.4f %.4f %.4f %.4f], radius=%.4f m",
          start_pose_[0], start_pose_[1], start_pose_[2], start_pose_[3], start_pose_[4], start_pose_[5],
          radius_m_);
      }
    }

    void publish_circle_pose()
    {
      if (!initialized_) {
        return;
      }

      const double t = this->now().seconds() - start_time_sec_;
      const double theta = angular_speed_rad_s_ * t;
      const double c = std::cos(theta);
      const double s = std::sin(theta);

      desired_pose_[0] = circle_center_[0] + radius_m_ * (c * circle_u_[0] + s * circle_v_[0]);
      desired_pose_[1] = circle_center_[1] + radius_m_ * (c * circle_u_[1] + s * circle_v_[1]);
      desired_pose_[2] = circle_center_[2] + radius_m_ * (c * circle_u_[2] + s * circle_v_[2]);

      // Keep orientation fixed to the initial measured pose.
      desired_pose_[3] = start_pose_[3];
      desired_pose_[4] = start_pose_[4];
      desired_pose_[5] = start_pose_[5];

      std_msgs::msg::Float64MultiArray desired_pose_msg;
      desired_pose_msg.data.assign(desired_pose_.begin(), desired_pose_.end());
      desired_pose_publisher_->publish(desired_pose_msg);
    }

    std::array<double, 7> measured_joint_positions_{};
    std::array<double, 6> start_pose_{};
    std::array<double, 6> desired_pose_{};

    std::array<double, 3> circle_center_{};
    std::array<double, 3> circle_u_{};
    std::array<double, 3> circle_v_{};

    double diameter_m_ = 0.1;
    double radius_m_ = diameter_m_ / 2.0;
    double angular_speed_rad_s_ = 0.4;

    bool initialized_ = false;
    double start_time_sec_ = 0.0;

    rclcpp::Logger logger_ = rclcpp::get_logger("TrajectoryGeneratorNode");

    Dynamics_Utilities dynamics_utilities_;

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr desired_pose_publisher_;
    rclcpp::Subscription<lbr_fri_idl::msg::LBRState>::SharedPtr state_subscription_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TrajectoryGeneratorNode>());
  rclcpp::shutdown();

  return 0;
}
