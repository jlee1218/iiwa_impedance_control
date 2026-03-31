#ifndef DYNAMICS_UTILITIES
#define DYNAMICS_UTILITIES

#define URDF_FILE_PATH "/home/ros2_ws/src/lbr_fri_ros2_stack/lbr_description/urdf/iiwa14/iiwa14.urdf"

#include <pinocchio/multibody/model.hpp>

class Dynamics_Utilities {
  public:
    Dynamics_Utilities();
    ~Dynamics_Utilities();

    Eigen::VectorXd get_tau_g(Eigen::VectorXd q);

    Eigen::VectorXd get_tau_cor(Eigen::VectorXd q, Eigen::VectorXd q_dot);

    Eigen::MatrixXd get_J(Eigen::VectorXd q);

    Eigen::MatrixXd get_C(Eigen::VectorXd q, Eigen::VectorXd q_dot);

    Eigen::MatrixXd get_M(Eigen::VectorXd q);

    void forward_kinematics(Eigen::VectorXd q);

    void calculate_ee_pose_delta(Eigen::VectorXd q_des);

    Eigen::VectorXd convertTorqueToWrench(Eigen::VectorXd torque, Eigen::VectorXd q);

    // Eigen::VectorXd joint_impedance(Eigen::VectorXd q_des, Eigen::VectorXd q_dot_des, Eigen::VectorXd q, Eigen::VectorXd q_dot, double K_p = 1.0, double K_d = 1.0);


    Eigen::VectorXd cartesian_impedance_no_g(Eigen::VectorXd x_des, Eigen::VectorXd q, Eigen::VectorXd q_dot);

    Eigen::VectorXd current_pose_delta = Eigen::VectorXd(6).setZero();
    Eigen::VectorXd current_pose = Eigen::VectorXd(6).setZero();
    pinocchio::SE3 current_pose_SE3;
    Eigen::VectorXd prev_commanded_torque = Eigen::VectorXd(7).setZero();

    Eigen::VectorXd default_Kp_cart = (Eigen::VectorXd(6) << 100.0, 100.0, 100.0, 5.0, 5.0, 5.0).finished();
    void set_cartesian_impedance_parameters(double Kp_x, double Kp_y, double Kp_z, double Kp_roll, double Kp_pitch, double Kp_yaw);

    Eigen::MatrixXd Kp_cart = Eigen::MatrixXd(6,6).setZero();
    Eigen::MatrixXd Kd_cart = Eigen::MatrixXd(6,6).setZero();

    Eigen::VectorXd low_pass_filter(Eigen::VectorXd desired_signal, Eigen::VectorXd prev_signal, double cutoff_freq =  20, double process_freq = 500);

  private:
    pinocchio::Model robot_model;
    std::unique_ptr<pinocchio::Data> data; 

    const std::string PLANNING_FRAME = "link_ee";

    const Eigen::VectorXd K_stiction = (Eigen::VectorXd(7) << 50.0, 20.0, 50.0, 80.0, 50.0, 20.0, 50.0).finished();

    const double step_time = 0.001;

    Eigen::Vector3d calculateOrientationError(Eigen::Quaterniond orientation_d, Eigen::Quaterniond orientation);
};



#endif // DYNAMICS_UTILITIES