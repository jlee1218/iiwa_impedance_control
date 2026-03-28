cmake_minimum_required(VERSION 3.8)
project(jeffrey_controllers)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(lbr_fri_idl REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Eigen3 REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

add_executable(impedance_controller src/impedance_controller.cpp src/dynamics_utilities.cpp)
ament_target_dependencies(impedance_controller
  rclcpp
  tf2
  tf2_ros
  tf2_geometry_msgs
  std_msgs
  geometry_msgs
  lbr_fri_idl
  pinocchio
  Eigen3
)

add_executable(trajectory_generator src/trajectory_generator.cpp src/dynamics_utilities.cpp)
ament_target_dependencies(trajectory_generator
  rclcpp
  tf2
  std_msgs
  geometry_msgs
  lbr_fri_idl
  pinocchio
  Eigen3
)

target_include_directories(impedance_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  ${EIGEN3_INCLUDE_DIRS})
target_compile_features(impedance_controller PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

install(TARGETS impedance_controller
  DESTINATION lib/${PROJECT_NAME})

install(TARGETS trajectory_generator
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
