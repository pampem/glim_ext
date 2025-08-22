cmake_minimum_required(VERSION 3.3)
project(glim_octomap)

add_compile_options(-std=c++17)

find_package(glim REQUIRED)
find_package(rclcpp REQUIRED)
find_package(octomap REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

add_library(glim_octomap SHARED
  src/glim_ext/glim_octomap.cpp
)

target_include_directories(glim_octomap PRIVATE
  include
  ${OCTOMAP_INCLUDE_DIRS}
  ${octomap_msgs_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${glim_ext_LIBRARIES}
)

target_link_libraries(glim_octomap
  glim::glim
  rclcpp::rclcpp
  ${OCTOMAP_LIBRARIES}
  ${glim_ext_LIBRARIES}
)

# シンボルルックアップエラーが出るため、これは必要。
ament_target_dependencies(
  glim_octomap
  octomap_msgs
  sensor_msgs
)