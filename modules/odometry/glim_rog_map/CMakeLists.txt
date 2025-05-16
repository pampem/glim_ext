cmake_minimum_required(VERSION 3.3)
project(glim_rog_map)

add_compile_options(-std=c++17)

# find_package(ament_cmake REQUIRED)
find_package(glim REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)

add_library(glim_rog_map SHARED
  src/glim_ext/glim_rog_map.cpp
)
target_include_directories(glim_rog_map PRIVATE
  include
  ${nav_msgs_INCLUDE_DIRS}
  ${glim_ext_LIBRARIES}
)
target_link_libraries(glim_rog_map
  glim::glim
  rclcpp::rclcpp
  ${glim_ext_LIBRARIES}
)

# シンボルルックアップエラーが出るため、これは必要。
# なお、上のほうの  ${nav_msgs_INCLUDE_DIRS} これはいるかわからん。
ament_target_dependencies(
  glim_rog_map
  nav_msgs
)