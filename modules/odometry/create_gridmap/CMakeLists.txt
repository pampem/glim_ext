cmake_minimum_required(VERSION 3.3)
project(create_gridmap)

add_compile_options(-std=c++17)

# find_package(ament_cmake REQUIRED)
find_package(glim REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)

add_library(create_gridmap SHARED
  src/glim_ext/create_gridmap.cpp
)
target_include_directories(create_gridmap PRIVATE
  include
  ${nav_msgs_INCLUDE_DIRS}
  ${glim_ext_LIBRARIES}
)
target_link_libraries(create_gridmap
  glim::glim
  rclcpp::rclcpp
  ${glim_ext_LIBRARIES}
)

# シンボルルックアップエラーが出るため、これは必要。
# なお、上のほうの  ${nav_msgs_INCLUDE_DIRS} これはいるかわからん。
ament_target_dependencies(
  create_gridmap
  nav_msgs
)