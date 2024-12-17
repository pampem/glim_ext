#include <thread>
#include <iostream>
#include <boost/format.hpp>

#include <rclcpp/rclcpp.hpp>

#include <opencv2/core.hpp>

#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>
#include <gtsam_points/optimizers/incremental_fixed_lag_smoother_ext.hpp>
#include <gtsam_points/optimizers/incremental_fixed_lag_smoother_with_fallback.hpp>

#include <glim/odometry/callbacks.hpp>
#include <glim/mapping/callbacks.hpp>
#include <glim/util/logging.hpp>
#include <glim/util/extension_module.hpp>
#include <glim_ext/console_colors.hpp>
#include <glim_ext/util/config_ext.hpp>

#include <nav_msgs/msg/occupancy_grid.hpp>

// グローバル変数は使えないことが多いので注意。特にROS関連。

namespace glim {

class CreateGridmap : public ExtensionModule {
public:
  CreateGridmap() : logger_(create_module_logger("create_gridmap")) {
    logger_ -> info("Starting create_gridmap...");

    glim::Config config(glim::GlobalConfigExt::get_config_path("config_create_gridmap"));

    gridmap_frame_id_ = config.param<std::string>("gridmap_param", "frame_id", "odom");
    grid_width_ = config.param<int>("gridmap_param", "width", 60);
    grid_height_ = config.param<int>("gridmap_param", "height", 60);
    gridmap_origin_x_ = config.param<float>("gridmap_param", "origin_x", -25.0F);
    gridmap_origin_y_ = config.param<float>("gridmap_param", "origin_y", -25.0F);
    resolution_ = config.param<float>("gridmap_param", "resolution", 1.0F);
    lower_bound_for_pt_z_ = config.param<float>("gridmap_param", "lower_bound_for_pt_z", 0.5F);
    upper_bound_for_pt_z_ = config.param<float>("gridmap_param", "upper_bound_for_pt_z", 1.5F);
    gridmap_data_ = std::vector<int>(grid_width_ * grid_height_, 0);
    gridmap_realtime_data_ = std::vector<int>(grid_width_ * grid_height_, 0);
    gridmap_submap_data_ = std::vector<int>(grid_width_ * grid_height_, 0);

    node_ = rclcpp::Node::make_shared("create_gridmap_new_node");

    gridmap_pub_ = node_->create_publisher<nav_msgs::msg::OccupancyGrid>("gridmap", 10);

    // 1. rosにon_new_submapからPubできるかテストしてみる
    // 2. SubmapをGridmapにしてPubする
    // 3. Realtime点群、on_new_frameか、on_update_framesのどちらがいいのか検討してこれも実装

    SubMappingCallbacks::on_new_submap.add([this](const SubMap::ConstPtr& submap) {
      std::cout << console::blue;
      std::cout << boost::format("--- SubMapping::on_new_submap (thread:%d) ---") %
                     std::this_thread::get_id()
                << std::endl;
      std::cout << "id:" << submap->id << std::endl;
      std::cout << console::reset;
      on_new_submap(submap);
    });

    OdometryEstimationCallbacks::on_update_frames.add([this](const
    std::vector<EstimationFrame::ConstPtr>& frames) { std::cout << console::green; std::cout <<
    boost::format("--- OdometryEstimation::on_update_frames (thread:%d) ---") %
    std::this_thread::get_id() << std::endl; std::cout << "frames:" << frames.size() << std::endl;
      std::cout << console::reset;
      on_update_frames(frames);
    });
  }
  ~CreateGridmap() override = default;
  
  void on_new_submap(const SubMap::ConstPtr&  /*submap*/){
    logger_ -> info("New submap received");
    nav_msgs::msg::OccupancyGrid gridmap;
    gridmap.header.stamp = node_->get_clock()->now();
    gridmap.header.frame_id = "map";
    gridmap.info.resolution = 0.1;
    gridmap.info.width = 100;
    gridmap.info.height = 100;
    gridmap.info.origin.position.x = 0.0;
    gridmap.info.origin.position.y = 0.0;
    gridmap.info.origin.position.z = 0.0;
    gridmap.info.origin.orientation.x = 0.0;
    gridmap.info.origin.orientation.y = 0.0;
    gridmap.info.origin.orientation.z = 0.0;
    gridmap.info.origin.orientation.w = 1.0;
    gridmap.data.resize(gridmap.info.width * gridmap.info.height, 0);
    gridmap_pub_->publish(gridmap);
  }

  void on_update_frames(const std::vector<EstimationFrame::ConstPtr>& frames) {
    for (const auto& frame : frames) {

    }

  }

private:
  std::shared_ptr<spdlog::logger> logger_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr gridmap_pub_;

  std::vector<int> gridmap_data_;
  std::vector<int> gridmap_realtime_data_;
  std::vector<int> gridmap_submap_data_;
  int grid_width_;          // gridmapの幅。gridmapのセルがいくつ横にあるか
  int grid_height_;         // gridmapの高さ。gridmapのセルがいくつ縦にあるか
  float gridmap_origin_x_;  // gridmapの原点[m]。ワールド座標系に対応する。
  // 原点はGridmapの左下の点。それがワールド座標系のどこに位置するか。
  float gridmap_origin_y_;  // gridmapの原点[m]。ワールド座標系に対応する。
  float resolution_;        // Gridmap 1セルの高さ(幅)。1セルは正方形。
  float lower_bound_for_pt_z_;  // Gridmapに入れる点の高さ方向のフィルタリング。最低高さ。
  float upper_bound_for_pt_z_;  // 最高高さ。
  std::string gridmap_frame_id_;
};

}  // namespace glim

extern "C" glim::ExtensionModule* create_extension_module() {
  return new glim::CreateGridmap();
}