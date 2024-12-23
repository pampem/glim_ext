#include <thread>
#include <iostream>
#include <boost/format.hpp>

#include <rclcpp/rclcpp.hpp>

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

namespace glim {

class CreateGridmap : public ExtensionModule {
public:
  CreateGridmap() : logger_(create_module_logger("create_gridmap")) {
    logger_->info("Starting create_gridmap...");

    glim::Config config(glim::GlobalConfigExt::get_config_path("config_create_gridmap"));

    gridmap_frame_id_ = config.param<std::string>("gridmap_param", "frame_id", "odom");
    grid_width_ = config.param<int>("gridmap_param", "width", 60);
    grid_height_ = config.param<int>("gridmap_param", "height", 60);
    gridmap_origin_x_ = config.param<float>("gridmap_param", "origin_x", -25.0F);
    gridmap_origin_y_ = config.param<float>("gridmap_param", "origin_y", -25.0F);
    gridmap_resolution_ = config.param<float>("gridmap_param", "resolution", 1.0F);
    lower_bound_for_pt_z_ = config.param<float>("gridmap_param", "lower_bound_for_pt_z", 0.5F);
    upper_bound_for_pt_z_ = config.param<float>("gridmap_param", "upper_bound_for_pt_z", 0.5F);
    gridmap_topic_name_ = config.param<std::string>("gridmap_param", "topic_name", "slam_gridmap");
    gridmap_data_ = std::vector<int>(grid_width_ * grid_height_, 0);
    gridmap_realtime_data_ = std::vector<int>(grid_width_ * grid_height_, 0);
    gridmap_submap_data_ = std::vector<int>(grid_width_ * grid_height_, 0);

    node_ = rclcpp::Node::make_shared("create_gridmap_new_node");

    gridmap_pub_ = node_->create_publisher<nav_msgs::msg::OccupancyGrid>(gridmap_topic_name_, 10);

    last_pub_time_ = node_->get_clock()->now();

    // on_update_frameが定期的に呼ばれるので、そこをTriggerにして処理することにする。

    SubMappingCallbacks::on_new_submap.add([this](const SubMap::ConstPtr& submap) {
      std::cout << console::blue;
      std::cout << boost::format("--- SubMapping::on_new_submap (thread:%d) ---") %
                     std::this_thread::get_id()
                << std::endl;
      std::cout << "id:" << submap->id << std::endl;
      std::cout << console::reset;
      on_new_submap(submap);
    });

    OdometryEstimationCallbacks::on_update_frames.add(
      [this](const std::vector<EstimationFrame::ConstPtr>& frames) {
        // std::cout << console::green;
        // std::cout << boost::format("--- OdometryEstimation::on_update_frames (thread:%d) ---") %
        //                std::this_thread::get_id()
        //           << std::endl;
        // std::cout << "frames:" << frames.size() << std::endl;
        // std::cout << console::reset;
        on_update_frames(frames);
      });
  }
  ~CreateGridmap() override = default;

  void on_new_submap(const SubMap::ConstPtr& submap) {
    logger_->info("New submap received");
    if(!is_submap_received_) {
      is_submap_received_ = true;
      // is_submap_received_ = trueにならないと、gridmapはPublishされない。
    }

    const auto& t_world_origin = submap->T_world_origin;
    for (size_t i = 0; i < submap->frame->size(); i++) {
      const Eigen::Vector4d& pt = submap->frame->points[i];
      Eigen::Vector4d pt_world = t_world_origin * pt;
      if (
        pt_world.z() >= (current_z_position_ - lower_bound_for_pt_z_) &&
        pt_world.z() <= (current_z_position_ + upper_bound_for_pt_z_)) {
        int x = static_cast<int>((pt_world.x() - gridmap_origin_x_) / gridmap_resolution_);
        int y = static_cast<int>((pt_world.y() - gridmap_origin_y_) / gridmap_resolution_);

        if (x >= 0 && x < grid_width_ && y >= 0 && y < grid_height_) {
          int index = y * grid_width_ + x;
          gridmap_submap_data_[index] = 100;
        }
      }
    }
  }

  void on_update_frames(const std::vector<EstimationFrame::ConstPtr>& frames) {
    std::fill(gridmap_realtime_data_.begin(), gridmap_realtime_data_.end(), 0);
    for (const auto& frame : frames) {
      const auto& t_world_sensor = frame->T_world_sensor();
      for (size_t i = 0; i < frame->frame->size(); i++) {
        const Eigen::Vector4d& pt = frame->frame->points[i];
        Eigen::Vector4d pt_world = t_world_sensor * pt;

        if (
          pt_world.z() >= (current_z_position_ - lower_bound_for_pt_z_) &&
          pt_world.z() <= (current_z_position_ + upper_bound_for_pt_z_)) {
          int x = static_cast<int>((pt_world.x() - gridmap_origin_x_) / gridmap_resolution_);
          int y = static_cast<int>((pt_world.y() - gridmap_origin_y_) / gridmap_resolution_);

          if (x >= 0 && x < grid_width_ && y >= 0 && y < grid_height_) {
            int index = y * grid_width_ + x;
            gridmap_realtime_data_[index] = 100;
          }
        }
      }
    }

    Eigen::Isometry3d t_world_sensor = frames.back()->T_world_sensor();
    current_z_position_ = t_world_sensor.translation().z();
    std::ostringstream oss;
    // oss << t_world_sensor.matrix();

    // logger_->info("T_world_sensor:\n{}", oss.str());
    logger_->info("z_position: {}", current_z_position_);

    // 前回のPubが1秒以上前であればPubする
    auto now = node_->get_clock()->now();
    if ((now - last_pub_time_).seconds() >= 1.0) {
      if(!is_submap_received_) {
        return;
      }
      nav_msgs::msg::OccupancyGrid gridmap;
      gridmap.header.stamp = node_->get_clock()->now();
      gridmap.header.frame_id = gridmap_frame_id_;
      gridmap.info.resolution = gridmap_resolution_;
      gridmap.info.width = grid_width_;
      gridmap.info.height = grid_height_;
      gridmap.info.origin.position.x = gridmap_origin_x_;
      gridmap.info.origin.position.y = gridmap_origin_y_;
      gridmap.info.origin.position.z = current_z_position_;
      gridmap.data.resize(gridmap.info.width * gridmap.info.height, 0);

      for (int y = 0; y < grid_height_; ++y) {
        for (int x = 0; x < grid_width_; ++x) {
          int index = y * grid_width_ + x;
          // gridmap.data[index] =
          //   std::max(gridmap_realtime_data_[index], gridmap_submap_data_[index]);
          gridmap.data[index] = gridmap_submap_data_[index];
        }
      }

      gridmap_pub_->publish(gridmap);
      last_pub_time_ = now;
    }
  }

private:
  std::shared_ptr<spdlog::logger> logger_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr gridmap_pub_;
  rclcpp::Time last_pub_time_;
  bool is_submap_received_ = false;

  std::vector<int> gridmap_data_;
  std::vector<int> gridmap_realtime_data_;
  std::vector<int> gridmap_submap_data_;
  int grid_width_;          // gridmapの幅。gridmapのセルがいくつ横にあるか
  int grid_height_;         // gridmapの高さ。gridmapのセルがいくつ縦にあるか
  float gridmap_origin_x_;  // gridmapの原点[m]。ワールド座標系に対応する。
  // 原点はGridmapの左下の点。それがワールド座標系のどこに位置するか。
  float gridmap_origin_y_;    // gridmapの原点[m]。ワールド座標系に対応する。
  float gridmap_resolution_;  // Gridmap 1セルの高さ(幅)。1セルは正方形。
  float lower_bound_for_pt_z_;  // Gridmapに入れる点の高さ方向のフィルタリング。最低高さ。
  float upper_bound_for_pt_z_;  // 最高高さ。
  float current_z_position_;
  std::string gridmap_frame_id_;
  std::string gridmap_topic_name_;
};

}  // namespace glim

extern "C" glim::ExtensionModule* create_extension_module() {
  return new glim::CreateGridmap();
}