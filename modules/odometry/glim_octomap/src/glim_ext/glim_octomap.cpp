/*
  Copyright (c) 2025 Masashi Izumita
  All rights reserved.
  3D Octomap generation for GLIM - realtime edition
*/

#include <glim/util/logging.hpp>
#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <boost/format.hpp>

#include <rclcpp/rclcpp.hpp>

#include <glim/odometry/callbacks.hpp>
#include <glim/util/extension_module.hpp>
#include <glim_ext/console_colors.hpp>
#include <glim_ext/util/config_ext.hpp>

#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <octomap_msgs/msg/octomap.hpp>
#include <octomap_msgs/conversions.h>

namespace glim {

class CreateOctomap : public ExtensionModule {
public:
  CreateOctomap() : logger_(create_module_logger("glim_octomap_rt")) {
    logger_->info("Starting realtime 3D Octomap generation...");

    // ── load params ───────────────────────────────────────────────────────
    glim::Config cfg(glim::GlobalConfigExt::get_config_path("config_glim_octomap"));
    frame_id_ = cfg.param<std::string>("octomap_param", "frame_id", "odom");
    resolution_ = cfg.param<float>("octomap_param", "resolution", 0.2F);
    max_range_ = cfg.param<float>("octomap_param", "max_range", 10.0F);

    min_x_ = cfg.param<float>("octomap_param", "min_x", -25.0F);
    max_x_ = cfg.param<float>("octomap_param", "max_x", 25.0F);
    min_y_ = cfg.param<float>("octomap_param", "min_y", -25.0F);
    max_y_ = cfg.param<float>("octomap_param", "max_y", 25.0F);
    min_z_ = cfg.param<float>("octomap_param", "min_z", -5.0F);
    max_z_ = cfg.param<float>("octomap_param", "max_z", 5.0F);

    prob_hit_ = cfg.param<float>("octomap_param", "prob_hit", 0.9F);
    prob_miss_ = cfg.param<float>("octomap_param", "prob_miss", 0.3F);
    clamping_thresh_min_ = cfg.param<float>("octomap_param", "clamping_thresh_min", 0.12F);
    clamping_thresh_max_ = cfg.param<float>("octomap_param", "clamping_thresh_max", 0.97F);

    topic_ = cfg.param<std::string>("octomap_param", "topic_name", "/slam_octomap");
    pub_rate_ = cfg.param<float>("octomap_param", "pub_rate", 5.0F);  // Hz

    logger_->info("Configured pub_rate = {} Hz, resolution = {} m", pub_rate_, resolution_);

    // Initialize Octomap
    octree_ = std::make_shared<octomap::OcTree>(resolution_);

    // Set probability parameters using the correct API from AbstractOccupancyOcTree
    octree_->setClampingThresMin(clamping_thresh_min_);
    octree_->setClampingThresMax(clamping_thresh_max_);
    octree_->setProbHit(prob_hit_);
    octree_->setProbMiss(prob_miss_);

    node_ = rclcpp::Node::make_shared("glim_octomap_rt_node");
    pub_ = node_->create_publisher<octomap_msgs::msg::Octomap>(topic_, rclcpp::QoS{10});
    last_pub_ = node_->get_clock()->now();

    OdometryEstimationCallbacks::on_update_frames.add(
      [this](const std::vector<EstimationFrame::ConstPtr>& fs) { on_frames(fs); });
  }
  ~CreateOctomap() override = default;

private:
  // ── Frame callback ──────────────────────────────────────────────────────
  void on_frames(const std::vector<EstimationFrame::ConstPtr>& frames) {
    if (frames.empty()) return;

    for (const auto& fr : frames) {
      Eigen::Isometry3d t = fr->T_world_sensor();
      Eigen::Vector3d sensor_origin = t.translation();

      // センサ位置の境界チェック
      if (!is_in_bounds(sensor_origin)) continue;

      octomap::point3d sensor_pos(sensor_origin.x(), sensor_origin.y(), sensor_origin.z());

      // 点群の処理
      octomap::Pointcloud cloud;
      for (size_t i = 0; i < fr->frame->size(); ++i) {
        const Eigen::Vector4d& pt = fr->frame->points[i];
        Eigen::Vector4d pw = t * pt;

        // 境界チェック
        if (!is_in_bounds(Eigen::Vector3d(pw.x(), pw.y(), pw.z()))) continue;

        // 距離チェック
        double distance = (sensor_origin - Eigen::Vector3d(pw.x(), pw.y(), pw.z())).norm();
        if (distance > max_range_) continue;

        cloud.push_back(octomap::point3d(pw.x(), pw.y(), pw.z()));
      }

      // Octomapの更新（レイキャスティング含む）
      if (cloud.size() > 0) {
        octree_->insertPointCloud(cloud, sensor_pos, max_range_);
      }
    }

    publish_if_due();
  }

  // ── Boundary check helper ──────────────────────────────────────────────
  bool is_in_bounds(const Eigen::Vector3d& point) const {
    return point.x() >= min_x_ && point.x() <= max_x_ && point.y() >= min_y_ &&
           point.y() <= max_y_ && point.z() >= min_z_ && point.z() <= max_z_;
  }

  // ── Publish helper ──────────────────────────────────────────────────────
  void publish_if_due() {
    auto now = node_->get_clock()->now();
    if ((now - last_pub_).seconds() < 1.0 / pub_rate_) return;

    // Octomapメッセージの作成
    octomap_msgs::msg::Octomap octomap_msg;

    // ヘッダー設定
    octomap_msg.header.stamp = now;
    octomap_msg.header.frame_id = frame_id_;

    // Octomapデータの変換
    if (octomap_msgs::binaryMapToMsg(*octree_, octomap_msg)) {
      pub_->publish(octomap_msg);
      last_pub_ = now;

      // 統計情報をログ出力
      size_t num_nodes = octree_->getNumLeafNodes();
      logger_->debug("Published octomap with {} leaf nodes", num_nodes);
    } else {
      logger_->warn("Failed to convert octomap to message");
    }
  }

  // ── members ─────────────────────────────────────────────────────────────
  std::shared_ptr<spdlog::logger> logger_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<octomap_msgs::msg::Octomap>::SharedPtr pub_;
  rclcpp::Time last_pub_;

  // Octomap
  std::shared_ptr<octomap::OcTree> octree_;

  // Parameters
  double resolution_;
  double max_range_;
  double min_x_, max_x_, min_y_, max_y_, min_z_, max_z_;
  double prob_hit_, prob_miss_;
  double clamping_thresh_min_, clamping_thresh_max_;

  std::string frame_id_;
  std::string topic_;
  double pub_rate_;
};

}  // namespace glim

extern "C" glim::ExtensionModule* create_extension_module() {
  return new glim::CreateOctomap();
}
