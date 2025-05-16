/*
  Copyright (c) 2025 Masashi Izumita
  All rights reserved.
  Bayesian OGM – realtime‑only edition (no Submap)
*/

#include <glim/util/logging.hpp>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <boost/format.hpp>

#include <rclcpp/rclcpp.hpp>

#include <glim/odometry/callbacks.hpp>
#include <glim/util/extension_module.hpp>
#include <glim_ext/console_colors.hpp>
#include <glim_ext/util/config_ext.hpp>

#include <nav_msgs/msg/occupancy_grid.hpp>

namespace glim {

class CreateGridmapBayes : public ExtensionModule {
public:
  CreateGridmapBayes() : logger_(create_module_logger("glim_rog_bayes_rt")) {
    logger_->info("Starting realtime Bayesian gridmap …");

    // ── load params ───────────────────────────────────────────────────────
    glim::Config cfg(glim::GlobalConfigExt::get_config_path("config_glim_rog_map"));
    frame_id_ = cfg.param<std::string>("gridmap_param", "frame_id", "odom");
    w_ = cfg.param<int>("gridmap_param", "width", 60);
    h_ = cfg.param<int>("gridmap_param", "height", 60);
    origin_x_ = cfg.param<float>("gridmap_param", "origin_x", -25.F);
    origin_y_ = cfg.param<float>("gridmap_param", "origin_y", -25.F);
    res_ = cfg.param<float>("gridmap_param", "resolution", 1.F);
    z_lower_ = cfg.param<float>("gridmap_param", "lower_bound_for_pt_z", 0.5F);
    z_upper_ = cfg.param<float>("gridmap_param", "upper_bound_for_pt_z", 0.5F);
    topic_ = cfg.param<std::string>("gridmap_param", "topic_name", "slam_gridmap");
    pub_rate_ = cfg.param<float>("gridmap_param", "pub_rate", 10.F);  // Hz

    log_odds_.assign(w_ * h_, 0.F);  // Prior 0.5

    node_ = rclcpp::Node::make_shared("glim_rog_bayes_rt_node");
    pub_ = node_->create_publisher<nav_msgs::msg::OccupancyGrid>(topic_, rclcpp::QoS{10});
    last_pub_ = node_->get_clock()->now();

    // register frame callback only (no Submap)
    OdometryEstimationCallbacks::on_update_frames.add(
      [this](const std::vector<EstimationFrame::ConstPtr>& fs) { on_frames(fs); });
  }
  ~CreateGridmapBayes() override = default;

private:
  // ── Bayesian helper ─────────────────────────────────────────────────────
  inline void update_cell(int idx, bool occ) {
    constexpr float k_l_occ = 2.197224F;    // logit(0.9)
    constexpr float k_l_free = -0.847298F;  // logit(0.3)
    constexpr float k_l_clip = 4.59512F;    // |logit(0.99)|
    log_odds_[idx] = std::clamp(log_odds_[idx] + (occ ? k_l_occ : k_l_free), -k_l_clip, k_l_clip);
  }

  // ── Frame callback ──────────────────────────────────────────────────────
  void on_frames(const std::vector<EstimationFrame::ConstPtr>& frames) {
    if (frames.empty()) return;
    Eigen::Isometry3d t_ws = frames.back()->T_world_sensor();
    current_z_ = t_ws.translation().z();

    for (const auto& fr : frames) {
      Eigen::Isometry3d t = fr->T_world_sensor();
      Eigen::Vector3d sensor = t.translation();

      for (size_t i = 0; i < fr->frame->size(); ++i) {
        const Eigen::Vector4d& pt = fr->frame->points[i];
        Eigen::Vector4d pw = t * pt;
        if (pw.z() < current_z_ - z_lower_ || pw.z() > current_z_ + z_upper_) continue;
        int gx = static_cast<int>((pw.x() - origin_x_) / res_);
        int gy = static_cast<int>((pw.y() - origin_y_) / res_);
        if (gx < 0 || gx >= w_ || gy < 0 || gy >= h_) continue;
        // occupied cell
        update_cell(gy * w_ + gx, true);
        // freespace raycast
        int sx = static_cast<int>((sensor.x() - origin_x_) / res_);
        int sy = static_cast<int>((sensor.y() - origin_y_) / res_);
        int dx = std::abs(gx - sx);
        int dy = std::abs(gy - sy);
        int sx_step = (sx < gx) ? 1 : -1;
        int sy_step = (sy < gy) ? 1 : -1;
        int err = dx - dy;
        int x = sx;
        int y = sy;
        while (x != gx || y != gy) {
          update_cell(y * w_ + x, false);
          int e2 = 2 * err;
          if (e2 > -dy) {
            err -= dy;
            x += sx_step;
          }
          if (e2 < dx) {
            err += dx;
            y += sy_step;
          }
        }
      }
    }

    maybe_slide_window(t_ws.translation());
    publish_if_due();
  }

  // ── Publish helper ──────────────────────────────────────────────────────
  void publish_if_due() {
    auto now = node_->get_clock()->now();
    if ((now - last_pub_).seconds() < 1.0 / pub_rate_) return;
    nav_msgs::msg::OccupancyGrid grid;
    grid.header.stamp = now;
    grid.header.frame_id = frame_id_;
    grid.info.resolution = res_;
    grid.info.width = w_;
    grid.info.height = h_;
    grid.info.origin.position.x = origin_x_;
    grid.info.origin.position.y = origin_y_;
    grid.data.resize(w_ * h_, -1);
    for (size_t i = 0; i < log_odds_.size(); ++i) {
      float p = 1.F - 1.F / (1.F + std::exp(log_odds_[i]));
      if (std::fabs(log_odds_[i]) < 0.01F)
        grid.data[i] = -1;
      else
        grid.data[i] = static_cast<int8_t>(std::round(p * 100.F));
    }
    pub_->publish(grid);
    last_pub_ = now;
  }

  // ── Sliding window (ROG-style) ──────────────────────────────────────────
  void maybe_slide_window(const Eigen::Vector3d& pos) {
    const float thresh = res_ * w_ * 0.25F;  // 25% edge
    bool slid = false;
    while (pos.x() - origin_x_ > thresh) {
      origin_x_ += res_;
      slid = true;
    }
    while (origin_x_ - pos.x() > thresh) {
      origin_x_ -= res_;
      slid = true;
    }
    while (pos.y() - origin_y_ > thresh) {
      origin_y_ += res_;
      slid = true;
    }
    while (origin_y_ - pos.y() > thresh) {
      origin_y_ -= res_;
      slid = true;
    }
    if (slid) logger_->debug("slide window: origin ({},{})", origin_x_, origin_y_);
  }

  // ── members ─────────────────────────────────────────────────────────────
  std::shared_ptr<spdlog::logger> logger_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_;
  rclcpp::Time last_pub_;

  int w_, h_;
  float origin_x_, origin_y_, res_;
  float z_lower_, z_upper_;
  float current_z_ = 0.F;
  std::string frame_id_;
  std::string topic_;
  float pub_rate_;
  std::vector<float> log_odds_;
};

}  // namespace glim

extern "C" glim::ExtensionModule* create_extension_module() {
  return new glim::CreateGridmapBayes();
}
