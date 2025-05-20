/*
  Copyright (c) 2025 Masashi Izumita
  All rights reserved.
  Bayesian OGM – realtime-only edition (no Submap) + Incremental Inflation
*/

#include <glim/util/logging.hpp>
#include <memory>
#include <string>
#include <vector>
#include <queue>
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
    logger_->info("Starting realtime Bayesian gridmap with incremental inflation …");

    // ── load params ───────────────────────────────────────────────────────
    glim::Config cfg(glim::GlobalConfigExt::get_config_path("config_glim_rog_map"));
    frame_id_ = cfg.param<std::string>("gridmap_param", "frame_id", "odom");
    w_ = cfg.param<int>("gridmap_param", "width", 20);
    h_ = cfg.param<int>("gridmap_param", "height", 20);
    origin_x_ = cfg.param<float>("gridmap_param", "origin_x", -25.0F);
    origin_y_ = cfg.param<float>("gridmap_param", "origin_y", -25.0F);
    res_ = cfg.param<float>("gridmap_param", "resolution", 1.0F);
    inflation_r_ = cfg.param<float>("gridmap_param", "inflation_radius", 1.5F);  // [m]

    grid_z_center_ = cfg.param<float>("gridmap_param", "grid_z_center", 1.0F);      // [m]
    z_half_range_ = cfg.param<float>("gridmap_param", "grid_z_half_range", 0.25F);  // ±[m]
    z_min_ = grid_z_center_ - z_half_range_;
    z_max_ = grid_z_center_ + z_half_range_;

    topic_ = cfg.param<std::string>("gridmap_param", "topic_name", "slam_gridmap");
    pub_rate_ = cfg.param<float>("gridmap_param", "pub_rate", 1.0F);  // Hz

    logger_->info("Configured pub_rate = {} Hz", pub_rate_);

    cells_inflation_ = static_cast<int>(std::ceil(inflation_r_ / res_));
    precompute_offsets();

    log_odds_.assign(w_ * h_, 0.0F);  // Prior 0.5
    inflated_.assign(w_ * h_, 0);     // 0=not inflated, 100=inflated

    node_ = rclcpp::Node::make_shared("glim_rog_bayes_rt_node");
    pub_ = node_->create_publisher<nav_msgs::msg::OccupancyGrid>(topic_, rclcpp::QoS{10});
    last_pub_ = node_->get_clock()->now();

    OdometryEstimationCallbacks::on_update_frames.add(
      [this](const std::vector<EstimationFrame::ConstPtr>& fs) { on_frames(fs); });
  }
  ~CreateGridmapBayes() override = default;

private:
  // ── helper: log-odds ⇄ prob ─────────────────────────────────────────────
  static inline float prob_from_log(float L) { return 1.0F - 1.0F / (1.0F + std::exp(L)); }
  static inline bool is_occupied(float L) { return prob_from_log(L) > 0.65F; }

  // ── Bayesian update ─────────────────────────────────────────────────────
  inline void bayes_update(int idx, bool occ) {
    constexpr float l_occ = 2.197224F;    // logit(0.9)
    constexpr float l_free = -0.847298F;  // logit(0.3)
    constexpr float l_clip = 4.595120F;   // |logit(0.99)|

    bool was_occ = is_occupied(log_odds_[idx]);
    log_odds_[idx] = std::clamp(log_odds_[idx] + (occ ? l_occ : l_free), -l_clip, l_clip);
    bool now_occ = is_occupied(log_odds_[idx]);

    if (now_occ && !was_occ) rising_q_.push(idx);
    if (!now_occ && was_occ) falling_q_.push(idx);
  }

  // ── Incremental Inflation ───────────────────────────────────────────────
  void process_inflation() {
    // Rising queue – add inflated cells
    while (!rising_q_.empty()) {
      int center = rising_q_.front();
      rising_q_.pop();
      int cx = center % w_;
      int cy = center / w_;
      for (const auto& off : offsets_) {
        int nx = cx + off.first;
        int ny = cy + off.second;
        if (nx < 0 || nx >= w_ || ny < 0 || ny >= h_) continue;
        inflated_[ny * w_ + nx] = 100;
      }
    }
    // Falling queue – remove inflated cells no longer supported
    while (!falling_q_.empty()) {
      int center = falling_q_.front();
      falling_q_.pop();
      int cx = center % w_;
      int cy = center / w_;
      for (const auto& off : offsets_) {
        int nx = cx + off.first;
        int ny = cy + off.second;
        if (nx < 0 || nx >= w_ || ny < 0 || ny >= h_) continue;
        int nidx = ny * w_ + nx;

        if (!is_occupied(log_odds_[nidx])) {
          inflated_[nidx] = 0;
        }
      }
    }
  }

  void precompute_offsets() {
    offsets_.clear();
    for (int dy = -cells_inflation_; dy <= cells_inflation_; ++dy) {
      for (int dx = -cells_inflation_; dx <= cells_inflation_; ++dx) {
        if (dx * dx + dy * dy <= cells_inflation_ * cells_inflation_) {
          offsets_.emplace_back(dx, dy);
        }
      }
    }
  }

  // ── Frame callback ──────────────────────────────────────────────────────
  void on_frames(const std::vector<EstimationFrame::ConstPtr>& frames) {
    if (frames.empty()) return;

    const Eigen::Vector3d& pos_ws = frames.back()->T_world_sensor().translation();
    // 原点移動処理を削除

    for (const auto& fr : frames) {
      Eigen::Isometry3d t = fr->T_world_sensor();
      Eigen::Vector3d sens = t.translation();

      // センサ位置をグリッドに投影（整数）
      int sx = static_cast<int>((sens.x() - origin_x_) / res_);
      int sy = static_cast<int>((sens.y() - origin_y_) / res_);

      for (size_t i = 0; i < fr->frame->size(); ++i) {
        const Eigen::Vector4d& pt = fr->frame->points[i];
        Eigen::Vector4d pw = t * pt;

        // ★★ 固定高さフィルタ（1 m ± 0.25 m）★★
        if (pw.z() < z_min_ || pw.z() > z_max_) continue;

        int gx = static_cast<int>((pw.x() - origin_x_) / res_);
        int gy = static_cast<int>((pw.y() - origin_y_) / res_);
        if (gx < 0 || gx >= w_ || gy < 0 || gy >= h_) continue;

        // occupied cell
        bayes_update(gy * w_ + gx, true);

        // freespace raycast (Bresenham)
        int dx = std::abs(gx - sx);
        int dy = std::abs(gy - sy);
        int step_x = (sx < gx) ? 1 : -1;
        int step_y = (sy < gy) ? 1 : -1;
        int err = dx - dy;
        int x = sx;
        int y = sy;
        while (true) {
          int e2 = 2 * err;
          if (e2 > -dy) {
            err -= dy;
            x += step_x;
          }
          if (e2 < dx) {
            err += dx;
            y += step_y;
          }
          if (x == gx && y == gy) break;
          if (x < 0 || x >= w_ || y < 0 || y >= h_) continue;
          bayes_update(y * w_ + x, false);
        }
      }
    }

    process_inflation();
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
    grid.info.origin.position.z = grid_z_center_;  // ← 高さは固定
    grid.data.resize(w_ * h_, -1);

    for (size_t i = 0; i < log_odds_.size(); ++i) {
      if (is_occupied(log_odds_[i]))
        grid.data[i] = 100;
      else if (inflated_[i] != 0)
        grid.data[i] = inflated_[i];
      else if (std::fabs(log_odds_[i]) < 0.01F)
        grid.data[i] = -1;
      else
        grid.data[i] = static_cast<int8_t>(std::round(prob_from_log(log_odds_[i]) * 100.0F));
    }
    pub_->publish(grid);
    last_pub_ = now;
  }

  // ── members ─────────────────────────────────────────────────────────────
  std::shared_ptr<spdlog::logger> logger_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_;
  rclcpp::Time last_pub_;

  int w_, h_;
  float origin_x_, origin_y_, res_;
  float inflation_r_;
  int cells_inflation_;

  // ★ 高さフィルタ関連
  float grid_z_center_;
  float z_half_range_;
  float z_min_, z_max_;

  std::string frame_id_;
  std::string topic_;
  float pub_rate_;

  std::vector<float> log_odds_;
  std::vector<uint8_t> inflated_;
  std::queue<int> rising_q_, falling_q_;
  std::vector<std::pair<int, int>> offsets_;
};

}  // namespace glim

extern "C" glim::ExtensionModule* create_extension_module() {
  return new glim::CreateGridmapBayes();
}
