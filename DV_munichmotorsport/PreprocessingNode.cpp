// Copyright 2025 municHMotorsport e.V. <info@munichmotorsport.de>
// Default C++
#include "PreprocessingNode.hpp"
#include "utils/pointcloud_utils.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/detail/point_cloud2__struct.hpp>
#include <utility>
#include <vector>

std::pair<int, int> PreprocessingNode::get_grid_key(float x, float y) const {
  int gx = static_cast<int>(std::floor(x / grid_size_));
  int gy = static_cast<int>(std::floor(y / grid_size_));
  return std::make_pair(gx, gy);
}

PreprocessingNode::PreprocessingNode()
    : Node("preprocessing"), logger_(this->get_logger()),
      tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_),
      lidar_pitch_angle_(11.5f) {
  /* Init Class */
  RCLCPP_INFO(logger_, "HelloWorld - PreprocessingNode: v0.3");

  /* Declare ros2 class variables */
  this->declare_parameter(std::string(LIDAR_ANGLE_PARAM), 11.5);
  this->declare_parameter(std::string(LIDAR_Z_BOX_OFFSET_PARAM), -0.78);

  this->declare_parameter(std::string(GROUND_REMOVAL_ALGORITHM_PARAM), "tile_parallel_gr");
  this->declare_parameter(std::string(GROUND_REMOVAL_G_OFFSET_PARAM), 0.125);

  /* Initialize current algorithm */
  std::string initial_algorithm = this->get_parameter(std::string(GROUND_REMOVAL_ALGORITHM_PARAM)).as_string();
  current_algorithm_ = parse_algorithm_string(initial_algorithm);

  /* Get ros2 params */
  /* Set up parameter callback */
  auto parameter_callback = [this](const std::vector<rclcpp::Parameter>& parameters) -> rcl_interfaces::msg::SetParametersResult {
    RCLCPP_INFO(logger_, "Parameter callback called");
    this->on_parameter_change(parameters);
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    return result;
  };
  param_callback_handle_ = this->add_on_set_parameters_callback(parameter_callback);

  // Get lidar angle
  timer_ = this->create_wall_timer(
      std::chrono::milliseconds(500),
      std::bind(&PreprocessingNode::get_lidar_transform, this));

  /* Create ros2 publisher */
  nonground_publisher_ =
      this->create_publisher<msg::PointCloud2>(std::string(NONGROUND_PUB), 10);
  ground_publisher_ =
      this->create_publisher<msg::PointCloud2>(std::string(GROUND_PUB), 10);
  ground_debug_publisher_ = this->create_publisher<msg::PointCloud2>(
      std::string(GROUND_DEBUG_PUB), 10);

  /* Create ros2 subscriber */
  subscriber_ = this->create_subscription<msg::PointCloud2>(
      "rslidar_points", 10,
      std::bind(&PreprocessingNode::filter_rslidar_points_cb, this,
                std::placeholders::_1));
}

PreprocessingNode::~PreprocessingNode() {
  Stats stats = compute_stats(gr_timings_);

  RCLCPP_INFO(logger_, "=== Ground Removal Timing Stats ===");
  RCLCPP_INFO(logger_, "  Count:             %zu", stats.count);
  RCLCPP_INFO(logger_, "  Min:               %ld ms", stats.min);
  RCLCPP_INFO(logger_, "  Max:               %ld ms", stats.max);
  RCLCPP_INFO(logger_, "  Mean:              %.2f ms", stats.mean);
  RCLCPP_INFO(logger_, "  Median:            %ld ms", stats.median);
  RCLCPP_INFO(logger_,
              "  Mode:              %ld ms (most frequent exact value)",
              stats.mode);
  RCLCPP_INFO(logger_,
              "  Histogram Peak:    %ld ms ±5 ms (most common time range)",
              stats.peak_bin_center);

  RCLCPP_INFO(logger_, "=== Ground Removal Timing Stats ===");
}

void PreprocessingNode::get_lidar_transform() {
  try {
    auto transform_stamped =
        tf_buffer_.lookupTransform("vehicle", "lidar", tf2::TimePointZero);

    tf2::Quaternion q(transform_stamped.transform.rotation.x,
                      transform_stamped.transform.rotation.y,
                      transform_stamped.transform.rotation.z,
                      transform_stamped.transform.rotation.w);

    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

    lidar_pitch_angle_ = static_cast<float>(pitch * 180.0 / M_PI);

    RCLCPP_INFO(this->get_logger(), "Lidar Angle: %f", lidar_pitch_angle_);

    timer_->cancel(); // Stop the timer once successful

  } catch (const tf2::TransformException &ex) {
    // RCLCPP_WARN(this->get_logger(), "Transform not available: %s",
    // ex.what());
  }
}

void PreprocessingNode::crop_pcl_cloud(CloudPtr cloud) {
  /* Do a first rough ROI */
  gr::PointCloudROI pc_roi;
  pc_roi.minX = 0.0;
  pc_roi.maxX = 40.0;
  pc_roi.minY = -10;
  pc_roi.maxY = 10;
  pc_roi.minZ = -4;
  pc_roi.maxZ = 4;

  float lidarAngle =
      -M_PI / 180 * this->get_parameter("lidar_angle").as_double();

  // perform crop box filtering
  pcl::CropBox<pcl::PointXYZ> boxFilter;
  boxFilter.setInputCloud(cloud);
  boxFilter.setMin(Eigen::Vector4f(pc_roi.minX, pc_roi.minY, pc_roi.minZ, 1.0));
  boxFilter.setMax(Eigen::Vector4f(pc_roi.maxX, pc_roi.maxY, pc_roi.maxZ, 1.0));
  boxFilter.setRotation(Eigen::Vector3f(0, lidarAngle, 0));
  boxFilter.filter(*cloud);

  /* Remove LiDar side effects */
  std::vector<std::pair<float, float>> quad_xy = {
      {10.0f, -2.1f},  // bottom-left
      {10.0f, -2.6f},  // bottom-right
      {30.0f, -8.0f},  // top-right
      {30.0f, -5.4f}}; // top-left
  cropPointCloud2DPolygon(cloud, quad_xy);

  /* Very rough */
  // INFO: Can be used to tweak performance arround 2ms. Its not worth for now but maybe later. Will provide worse results
  // TODO: Allign pcl_rslidar data to fix wrong sensor overlap between channel 3 & 4.
  // pc_roi.minX = 11.0;
  // pc_roi.maxX = 35.0;
  // pc_roi.minY = -8.0;
  // pc_roi.maxY = -2.5;
  // pc_roi.minZ = -5;
  // pc_roi.maxZ = 5;
  //
  // boxFilter.setInputCloud(cloud);
  // boxFilter.setMin(Eigen::Vector4f(pc_roi.minX, pc_roi.minY,
  // pc_roi.minZ, 1.0)); boxFilter.setMax(Eigen::Vector4f(pc_roi.maxX,
  // pc_roi.maxY, pc_roi.maxZ, 1.0)); boxFilter.setRotation(Eigen::Vector3f(0,
  // lidarAngle, 0)); boxFilter.setNegative(true); boxFilter.filter(*cloud);
}

void PreprocessingNode::remove_car_simple(CloudPtr cloud) {
  /* Remove car with simple ROI */
  gr::PointCloudROI car_roi;
  car_roi.minX = 0.0;
  car_roi.maxX = 2.0;
  car_roi.minY = -0.8;
  car_roi.maxY = 0.8;
  car_roi.minZ = -5.0;
  car_roi.maxZ = 5.0;
  float lidarAngle =
      -M_PI / 180 * this->get_parameter("lidar_angle").as_double();

  // perform crop box filtering
  pcl::CropBox<pcl::PointXYZ> boxFilter;
  boxFilter.setInputCloud(cloud);
  boxFilter.setMin(
      Eigen::Vector4f(car_roi.minX, car_roi.minY, car_roi.minZ, 1.0));
  boxFilter.setMax(
      Eigen::Vector4f(car_roi.maxX, car_roi.maxY, car_roi.maxZ, 1.0));
  boxFilter.setRotation(Eigen::Vector3f(0, lidarAngle, 0));
  boxFilter.setNegative(true);

  boxFilter.filter(*cloud);
}

void PreprocessingNode::filter_rslidar_points_cb(
    const msg::PointCloud2::SharedPtr msg) {
  // RCLCPP_INFO(logger_, "Got PCL point cloud");

  // Convert the pointcloud2 message to a pcl::PointCloud object
  CloudPtr cloud(new pcl::PointCloud<PointT>());
  CloudPtr non_ground(new pcl::PointCloud<PointT>());
  CloudPtr non_ground_low_res(new pcl::PointCloud<PointT>());

  pcl::fromROSMsg(*msg, *cloud);

  auto t0 = std::chrono::steady_clock::now();

  // Set ROI
  crop_pcl_cloud(cloud);
  remove_car_simple(cloud);

  /* Remove ground */
  switch (current_algorithm_) {
    case GroundRemovalAlgorithm::SIMPLE:
      simple_gr(cloud, non_ground);
      break;
    case GroundRemovalAlgorithm::TILE_FAST:
      tile_fast_gr(cloud, non_ground);
      break;
    case GroundRemovalAlgorithm::NON_TILE_FAST:
      non_tile_fast_gr(cloud, non_ground);
      break;
    case GroundRemovalAlgorithm::TILE_PARALLEL:
      tile_parallel_gr(cloud, non_ground);
      break;
    default:
      tile_parallel_gr(cloud, non_ground);
      break;
  }

  auto t1 = std::chrono::steady_clock::now();
  int64_t dt =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
      RCLCPP_INFO(logger_, "GROUND_REMOVAL_LATENCY: %ld ms", dt);
      // rclcpp::Time now = this->now();
      // RCLCPP_INFO(logger_, "GROUND_REMOVAL_LATENCY: %ld ms at time %.3f", dt, now.seconds());
  gr_timings_.push_back(dt);
  // RCLCPP_INFO(logger_, "Ground Removal took: %" PRId64 "ms\n", dt);

  // Down sample
  non_ground_low_res->reserve(non_ground->size());
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud(non_ground);
  sor.setLeafSize(0.02f, 0.02f, 0.02f);
  sor.filter(*non_ground_low_res);

  /* Publish the filtered pointcloud on the output topic */
  // Convert pcls to a pointcloud2 message
  msg::PointCloud2 nonground_msg;
  msg::PointCloud2 ground_msg;
  pcl::toROSMsg(*non_ground_low_res, nonground_msg);
  pcl::toROSMsg(*cloud, ground_msg);
  // msg::PointCloud2 ground_msg;
  // pcl::toROSMsg(*ground, ground_msg);

  // Add meta data
  nonground_msg.header.frame_id = "lidar";
  nonground_msg.header.stamp = msg.get()->header.stamp;
  ground_msg.header.stamp = msg.get()->header.stamp;
  ground_msg.header.frame_id = "lidar";

  // Publish
  nonground_publisher_->publish(nonground_msg);
  ground_publisher_->publish(ground_msg);
}

void PreprocessingNode::simple_gr(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr nonground) {
  gr::PointCloudROI pc_roi;
  GroundRemoval gr;

  pc_roi.minX = 0.0;
  pc_roi.maxX = 25.0;
  pc_roi.minY = -4.0;
  pc_roi.maxY = 4.0;
  pc_roi.minZ = this->get_parameter("lidar_z_box_offset").as_double();
  pc_roi.maxZ = 0.0;
  pc_roi.lidarAngle = this->get_parameter("lidar_angle").as_double();

  // Remove the ground points from the pointcloud
  gr.ground_removal(cloud, pc_roi);

  pc_roi.minX = 0.0;
  pc_roi.maxX = 2.5;
  pc_roi.minY = -0.8;
  pc_roi.maxY = 0.8;
  pc_roi.minZ = -5.0;
  pc_roi.maxZ = 5.0;

  // Romove Car
  gr.vehicle_removal(cloud, pc_roi);

  // remove_ground_ransac(cloud, nonground, ground);
  nonground = cloud;
}


void PreprocessingNode::non_tile_fast_gr(CloudPtr cloud, CloudPtr nonground) {
  if (!cloud || cloud->empty()) {
    throw std::runtime_error("GroundRemoval: Given cloud pcl is null or empty. "
                             "Nothing to remove ground!");
  }

  if (!nonground || !nonground->empty()) {
    throw std::runtime_error("GroundRemoval: Given nonground pcl pointer is "
                             "null or non empty. Empty pcl object needed: "
                             "Function does not support force override logic!");
  }

  // float cam_angle_deg = 11.0;
  // float cam_angle_rad = cam_angle_deg * M_PI / 180.0;
  // float g_threshold = 0.75;
  float cam_angle_rad = this->lidar_pitch_angle_ * M_PI / 180.0;

  Eigen::Affine3f transform = Eigen::Affine3f::Identity();

  // Rotate points
  transform.rotate(Eigen::AngleAxisf(cam_angle_rad, Eigen::Vector3f::UnitY()));

  // Reserve enough memory
  nonground->reserve(cloud->size());

  // Transform cloud
  pcl::transformPointCloud(*cloud, *nonground, transform);
  float min_z = std::numeric_limits<float>::max();
  for (const auto &pt : nonground->points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
      continue;
    }
    if (pt.z < min_z) {
      min_z = pt.z;
    }
  }

  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud(nonground);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(min_z + this->ground_offset_, std::numeric_limits<float>::max());
  pass.filter(*nonground);

  // Inverse transform
  Eigen::Affine3f back_transform = transform.inverse();
  pcl::transformPointCloud(*nonground, *nonground, back_transform);
}

void PreprocessingNode::tile_parallel_gr(const CloudPtr cloud,
                                          CloudPtr nonground) {

  if (!cloud || cloud->empty()) {
    RCLCPP_WARN(logger_, "GroundRemoval: Given cloud pcl is null or empty. "
                         "Nothing to remove ground!");
    return;
  }

  if (!nonground || !nonground->empty()) {
    RCLCPP_WARN(logger_, "GroundRemoval: Given nonground pcl pointer is null "
                         "or non empty. Empty pcl object needed: Function does "
                         "not support force override logic!");
    return;
  }

  // Reserve enough memory
  nonground->reserve(cloud->size());

  float cam_angle_rad = this->lidar_pitch_angle_ * M_PI / 180.0;
  // float cam_angle_rad = 11.5 * M_PI / 180.0;
  // Rotate points
  rotatePointCloudY(cloud, nonground, cam_angle_rad);

  // float g_threshold = 0.125;

  /* Find min z per tile */
  float tile_size = 1.0f;
  // CloudPtr filtered(new CloudT());

  auto filter_ground = [&](int start_index, int end_index, CloudPtr filtered) {
    std::unordered_map<TileKey, float> tile_min_z;
    // std::unordered_map<TileKey, std::vector<float *>> tile_all_z;

    // Find Minimum-Z per tile
    for (int i = start_index; i <= end_index; i++) {
      auto &pt = nonground->points[i];
      // auto &pt = cloud->points[i];
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
        continue;

      TileKey key{static_cast<int>(std::floor(pt.x / tile_size)),
                  static_cast<int>(std::floor(pt.y / tile_size))};

      auto it = tile_min_z.find(key);
      if (it == tile_min_z.end()) {
        tile_min_z[key] = pt.z;
      } else if (pt.z < it->second) {
        it->second = pt.z;
      }
    }

    // Add non-ground points to filtered pcl
    for (int i = start_index; i <= end_index; i++) {
      auto &pt = nonground->points[i];
      // auto &pt = cloud->points[i];
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
        continue;

      TileKey key{static_cast<int>(std::floor(pt.x / tile_size)),
                  static_cast<int>(std::floor(pt.y / tile_size))};

      float g_threshold2 = this->ground_offset_; 
      if (key.ix > 4.0) {
        g_threshold2 = 0.09;
      } else if (key.ix > 6.0) {
        g_threshold2 = 0.082;
      } else if (key.ix > 8.0) {
        g_threshold2 = 0.075;
      } else if (key.ix > 7.0) {
        g_threshold2 = 0.05;
      }

      auto it = tile_min_z.find(key);
      if (it != tile_min_z.end() && pt.z > (it->second + g_threshold2)) {
        filtered->points.push_back(pt);
      }
    }
  };

  // Create Threads
  std::vector<std::thread> threads;
  std::vector<CloudPtr> sub_clouds;
  int pcl_points = (int)nonground->points.size();
  int intervals = 8;
  int base_size = pcl_points / intervals;
  int remainder = pcl_points % intervals;

  int start = 0;
  for (int i = 0; i < intervals; i++) {
    int size = base_size + (i < remainder ? 1 : 0);
    int end = start + size - 1;
    CloudPtr filtered(new CloudT());
    sub_clouds.push_back(filtered);
    threads.emplace_back(filter_ground, start, end, filtered);
    start = end + 1;
  }

  for (auto &t : threads)
    t.join();
  threads.clear(); // Clean all threads

  // Combine the clouds
  pcl::PointCloud<pcl::PointXYZ>::Ptr merged(
      new pcl::PointCloud<pcl::PointXYZ>());
  size_t size = 0;
  for (auto &sub : sub_clouds) {
    size += sub->size();
  }
  merged->reserve(size);

  // Merge subclouds and caculate centroids
  for (auto &sub : sub_clouds) {
    if (sub->points.size() > 1) {
    }
    merged->insert(merged->begin(), sub->begin(), sub->end());
  }

  // this->cone_pose_pub_->publish(output_msg);
  // Reserve enough memory
  if (!merged->empty()) {
    CloudPtr rotated(new pcl::PointCloud<pcl::PointXYZ>());
    rotatePointCloudY(merged, rotated, -cam_angle_rad);
    *nonground = *rotated;
  }
}

void PreprocessingNode::tile_fast_gr(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr nonground) {

  if (!cloud || cloud->empty()) {
    RCLCPP_WARN(logger_, "GroundRemoval: Given cloud pcl is null or empty. "
                         "Nothing to remove ground!");
    return;
  }

  if (!nonground || !nonground->empty()) {
    RCLCPP_WARN(logger_, "GroundRemoval: Given nonground pcl pointer is null "
                         "or non empty. Empty pcl object needed: Function does "
                         "not support force override logic!");
    return;
  }

  float cam_angle_rad = this->lidar_pitch_angle_ * M_PI / 180.0;
  float g_threshold = this->ground_offset_;

  // Reserve enough memory
  nonground->reserve(cloud->size());

  // Rotate points
  Eigen::Affine3f transform = Eigen::Affine3f::Identity();
  transform.rotate(Eigen::AngleAxisf(cam_angle_rad, Eigen::Vector3f::UnitY()));
  // Transform cloud
  pcl::transformPointCloud(*cloud, *nonground, transform);

  /* Find min z per tile */
  float tile_size = 1.0f;
  std::unordered_map<TileKey, float> tile_min_z;
  CloudPtr filtered(new CloudT());

  // Find Minimum-Z per tile
  for (auto &pt : nonground->points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
      continue;

    TileKey key{static_cast<int>(std::floor(pt.x / tile_size)),
                static_cast<int>(std::floor(pt.y / tile_size))};

    auto it = tile_min_z.find(key);
    if (it == tile_min_z.end()) {
      tile_min_z[key] = pt.z;
    } else if (pt.z < it->second) {
      it->second = pt.z;
    }
  }

  // Add non-ground points to filtered pcl
  for (auto &pt : nonground->points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
      continue;

    TileKey key{static_cast<int>(std::floor(pt.x / tile_size)),
                static_cast<int>(std::floor(pt.y / tile_size))};

    auto it = tile_min_z.find(key);
    if (it != tile_min_z.end() && pt.z > it->second + g_threshold) {
      filtered->points.push_back(pt);
    }
  }

  // Reserve enough memory
  if (!filtered->empty()) {
    // Inverse transform
    Eigen::Affine3f back_transform = transform.inverse();
    pcl::transformPointCloud(*filtered, *filtered, back_transform);

    *nonground = *filtered;
  }
}

void PreprocessingNode::on_parameter_change(const std::vector<rclcpp::Parameter>& parameters) {
  RCLCPP_INFO(logger_, "Parameter change callback triggered with %zu parameters", parameters.size());
  
  for (const auto& param : parameters) {
    RCLCPP_INFO(logger_, "Processing parameter: %s", param.get_name().c_str());
    
    if (param.get_name() == GROUND_REMOVAL_ALGORITHM_PARAM) {
      if (param.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
        GroundRemovalAlgorithm new_algorithm = parse_algorithm_string(param.as_string());
        if (new_algorithm != current_algorithm_) {
          current_algorithm_ = new_algorithm;
          RCLCPP_INFO(logger_, "Ground removal algorithm changed to: %s", param.as_string().c_str());
        } else {
          RCLCPP_INFO(logger_, "Algorithm unchanged: %s", param.as_string().c_str());
        }
      } else {
        RCLCPP_WARN(logger_, "Parameter type mismatch for %s", param.get_name().c_str());
      }
    }
  }
}

GroundRemovalAlgorithm PreprocessingNode::parse_algorithm_string(const std::string& algorithm_str) const {
  std::string lower_str = algorithm_str;
  std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);

  RCLCPP_INFO(this->get_logger(), "Change algorithm called with: %s", lower_str.c_str());
  
  // Support multiple string variants for each algorithm
  if (lower_str == "simple_gr" || lower_str == "simple") {
    return GroundRemovalAlgorithm::SIMPLE;
  } else if (lower_str == "tile_fast_gr" || lower_str == "tile_fast" || lower_str == "fast") {
    return GroundRemovalAlgorithm::TILE_FAST;
  } else if (lower_str == "non_tile_fast_gr" || lower_str == "non_tile_fast" || 
             lower_str == "nontile_fast" || lower_str == "non_tile") {
    return GroundRemovalAlgorithm::NON_TILE_FAST;
  } else if (lower_str == "tile_parallel_gr" || lower_str == "tile_parallel" || 
             lower_str == "parallel") {
    return GroundRemovalAlgorithm::TILE_PARALLEL;
  } else {
    RCLCPP_WARN(logger_, "Unknown ground removal algorithm: %s. Using default: tile_parallel_gr", 
                algorithm_str.c_str());
    return GroundRemovalAlgorithm::TILE_PARALLEL;
  }
}