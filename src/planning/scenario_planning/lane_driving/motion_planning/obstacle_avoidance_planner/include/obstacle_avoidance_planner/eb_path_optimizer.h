/*
 * Copyright 2020 Tier IV, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef EB_PATH_OPTIMIZER_H
#define EB_PATH_OPTIMIZER_H

#include <Eigen/Core>

namespace geometry_msgs
{
ROS_DECLARE_MESSAGE(Pose);
ROS_DECLARE_MESSAGE(Point);
}  // namespace geometry_msgs
namespace autoware_planning_msgs
{
ROS_DECLARE_MESSAGE(Path);
ROS_DECLARE_MESSAGE(PathPoint);
ROS_DECLARE_MESSAGE(TrajectoryPoint);
}  // namespace autoware_planning_msgs

namespace autoware_perception_msgs
{
ROS_DECLARE_MESSAGE(DynamicObject);
}

namespace nav_msgs
{
ROS_DECLARE_MESSAGE(MapMetaData);
}

struct ConstrainLines
{
  double x_coef;
  double y_coef;
  double lower_bound;
  double upper_bound;
};

struct Constrain
{
  ConstrainLines top_and_bottom;
  ConstrainLines left_and_right;
};

struct Rectangle
{
  geometry_msgs::Point top_left;
  geometry_msgs::Point top_right;
  geometry_msgs::Point bottom_left;
  geometry_msgs::Point bottom_right;
};

struct OccupancyMaps
{
  std::vector<std::vector<int>> object_occupancy_map;
  std::vector<std::vector<int>> road_occupancy_map;
};

namespace osqp
{
class OSQPInterface;
}

struct ConstrainRectangle
{
  geometry_msgs::Point top_left;
  geometry_msgs::Point top_right;
  geometry_msgs::Point bottom_left;
  geometry_msgs::Point bottom_right;
  bool is_dynamic_joint_rectangle = false;
};

struct ConstrainRectangles
{
  ConstrainRectangle object_constrain_rectangle;
  ConstrainRectangle road_constrain_rectangle;
};

struct CandidatePoints
{
  std::vector<geometry_msgs::Pose> fixed_points;
  std::vector<geometry_msgs::Pose> non_fixed_points;
  int begin_path_idx;
  int end_path_idx;
};

struct QPParam
{
  int max_iteration;
  double eps_abs;
  double eps_rel;
  double eps_abs_for_extending;
  double eps_rel_for_extending;
};

struct TrajectoryParam
{
  int num_sampling_points;
  int num_joint_buffer_points;
  int num_offset_for_begin_idx;
  int num_fix_points_for_extending;
  double delta_arc_length_for_optimization;
  double delta_arc_length_for_trajectory;
  double delta_yaw_threshold_for_closest_point;
  double delta_yaw_threshold_for_straight;
  double trajectory_length;
  double forward_fixing_distance;
  double backward_fixing_distance;
};

struct ConstrainParam
{
  double clearance_for_fixing;
  double clearance_for_straight_line;
  double clearance_for_joint;
  double clearance_for_only_smoothing;
  double clearance_from_object_for_straight;
  double min_clearance_from_road;
  double min_clearance_from_object;
  double min_lat_constrain_space;
  double min_lon_constrain_space;
  double max_x_constrain_search_range;
  double max_avoiding_objects_velocity_ms;
  double coef_x_cosntrain_search_resolution;
  double coef_y_cosntrain_search_resolution;
  double keep_space_shape_x;
  double keep_space_shape_y;
};

struct DebugData
{
  std::vector<geometry_msgs::Point> interpolated_points;
  std::vector<geometry_msgs::Point> straight_points;
  std::vector<geometry_msgs::Pose> fixed_points;
  std::vector<geometry_msgs::Pose> non_fixed_points;
  std::vector<ConstrainRectangle> constrain_rectangles;
  cv::Mat clearance_map;
  cv::Mat only_object_clearance_map;
};

class EBPathOptimizer
{
private:
  const bool is_showing_debug_info_;
  const double epsilon_;

  const QPParam qp_param_;
  const TrajectoryParam traj_param_;
  const ConstrainParam constrain_param_;

  Eigen::MatrixXd default_a_matrix_;
  std::unique_ptr<geometry_msgs::Vector3> keep_space_shape_ptr_;
  std::unique_ptr<osqp::OSQPInterface> osqp_solver_ptr_;

  void initializeSolver();

  Eigen::MatrixXd makePMatrix();

  Eigen::MatrixXd makeAMatrix();

  geometry_msgs::Pose getOriginPose(
    const std::vector<geometry_msgs::Point> & interpolated_points, const int interpolated_idx,
    const std::vector<autoware_planning_msgs::PathPoint> & path_points);

  std::shared_ptr<std::vector<std::vector<geometry_msgs::Point>>> getOccupancyPoints(
    const geometry_msgs::Pose & origin, const cv::Mat & clearance_map,
    const nav_msgs::MapMetaData & map_info);

  Constrain getConstrainFromConstrainRectangle(
    const geometry_msgs::Point & interpolated_point, const ConstrainRectangle & constrain_range);

  ConstrainLines getConstrainLines(
    const double dx, const double dy, const geometry_msgs::Point & point,
    const geometry_msgs::Point & oppsite_point);

  ConstrainRectangles getConstrainRectangles(
    const geometry_msgs::Pose & origin_pose, const cv::Mat & clearance_map,
    const cv::Mat & only_objects_clearance_map, const nav_msgs::MapMetaData & map_info);

  ConstrainRectangle getConstrainRectangle(
    const geometry_msgs::Pose & origin_pose, const double clearance);

  ConstrainRectangle getConstrainRectangle(
    const std::vector<std::vector<int>> & occupancy_map,
    const std::vector<std::vector<geometry_msgs::Point>> & occupancy_points,
    const geometry_msgs::Pose & origin_pose);

  ConstrainRectangle getConstrainRectangle(
    const std::vector<autoware_planning_msgs::PathPoint> & path_points,
    const geometry_msgs::Pose & origin_pose, const cv::Mat & clearance_map,
    const nav_msgs::MapMetaData & map_info);

  OccupancyMaps getOccupancyMaps(
    const std::vector<std::vector<geometry_msgs::Point>> & occupancy_points,
    const geometry_msgs::Pose & origin_pose, const geometry_msgs::Point & origin_point_in_image,
    const cv::Mat & clearance_map, const cv::Mat & only_objects_clearance_map,
    const nav_msgs::MapMetaData & map_info);

  int getStraightLineIdx(
    const std::vector<geometry_msgs::Point> & interpolated_points, const int farrest_point_idx,
    const cv::Mat & only_objects_clearance, const nav_msgs::MapMetaData & map_info,
    std::vector<geometry_msgs::Point> & debug_detected_straight_points);

  int getEndPathIdx(
    const std::vector<autoware_planning_msgs::PathPoint> & path_points, const int begin_path_idx,
    const double required_trajectory_length);

  int getEndPathIdxInsideArea(
    const std::vector<autoware_planning_msgs::PathPoint> & path_points, const int begin_path_idx,
    const int end_path_idx, const cv::Mat & drivable_area, const nav_msgs::MapMetaData & map_info);

  std::vector<ConstrainRectangle> getPostProcessedConstrainRectangles(
    const bool enable_avoidance, const std::vector<ConstrainRectangle> & object_constrains,
    const std::vector<ConstrainRectangle> & road_constrains,
    const std::vector<ConstrainRectangle> & only_smooth_constrains,
    const std::vector<geometry_msgs::Point> & interpolated_points,
    const std::vector<autoware_planning_msgs::PathPoint> & path_points, const int farrest_point_idx,
    const int num_fixed_points, const int straight_idx);

  std::vector<ConstrainRectangle> getConstrainRectangleVec(
    const bool enable_avoidance, const autoware_planning_msgs::Path & input_path,
    const std::vector<geometry_msgs::Point> & interpolated_points, const int num_fixed_points,
    const int farrest_point_idx, const int straight_idx, const cv::Mat & clearnce_map,
    const cv::Mat & only_objects_clearance_map);

  std::vector<ConstrainRectangle> getConstrainRectangleVec(
    const std::vector<autoware_planning_msgs::PathPoint> & input_path,
    const std::vector<geometry_msgs::Point> & interpolated_points, const int num_fixed_points,
    const int farrest_point_idx);

  Rectangle getRelShapeRectangle(
    const geometry_msgs::Vector3 & vehicle_shape, const geometry_msgs::Pose & origin);

  Rectangle getAbsShapeRectangle(
    const Rectangle & rel_shape_rectangle_points, const geometry_msgs::Point & offset_point,
    const geometry_msgs::Pose & origin);

  std::vector<geometry_msgs::Point> getPaddedInterpolatedPoints(
    const std::vector<geometry_msgs::Point> & interpolated_points, const int farrest_idx);

  int getNumFixiedPoints(
    const std::vector<geometry_msgs::Pose> & fixed_points,
    const std::vector<geometry_msgs::Point> & interpolated_points, const int farrest_idx);

  std::vector<autoware_planning_msgs::TrajectoryPoint> getOptimizedTrajectory(
    const bool enable_avoidance, const autoware_planning_msgs::Path & path,
    const CandidatePoints & candidate_points, const cv::Mat & clearance_map,
    const cv::Mat & only_objects_clearance_map, DebugData & debug_data);

  std::vector<autoware_planning_msgs::TrajectoryPoint> getExtendedOptimizedTrajectory(
    const std::vector<autoware_planning_msgs::PathPoint> & path_points,
    const std::vector<autoware_planning_msgs::TrajectoryPoint> & optimized_points,
    const CandidatePoints & candidate_points);

  double getArcLength(
    const std::vector<autoware_planning_msgs::TrajectoryPoint> & optimized_points);

  void updateConstrain(
    const std::vector<geometry_msgs::Point> & interpolated_points,
    const std::vector<ConstrainRectangle> & rectangle_points);

  std::vector<autoware_planning_msgs::TrajectoryPoint> convertOptimizedPointsToTrajectory(
    const std::vector<double> optimzied_points, const int farrest_idx);

  std::vector<geometry_msgs::Pose> getFixedPoints(
    const geometry_msgs::Pose & ego_pose,
    const std::vector<autoware_planning_msgs::PathPoint> & path_points,
    const std::unique_ptr<std::vector<autoware_planning_msgs::TrajectoryPoint>> &
      prev_optimized_points,
    const cv::Mat & drivable_area, const nav_msgs::MapMetaData & map_info);

  CandidatePoints getCandidatePoints(
    const geometry_msgs::Pose & ego_pose,
    const std::vector<autoware_planning_msgs::PathPoint> & path_points,
    const std::unique_ptr<std::vector<autoware_planning_msgs::TrajectoryPoint>> &
      prev_optimized_points,
    const cv::Mat & drivable_area, const nav_msgs::MapMetaData & map_info);

  bool isPointInsideDrivableArea(
    const geometry_msgs::Point & point, const cv::Mat & drivable_area,
    const nav_msgs::MapMetaData & map_info);

  CandidatePoints getDefaultCandidatePoints(
    const std::vector<autoware_planning_msgs::PathPoint> & path_points);

  std::vector<double> solveQP(const bool is_extending);

  bool isFixingPathPoint(const std::vector<autoware_planning_msgs::PathPoint> & path_points);

public:
  EBPathOptimizer(
    const bool is_showing_debug_info, const QPParam qp_param, const TrajectoryParam traj_param,
    const ConstrainParam constrain_param);

  ~EBPathOptimizer();

  std::vector<autoware_planning_msgs::TrajectoryPoint> generateOptimizedTrajectory(
    const bool enable_avoidance, const geometry_msgs::Pose & ego_pose,
    const autoware_planning_msgs::Path & path,
    const std::unique_ptr<std::vector<autoware_planning_msgs::TrajectoryPoint>> &
      prev_optimized_points,
    const std::vector<autoware_perception_msgs::DynamicObject> & objects, DebugData & debug_data);
};

#endif
