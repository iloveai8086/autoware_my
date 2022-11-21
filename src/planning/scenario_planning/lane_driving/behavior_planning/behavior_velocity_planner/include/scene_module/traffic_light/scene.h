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
#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include <boost/assert.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/point_xy.hpp>

#define EIGEN_MPL2_ONLY
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ros/ros.h>

#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_extension/utility/query.h>
#include <lanelet2_routing/RoutingGraph.h>

#include <scene_module/scene_module_interface.h>

class TrafficLightModule : public SceneModuleInterface
{
public:
  enum class State { APPROARCH, GO_OUT };

  struct DebugData
  {
    double base_link2front;
    std::vector<std::tuple<
      std::shared_ptr<const lanelet::TrafficLight>, autoware_perception_msgs::TrafficLightState>>
      tl_state;  // TODO: replace tuple with struct
    std::vector<geometry_msgs::Pose> stop_poses;
    std::vector<geometry_msgs::Pose> judge_poses;
  };

public:
  TrafficLightModule(const int64_t module_id, const lanelet::TrafficLight & traffic_light_reg_elem);

  bool modifyPathVelocity(autoware_planning_msgs::PathWithLaneId * path) override;

  visualization_msgs::MarkerArray createDebugMarkerArray() override;

private:
  bool getBackwordPointFromBasePoint(
    const Eigen::Vector2d & line_point1, const Eigen::Vector2d & line_point2,
    const Eigen::Vector2d & base_point, const double backward_length,
    Eigen::Vector2d & output_point);

  bool insertTargetVelocityPoint(
    const autoware_planning_msgs::PathWithLaneId & input,
    const boost::geometry::model::linestring<boost::geometry::model::d2::point_xy<double>> &
      stop_line,
    const double & margin, const double & velocity,
    autoware_planning_msgs::PathWithLaneId & output);

  bool getHighestConfidenceTrafficLightState(
    lanelet::ConstLineStringsOrPolygons3d & traffic_lights,
    autoware_perception_msgs::TrafficLightState & highest_confidence_tl_state);

  bool createTargetPoint(
    const autoware_planning_msgs::PathWithLaneId & input,
    const boost::geometry::model::linestring<boost::geometry::model::d2::point_xy<double>> &
      stop_line,
    const double & margin, size_t & target_point_idx, Eigen::Vector2d & target_point);

  // Key Feature
  const lanelet::TrafficLight & traffic_light_reg_elem_;

  // State
  State state_;

  // Parameter
  const double stop_margin_ = 0.0;
  const double tl_state_timeout_ = 1.0;
  const double max_stop_acceleration_threshold_ = -5.0;

  // Debug
  DebugData debug_data_;
};
