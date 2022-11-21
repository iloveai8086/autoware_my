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

/*
 *  Copyright (c) 2018, Nagoya University
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither the name of Autoware nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ********************/

#ifndef COSTMAP_GENERATOR_H
#define COSTMAP_GENERATOR_H

#include <memory>
#include <string>

#include <ros/ros.h>

#include <grid_map_msgs/GridMap.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include <grid_map_ros/grid_map_ros.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <autoware_lanelet2_msgs/MapBin.h>
#include <autoware_perception_msgs/DynamicObjectArray.h>
#include <autoware_planning_msgs/Scenario.h>
#include <costmap_generator/objects_to_costmap.h>
#include <costmap_generator/points_to_costmap.h>
#include <lanelet2_extension/utility/message_conversion.h>

class CostmapGenerator
{
public:
  CostmapGenerator();

private:
  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;

  bool use_objects_;
  bool use_points_;
  bool use_wayarea_;

  lanelet::LaneletMapPtr lanelet_map_;
  autoware_perception_msgs::DynamicObjectArray::ConstPtr objects_;
  sensor_msgs::PointCloud2::ConstPtr points_;

  std::string costmap_frame_;
  std::string vehicle_frame_;
  std::string map_frame_;

  double update_rate_;

  double grid_min_value_;
  double grid_max_value_;
  double grid_resolution_;
  double grid_length_x_;
  double grid_length_y_;
  double grid_position_x_;
  double grid_position_y_;

  double maximum_lidar_height_thres_;
  double minimum_lidar_height_thres_;

  double expand_polygon_size_;
  int size_of_expansion_kernel_;

  grid_map::GridMap costmap_;

  ros::Publisher pub_costmap_;
  ros::Publisher pub_occupancy_grid_;

  ros::Subscriber sub_waypoint_;
  ros::Subscriber sub_points_;
  ros::Subscriber sub_objects_;
  ros::Subscriber sub_lanelet_bin_map_;
  ros::Subscriber sub_scenario_;

  ros::Timer timer_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::vector<std::vector<geometry_msgs::Point>> area_points_;

  PointsToCostmap points2costmap_;
  ObjectsToCostmap objects2costmap_;

  autoware_planning_msgs::Scenario::ConstPtr scenario_;

  struct LayerName
  {
    static constexpr const char * objects = "objects";
    static constexpr const char * points = "points";
    static constexpr const char * wayarea = "wayarea";
    static constexpr const char * combined = "combined";
  };

  /// \brief wait for lanelet2 map to load and build routing graph
  void initLaneletMap();

  /// \brief callback for loading landlet2 map
  void onLaneletMapBin(const autoware_lanelet2_msgs::MapBin & msg);

  /// \brief callback for DynamicObjectArray
  /// \param[in] in_objects input DynamicObjectArray usually from prediction or perception
  /// component
  void onObjects(const autoware_perception_msgs::DynamicObjectArray::ConstPtr & msg);

  /// \brief callback for sensor_msgs::PointCloud2
  /// \param[in] in_points input sensot_msgs::PointCloud2. Assuming groud-fitered pointcloud
  /// by default
  void onPoints(const sensor_msgs::PointCloud2::ConstPtr & msg);

  void onScenario(const autoware_planning_msgs::Scenario::ConstPtr & msg);

  void onTimer(const ros::TimerEvent & event);

  /// \brief initialize gridmap parameters based on rosparam
  void initGridmap();

  /// \brief publish ros msg: grid_map::GridMap, and nav_msgs::OccupancyGrid
  /// \param[in] gridmap with calculated cost
  void publishCostmap(const grid_map::GridMap & costmap);

  /// \brief set area_points from lanelet polygons
  /// \param [in] input lanelet_map
  /// \param [out] calculated area_points of lanelet polygons
  void loadRoadAreasFromLaneletMap(
    const lanelet::LaneletMapPtr lanelet_map,
    std::vector<std::vector<geometry_msgs::Point>> * area_points);

  /// \brief set area_points from parking-area polygons
  /// \param [in] input lanelet_map
  /// \param [out] calculated area_points of lanelet polygons
  void loadParkingAreasFromLaneletMap(
    const lanelet::LaneletMapPtr lanelet_map,
    std::vector<std::vector<geometry_msgs::Point>> * area_points);

  /// \brief calculate cost from pointcloud data
  /// \param[in] in_points: subscribed pointcloud data
  grid_map::Matrix generatePointsCostmap(const pcl::PointCloud<pcl::PointXYZ>::Ptr & in_points);

  /// \brief calculate cost from DynamicObjectArray
  /// \param[in] in_objects: subscribed DynamicObjectArray
  grid_map::Matrix generateObjectsCostmap(
    const autoware_perception_msgs::DynamicObjectArray::ConstPtr & in_objects);

  /// \brief calculate cost from lanelet2 map
  grid_map::Matrix generateWayAreaCostmap();

  /// \brief calculate cost for final output
  grid_map::Matrix generateCombinedCostmap();
};

#endif  // COSTMAP_GENERATOR_H
