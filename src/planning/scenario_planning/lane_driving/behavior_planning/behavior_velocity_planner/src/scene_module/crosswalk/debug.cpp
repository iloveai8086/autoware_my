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
#include <scene_module/crosswalk/scene.h>

#include "utilization/marker_helper.h"
#include "utilization/util.h"

namespace
{
using DebugData = CrosswalkModule::DebugData;

visualization_msgs::MarkerArray createMarkers(const DebugData & debug_data)
{
  visualization_msgs::MarkerArray msg;
  ros::Time current_time = ros::Time::now();
  tf2::Transform tf_base_link2front(
    tf2::Quaternion(0.0, 0.0, 0.0, 1.0), tf2::Vector3(debug_data.base_link2front, 0.0, 0.0));

  // Crosswalk polygons
  for (size_t i = 0; i < debug_data.crosswalk_polygons.size(); ++i) {
    std::vector<Eigen::Vector3d> polygon = debug_data.crosswalk_polygons.at(i);

    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = current_time;

    marker.ns = "crosswalk polygon line";
    marker.id = i;
    marker.lifetime = ros::Duration(0.5);
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = 0;
    marker.pose.position.y = 0;
    marker.pose.position.z = 0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.1;
    marker.color.a = 0.999;  // Don't forget to set the alpha!
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    for (size_t j = 0; j < polygon.size(); ++j) {
      geometry_msgs::Point point;
      point.x = polygon.at(j).x();
      point.y = polygon.at(j).y();
      point.z = polygon.at(j).z();
      marker.points.push_back(point);
    }
    marker.points.push_back(marker.points.front());
    msg.markers.push_back(marker);

    marker.ns = "crosswalk polygon point";
    marker.id = i;
    marker.lifetime = ros::Duration(0.5);
    marker.type = visualization_msgs::Marker::POINTS;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = 0;
    marker.pose.position.y = 0;
    marker.pose.position.z = 0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.25;
    marker.scale.y = 0.25;
    marker.color.a = 0.999;  // Don't forget to set the alpha!
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    for (size_t j = 0; j < polygon.size(); ++j) {
      geometry_msgs::Point point;
      point.x = polygon.at(j).x();
      point.y = polygon.at(j).y();
      point.z = polygon.at(j).z();
      marker.points.push_back(point);
    }
    msg.markers.push_back(marker);
  }

  // Collision line
  for (size_t i = 0; i < debug_data.collision_lines.size(); ++i) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = current_time;
    marker.ns = "collision line";
    marker.id = i;
    marker.lifetime = ros::Duration(0.5);
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = 0;
    marker.pose.position.y = 0;
    marker.pose.position.z = 0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.1;
    marker.color.a = 0.999;  // Don't forget to set the alpha!
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    for (size_t j = 0; j < debug_data.collision_lines.at(i).size(); ++j) {
      geometry_msgs::Point point;
      point.x = debug_data.collision_lines.at(i).at(j).x();
      point.y = debug_data.collision_lines.at(i).at(j).y();
      point.z = debug_data.collision_lines.at(i).at(j).z();
      marker.points.push_back(point);
    }
    msg.markers.push_back(marker);
  }

  // Collision point
  if (!debug_data.collision_points.empty()) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = current_time;
    marker.ns = "collision point";
    marker.id = 0;
    marker.lifetime = ros::Duration(0.5);
    marker.type = visualization_msgs::Marker::POINTS;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = 0;
    marker.pose.position.y = 0;
    marker.pose.position.z = 0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.25;
    marker.scale.y = 0.25;
    marker.color.a = 0.999;  // Don't forget to set the alpha!
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    for (size_t j = 0; j < debug_data.collision_points.size(); ++j) {
      geometry_msgs::Point point;
      point.x = debug_data.collision_points.at(j).x();
      point.y = debug_data.collision_points.at(j).y();
      point.z = debug_data.collision_points.at(j).z();
      marker.points.push_back(point);
    }
    msg.markers.push_back(marker);
  }

  // Slow polygon
  for (size_t i = 0; i < debug_data.slow_polygons.size(); ++i) {
    std::vector<Eigen::Vector3d> polygon = debug_data.slow_polygons.at(i);

    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = current_time;

    marker.ns = "slow polygon line";
    marker.id = i;
    marker.lifetime = ros::Duration(0.5);
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = 0;
    marker.pose.position.y = 0;
    marker.pose.position.z = 0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.1;
    marker.color.a = 0.999;  // Don't forget to set the alpha!
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    for (size_t j = 0; j < polygon.size(); ++j) {
      geometry_msgs::Point point;
      point.x = polygon.at(j).x();
      point.y = polygon.at(j).y();
      point.z = polygon.at(j).z();
      marker.points.push_back(point);
    }
    marker.points.push_back(marker.points.front());
    msg.markers.push_back(marker);
  }

  // Slow point
  if (!debug_data.slow_poses.empty()) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = current_time;
    marker.ns = "slow point";
    marker.id = 0;
    marker.lifetime = ros::Duration(0.5);
    marker.type = visualization_msgs::Marker::POINTS;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = 0;
    marker.pose.position.y = 0;
    marker.pose.position.z = 0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.25;
    marker.scale.y = 0.25;
    marker.color.a = 0.999;  // Don't forget to set the alpha!
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    for (size_t j = 0; j < debug_data.slow_poses.size(); ++j) {
      geometry_msgs::Point point;
      point.x = debug_data.slow_poses.at(j).position.x;
      point.y = debug_data.slow_poses.at(j).position.y;
      point.z = debug_data.slow_poses.at(j).position.z;
      marker.points.push_back(point);
    }
    msg.markers.push_back(marker);
  }

  // Stop polygon
  for (size_t i = 0; i < debug_data.stop_polygons.size(); ++i) {
    std::vector<Eigen::Vector3d> polygon = debug_data.stop_polygons.at(i);

    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = current_time;

    marker.ns = "stop polygon line";
    marker.id = i;
    marker.lifetime = ros::Duration(0.5);
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = 0;
    marker.pose.position.y = 0;
    marker.pose.position.z = 0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.1;
    marker.color.a = 0.999;  // Don't forget to set the alpha!
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    for (size_t j = 0; j < polygon.size(); ++j) {
      geometry_msgs::Point point;
      point.x = polygon.at(j).x();
      point.y = polygon.at(j).y();
      point.z = polygon.at(j).z();
      marker.points.push_back(point);
    }
    marker.points.push_back(marker.points.front());
    msg.markers.push_back(marker);
  }

  // Stop point
  if (!debug_data.stop_poses.empty()) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = current_time;
    marker.ns = "stop point";
    marker.id = 0;
    marker.lifetime = ros::Duration(0.5);
    marker.type = visualization_msgs::Marker::POINTS;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = 0;
    marker.pose.position.y = 0;
    marker.pose.position.z = 0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.25;
    marker.scale.y = 0.25;
    marker.color.a = 0.999;  // Don't forget to set the alpha!
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    for (size_t j = 0; j < debug_data.stop_poses.size(); ++j) {
      geometry_msgs::Point point;
      point.x = debug_data.stop_poses.at(j).position.x;
      point.y = debug_data.stop_poses.at(j).position.y;
      point.z = debug_data.stop_poses.at(j).position.z;
      marker.points.push_back(point);
    }
    msg.markers.push_back(marker);
  }

  // Stop VirtualWall
  for (size_t j = 0; j < debug_data.stop_poses.size(); ++j) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = current_time;
    marker.ns = "stop_virtual_wall";
    marker.id = j;
    marker.lifetime = ros::Duration(0.5);
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;
    tf2::Transform tf_map2base_link;
    tf2::fromMsg(debug_data.stop_poses.at(j), tf_map2base_link);
    tf2::Transform tf_map2front = tf_map2base_link * tf_base_link2front;
    tf2::toMsg(tf_map2front, marker.pose);
    marker.pose.position.z += 1.0;
    marker.scale.x = 0.1;
    marker.scale.y = 5.0;
    marker.scale.z = 2.0;
    marker.color.a = 0.5;  // Don't forget to set the alpha!
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    msg.markers.push_back(marker);
  }
  // Factor Text
  for (size_t j = 0; j < debug_data.stop_poses.size(); ++j) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = current_time;
    marker.ns = "factor_text";
    marker.id = j;
    marker.lifetime = ros::Duration(0.5);
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::Marker::ADD;
    tf2::Transform tf_map2base_link;
    tf2::fromMsg(debug_data.stop_poses.at(j), tf_map2base_link);
    tf2::Transform tf_map2front = tf_map2base_link * tf_base_link2front;
    tf2::toMsg(tf_map2front, marker.pose);
    marker.pose.position.z += 2.0;
    marker.scale.x = 0.0;
    marker.scale.y = 0.0;
    marker.scale.z = 1.0;
    marker.color.a = 0.999;  // Don't forget to set the alpha!
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 1.0;
    marker.text = "crosswalk";
    msg.markers.push_back(marker);
  }

  // Slow VirtualWall
  for (size_t j = 0; j < debug_data.slow_poses.size(); ++j) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = current_time;
    marker.ns = "slow virtual_wall";
    marker.id = j;
    marker.lifetime = ros::Duration(0.5);
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;
    tf2::Transform tf_map2base_link;
    tf2::fromMsg(debug_data.slow_poses.at(j), tf_map2base_link);
    tf2::Transform tf_map2front = tf_map2base_link * tf_base_link2front;
    tf2::toMsg(tf_map2front, marker.pose);
    marker.pose.position.z += 1.0;
    marker.scale.x = 0.1;
    marker.scale.y = 5.0;
    marker.scale.z = 2.0;
    marker.color.a = 0.5;  // Don't forget to set the alpha!
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    msg.markers.push_back(marker);
  }
  // Slow Factor Text
  for (size_t j = 0; j < debug_data.slow_poses.size(); ++j) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = current_time;
    marker.ns = "slow factor_text";
    marker.id = j;
    marker.lifetime = ros::Duration(0.5);
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::Marker::ADD;
    tf2::Transform tf_map2base_link;
    tf2::fromMsg(debug_data.slow_poses.at(j), tf_map2base_link);
    tf2::Transform tf_map2front = tf_map2base_link * tf_base_link2front;
    tf2::toMsg(tf_map2front, marker.pose);
    marker.pose.position.z += 2.0;
    marker.scale.x = 0.0;
    marker.scale.y = 0.0;
    marker.scale.z = 1.0;
    marker.color.a = 0.999;  // Don't forget to set the alpha!
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 1.0;
    marker.text = "crosswalk";
    msg.markers.push_back(marker);
  }

  return msg;
}

}  // namespace

visualization_msgs::MarkerArray CrosswalkModule::createDebugMarkerArray()
{
  visualization_msgs::MarkerArray debug_marker_array;

  appendMarkerArray(createMarkers(debug_data_), &debug_marker_array);

  return debug_marker_array;
}
