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
#include <set>

#include <autoware_planning_msgs/Path.h>
#include <autoware_planning_msgs/PathWithLaneId.h>

#include <behavior_velocity_planner/planner_data.h>

// Debug
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>

class SceneModuleInterface
{
public:
  explicit SceneModuleInterface(const int64_t module_id) : module_id_(module_id) {}
  virtual ~SceneModuleInterface() = default;

  virtual bool modifyPathVelocity(autoware_planning_msgs::PathWithLaneId * path) = 0;
  virtual visualization_msgs::MarkerArray createDebugMarkerArray() = 0;

  int64_t getModuleId() const { return module_id_; }
  void setPlannerData(const std::shared_ptr<const PlannerData> & planner_data)
  {
    planner_data_ = planner_data;
  }

protected:
  const int64_t module_id_;
  std::shared_ptr<const PlannerData> planner_data_;
};

class SceneModuleManagerInterface
{
public:
  SceneModuleManagerInterface(const char * module_name)
  {
    const auto ns = std::string("debug/") + module_name;
    pub_debug_ = private_nh_.advertise<visualization_msgs::MarkerArray>(ns, 20);
  }

  virtual ~SceneModuleManagerInterface() = default;

  virtual const char * getModuleName() = 0;

  void updateSceneModuleInstances(
    const std::shared_ptr<const PlannerData> & planner_data,
    const autoware_planning_msgs::PathWithLaneId & path)
  {
    planner_data_ = planner_data;

    launchNewModules(path);
    deleteExpiredModules(path);
  }

  void modifyPathVelocity(autoware_planning_msgs::PathWithLaneId * path)
  {
    visualization_msgs::MarkerArray debug_marker_array;

    for (const auto & scene_module : scene_modules_) {
      scene_module->setPlannerData(planner_data_);
      scene_module->modifyPathVelocity(path);

      for (const auto & marker : scene_module->createDebugMarkerArray().markers) {
        debug_marker_array.markers.push_back(marker);
      }
    }

    pub_debug_.publish(debug_marker_array);
  }

protected:
  virtual void launchNewModules(const autoware_planning_msgs::PathWithLaneId & path) = 0;

  virtual std::function<bool(const std::shared_ptr<SceneModuleInterface> &)>
  getModuleExpiredFunction(const autoware_planning_msgs::PathWithLaneId & path) = 0;

  void deleteExpiredModules(const autoware_planning_msgs::PathWithLaneId & path)
  {
    const auto isModuleExpired = getModuleExpiredFunction(path);

    // Copy container to avoid iterator corruption due to scene_modules_.erase() in unregisterModule()
    const auto copied_scene_modules = scene_modules_;

    for (const auto & scene_module : copied_scene_modules) {
      if (isModuleExpired(scene_module)) {
        unregisterModule(scene_module);
      }
    }
  }

  bool isModuleRegistered(const int64_t module_id)
  {
    return registered_module_id_set_.count(module_id) != 0;
  }

  void registerModule(const std::shared_ptr<SceneModuleInterface> & scene_module)
  {
    ROS_INFO("register task: module = %s, id = %lu", getModuleName(), scene_module->getModuleId());
    registered_module_id_set_.emplace(scene_module->getModuleId());
    scene_modules_.insert(scene_module);
  }

  void unregisterModule(const std::shared_ptr<SceneModuleInterface> & scene_module)
  {
    ROS_INFO(
      "unregister task: module = %s, id = %lu", getModuleName(), scene_module->getModuleId());
    registered_module_id_set_.erase(scene_module->getModuleId());
    scene_modules_.erase(scene_module);
  }

  std::set<std::shared_ptr<SceneModuleInterface>> scene_modules_;
  std::set<int64_t> registered_module_id_set_;

  std::shared_ptr<const PlannerData> planner_data_;

  // Debug
  ros::NodeHandle private_nh_{"~"};
  ros::Publisher pub_debug_;
};
