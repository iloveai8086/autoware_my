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
#include <scene_module/stop_line/manager.h>

namespace
{
std::vector<lanelet::TrafficSignConstPtr> getTrafficSignRegElemsOnPath(
  const autoware_planning_msgs::PathWithLaneId & path, const lanelet::LaneletMapPtr lanelet_map)
{
  std::vector<lanelet::TrafficSignConstPtr> traffic_sign_reg_elems;

  for (const auto & p : path.points) {
    const auto lane_id = p.lane_ids.at(0);
    const auto ll = lanelet_map->laneletLayer.get(lane_id);

    const auto tss = ll.regulatoryElementsAs<const lanelet::TrafficSign>();
    for (const auto & ts : tss) {
      traffic_sign_reg_elems.push_back(ts);
    }
  }

  return traffic_sign_reg_elems;
}

std::vector<lanelet::ConstLineString3d> getStopLinesOnPath(
  const autoware_planning_msgs::PathWithLaneId & path, const lanelet::LaneletMapPtr lanelet_map)
{
  std::vector<lanelet::ConstLineString3d> stop_lines;

  for (const auto & traffic_sign_reg_elem : getTrafficSignRegElemsOnPath(path, lanelet_map)) {
    // Is stop sign?
    if (traffic_sign_reg_elem->type() != "stop_sign") {
      continue;
    }

    for (const auto & stop_line : traffic_sign_reg_elem->refLines()) {
      stop_lines.push_back(stop_line);
    }
  }

  return stop_lines;
}

std::set<int64_t> getStopLineIdSetOnPath(
  const autoware_planning_msgs::PathWithLaneId & path, const lanelet::LaneletMapPtr lanelet_map)
{
  std::set<int64_t> stop_line_id_set;

  for (const auto & stop_line : getStopLinesOnPath(path, lanelet_map)) {
    stop_line_id_set.insert(stop_line.id());
  }

  return stop_line_id_set;
}

}  // namespace

void StopLineModuleManager::launchNewModules(const autoware_planning_msgs::PathWithLaneId & path)
{
  for (const auto & stop_line : getStopLinesOnPath(path, planner_data_->lanelet_map)) {
    const auto module_id = stop_line.id();
    if (!isModuleRegistered(module_id)) {
      registerModule(std::make_shared<StopLineModule>(module_id, stop_line));
    }
  }
}

std::function<bool(const std::shared_ptr<SceneModuleInterface> &)>
StopLineModuleManager::getModuleExpiredFunction(const autoware_planning_msgs::PathWithLaneId & path)
{
  const auto stop_line_id_set = getStopLineIdSetOnPath(path, planner_data_->lanelet_map);

  return [stop_line_id_set](const std::shared_ptr<SceneModuleInterface> & scene_module) {
    return stop_line_id_set.count(scene_module->getModuleId()) == 0;
  };
}
