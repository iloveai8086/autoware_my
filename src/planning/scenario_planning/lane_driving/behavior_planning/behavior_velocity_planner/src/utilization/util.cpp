/*
 * Copyright 2015-2019 Autoware Foundation. All rights reserved.
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
 *
 * Author: Robin Karlsson
 */

#include "utilization/util.h"

namespace planning_utils
{
double normalizeEulerAngle(double euler)
{
  double res = euler;
  while (res > M_PI) {
    res -= (2.0 * M_PI);
  }
  while (res < -M_PI) {
    res += 2.0 * M_PI;
  }

  return res;
}

geometry_msgs::Quaternion getQuaternionFromYaw(double yaw)
{
  tf2::Quaternion q;
  q.setRPY(0, 0, yaw);
  return tf2::toMsg(q);
}

template <class T>
bool calcClosestIndex(
  const T & path, const geometry_msgs::Pose & pose, int & closest, double dist_thr,
  double angle_thr)
{
  double dist_squared_min = std::numeric_limits<double>::max();
  double yaw_pose = tf2::getYaw(pose.orientation);
  closest = -1;

  for (int i = 0; i < (int)path.points.size(); ++i) {
    const double dist_squared = calcSquaredDist2d(getPose(path, i), pose);

    /* check distance threshold */
    if (dist_squared > dist_thr * dist_thr) continue;

    /* check angle threshold */
    double yaw_i = tf2::getYaw(getPose(path, i).orientation);
    double yaw_diff = normalizeEulerAngle(yaw_pose - yaw_i);

    if (std::fabs(yaw_diff) > angle_thr) continue;

    if (dist_squared < dist_squared_min) {
      dist_squared_min = dist_squared;
      closest = i;
    }
  }

  return closest == -1 ? false : true;
}

template bool calcClosestIndex<autoware_planning_msgs::Trajectory>(
  const autoware_planning_msgs::Trajectory & path, const geometry_msgs::Pose & pose, int & closest,
  double dist_thr, double angle_thr);
template bool calcClosestIndex<autoware_planning_msgs::PathWithLaneId>(
  const autoware_planning_msgs::PathWithLaneId & path, const geometry_msgs::Pose & pose,
  int & closest, double dist_thr, double angle_thr);
template bool calcClosestIndex<autoware_planning_msgs::Path>(
  const autoware_planning_msgs::Path & path, const geometry_msgs::Pose & pose, int & closest,
  double dist_thr, double angle_thr);

geometry_msgs::Pose transformRelCoordinate2D(
  const geometry_msgs::Pose & target, const geometry_msgs::Pose & origin)
{
  // translation
  geometry_msgs::Point trans_p;
  trans_p.x = target.position.x - origin.position.x;
  trans_p.y = target.position.y - origin.position.y;

  // rotation (use inverse matrix of rotation)
  double yaw = tf2::getYaw(origin.orientation);

  geometry_msgs::Pose res;
  res.position.x = (std::cos(yaw) * trans_p.x) + (std::sin(yaw) * trans_p.y);
  res.position.y = ((-1.0) * std::sin(yaw) * trans_p.x) + (std::cos(yaw) * trans_p.y);
  res.position.z = target.position.z - origin.position.z;
  res.orientation = getQuaternionFromYaw(tf2::getYaw(target.orientation) - yaw);

  return res;
}

geometry_msgs::Pose transformAbsCoordinate2D(
  const geometry_msgs::Pose & relative, const geometry_msgs::Pose & origin)
{
  // rotation
  geometry_msgs::Point rot_p;
  double yaw = tf2::getYaw(origin.orientation);
  rot_p.x = (std::cos(yaw) * relative.position.x) + (-std::sin(yaw) * relative.position.y);
  rot_p.y = (std::sin(yaw) * relative.position.x) + (std::cos(yaw) * relative.position.y);

  // translation
  geometry_msgs::Pose absolute;
  absolute.position.x = rot_p.x + origin.position.x;
  absolute.position.y = rot_p.y + origin.position.y;
  absolute.position.z = relative.position.z + origin.position.z;
  absolute.orientation = getQuaternionFromYaw(tf2::getYaw(relative.orientation) + yaw);

  return absolute;
}

double calcJudgeLineDist(
  double velocity, double max_accel, double margin)  // TODO: also consider jerk
{
  double judge_line_dist = (velocity * velocity) / (2.0 * (-max_accel)) + margin;
  return judge_line_dist;
}

}  // namespace planning_utils
