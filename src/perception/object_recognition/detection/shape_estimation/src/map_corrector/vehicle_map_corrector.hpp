/*
 * Copyright 2018 Autoware Foundation. All rights reserved.
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
 *
 * v1.0 Yukihiro Saito
 */

#pragma once

#include "map_corrector_interface.hpp"

class VehicleMapCorrector : public MapCorrectorInterface
{
private:
  double rad_threshold_;
  bool use_rad_filter_;

public:
  VehicleMapCorrector(const bool use_rad_filter, const double rad_threshold = M_PI_2 / 2.0)
  : use_rad_filter_(use_rad_filter), rad_threshold_(rad_threshold){};

  ~VehicleMapCorrector(){};

  bool correct(
    const VectorMap & vector_map, const geometry_msgs::TransformStamped & transform_stamped,
    autoware_perception_msgs::Shape & shape_output, geometry_msgs::Pose & pose_output,
    bool & orientaion_output) override;
};