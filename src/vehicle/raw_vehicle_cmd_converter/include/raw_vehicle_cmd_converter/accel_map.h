/*
 * Copyright 2017-2019 Autoware Foundation. All rights reserved.
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

#ifndef RAW_VEHICLE_CMD_CONVERTER_ACCEL_MAP_H
#define RAW_VEHICLE_CMD_CONVERTER_ACCEL_MAP_H

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <ros/ros.h>

#include "raw_vehicle_cmd_converter/csv_loader.h"
#include "raw_vehicle_cmd_converter/interpolate.h"

class AccelMap
{
public:
  AccelMap();
  ~AccelMap();

  bool readAccelMapFromCSV(std::string csv_path);
  bool getThrottle(double acc, double vel, double & throttle);

private:
  std::string vehicle_name_;
  std::vector<double> vel_index_;
  std::vector<double> throttle_index_;
  std::vector<std::vector<double>> accel_map_;
};

#endif
