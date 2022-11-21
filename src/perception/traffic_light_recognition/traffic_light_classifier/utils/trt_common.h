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

#include <stdio.h>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cudnn.h>

#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace Tn
{
class Logger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * msg) override
  {
    if (severity != Severity::kINFO) std::cout << msg << std::endl;
  }
};

struct InferDeleter
{
  template <typename T>
  void operator()(T * obj) const
  {
    if (obj) {
      obj->destroy();
    }
  }
};

class TrtCommon
{
public:
  TrtCommon(std::string model_path, std::string cache_dir, std::string precision);
  ~TrtCommon(){};

  bool loadEngine(std::string engine_file_path);
  bool buildEngineFromOnnx(std::string onnx_file_path, std::string output_engine_file_path);
  void setup();

  bool isInitialized();
  int getNumInput();
  int getNumOutput();
  int getInputBindingIndex();
  int getOutputBindingIndex();

  template <typename T>
  using UniquePtr = std::unique_ptr<T, InferDeleter>;
  UniquePtr<nvinfer1::IExecutionContext> context_;

private:
  Logger logger_;
  bool is_initialized_;
  size_t max_batch_size_;
  std::string model_file_path_;
  UniquePtr<nvinfer1::IRuntime> runtime_;
  UniquePtr<nvinfer1::ICudaEngine> engine_;

  nvinfer1::Dims input_dims_;
  nvinfer1::Dims output_dims_;
  std::string input_name_;
  std::string output_name_;
  std::string precision_;
  std::string cache_dir_;
};

}  // namespace Tn
