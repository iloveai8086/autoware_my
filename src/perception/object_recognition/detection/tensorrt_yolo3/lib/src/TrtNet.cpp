/*
 * MIT License
 * 
 * Copyright (c) 2018 lewes6369
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
*/
#include "TrtNet.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include <string.h>
#include <time.h>
#include <cassert>
#include <chrono>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include "EntroyCalibrator.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

static Tn::Logger gLogger;

#define RETURN_AND_LOG(ret, severity, message)                            \
  do {                                                                    \
    std::string error_message = "ssd_error_log: " + std::string(message); \
    gLogger.log(ILogger::Severity::k##severity, error_message.c_str());   \
    return (ret);                                                         \
  } while (0)

inline void * safeCudaMalloc(size_t memSize)
{
  void * deviceMem;
  CUDA_CHECK(cudaMalloc(&deviceMem, memSize));
  if (deviceMem == nullptr) {
    std::cerr << "Out of memory" << std::endl;
    exit(1);
  }
  return deviceMem;
}

inline int64_t volume(const nvinfer1::Dims & d)
{
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
  switch (t) {
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kINT8:
      return 1;
  }
  throw std::runtime_error("Invalid DataType.");
  return 0;
}

namespace Tn
{
trtNet::trtNet(
  const std::string & prototxt, const std::string & caffemodel,
  const std::vector<std::string> & outputNodesName,
  const std::vector<std::vector<float>> & calibratorData, RUN_MODE mode /*= RUN_MODE::FLOAT32*/)
: mTrtContext(nullptr),
  mTrtEngine(nullptr),
  mTrtRunTime(nullptr),
  mTrtRunMode(mode),
  mTrtInputCount(0),
  mTrtIterationTime(0)
{
  std::cout << "init plugin proto: " << prototxt << " caffemodel: " << caffemodel << std::endl;
  auto parser = createCaffeParser();

  const int maxBatchSize = 1;
  IHostMemory * trtModelStream{nullptr};

  Int8EntropyCalibrator * calibrator = nullptr;
  if (calibratorData.size() > 0) {
    auto endPos = prototxt.find_last_of(".");
    auto beginPos = prototxt.find_last_of('/') + 1;
    std::string calibratorName = prototxt.substr(beginPos, endPos - beginPos);
    std::cout << "create calibrator,Named:" << calibratorName << std::endl;
    calibrator = new Int8EntropyCalibrator(maxBatchSize, calibratorData, calibratorName);
  }

  PluginFactory pluginFactorySerialize;
  ICudaEngine * tmpEngine = loadModelAndCreateEngine(
    prototxt.c_str(), caffemodel.c_str(), maxBatchSize, parser, &pluginFactorySerialize, calibrator,
    trtModelStream, outputNodesName);
  assert(tmpEngine != nullptr);
  assert(trtModelStream != nullptr);
  if (calibrator) {
    delete calibrator;
    calibrator = nullptr;
  }
  tmpEngine->destroy();
  pluginFactorySerialize.destroyPlugin();

  mTrtRunTime = createInferRuntime(gLogger);
  assert(mTrtRunTime != nullptr);
  mTrtEngine = mTrtRunTime->deserializeCudaEngine(
    trtModelStream->data(), trtModelStream->size(), &mTrtPluginFactory);
  assert(mTrtEngine != nullptr);
  // Deserialize the engine.
  trtModelStream->destroy();

  InitEngine();
}

trtNet::trtNet(const std::string & engineFile)
: mTrtContext(nullptr),
  mTrtEngine(nullptr),
  mTrtRunTime(nullptr),
  mTrtRunMode(RUN_MODE::FLOAT32),
  mTrtInputCount(0),
  mTrtIterationTime(0)
{
  using namespace std;
  fstream file;

  file.open(engineFile, ios::binary | ios::in);
  if (!file.is_open()) {
    cout << "read engine file" << engineFile << " failed" << endl;
    return;
  }
  file.seekg(0, ios::end);
  int length = file.tellg();
  file.seekg(0, ios::beg);
  std::unique_ptr<char[]> data(new char[length]);
  file.read(data.get(), length);

  file.close();

  // std::cout << "*** deserializing" << std::endl;
  mTrtRunTime = createInferRuntime(gLogger);
  assert(mTrtRunTime != nullptr);
  mTrtEngine = mTrtRunTime->deserializeCudaEngine(data.get(), length, &mTrtPluginFactory);
  assert(mTrtEngine != nullptr);

  InitEngine();
  // std::cerr << "finised deserializing " << std::endl;
}

void trtNet::InitEngine()
{
  const int maxBatchSize = 1;
  mTrtContext = mTrtEngine->createExecutionContext();
  assert(mTrtContext != nullptr);
  mTrtContext->setProfiler(&mTrtProfiler);

  // Input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings()
  int nbBindings = mTrtEngine->getNbBindings();

  mTrtCudaBuffer.resize(nbBindings);
  mTrtBindBufferSize.resize(nbBindings);
  for (int i = 0; i < nbBindings; ++i) {
    Dims dims = mTrtEngine->getBindingDimensions(i);
    DataType dtype = mTrtEngine->getBindingDataType(i);
    int64_t totalSize = volume(dims) * maxBatchSize * getElementSize(dtype);
    mTrtBindBufferSize[i] = totalSize;
    mTrtCudaBuffer[i] = safeCudaMalloc(totalSize);
    if (mTrtEngine->bindingIsInput(i)) mTrtInputCount++;
  }

  CUDA_CHECK(cudaStreamCreate(&mTrtCudaStream));
}

nvinfer1::ICudaEngine * trtNet::loadModelAndCreateEngine(
  const char * deployFile, const char * modelFile, int maxBatchSize, ICaffeParser * parser,
  nvcaffeparser1::IPluginFactory * pluginFactory, IInt8Calibrator * calibrator,
  IHostMemory *& trtModelStream, const std::vector<std::string> & outputNodesName)
{
  // Create the builder
  IBuilder * builder = createInferBuilder(gLogger);
  IBuilderConfig * config = builder->createBuilderConfig();

  // Parse the model to populate the network, then set the outputs.
  INetworkDefinition * network = builder->createNetworkV2(0U);
  parser->setPluginFactory(pluginFactory);

  std::cout << "Begin parsing model..." << std::endl;
  const IBlobNameToTensor * blobNameToTensor =
    parser->parse(deployFile, modelFile, *network, nvinfer1::DataType::kFLOAT);
  if (!blobNameToTensor) RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
  std::cout << "End parsing model..." << std::endl;

  // specify which tensors are outputs
  for (auto & name : outputNodesName) {
    auto output = blobNameToTensor->find(name.c_str());
    assert(output != nullptr);
    if (output == nullptr) std::cout << "can not find output named " << name << std::endl;

    network->markOutput(*output);
  }

  // Build the engine.
  builder->setMaxBatchSize(maxBatchSize);
  config->setMaxWorkspaceSize(1 << 30);  // 1G
  if (mTrtRunMode == RUN_MODE::INT8) {
    std::cout << "setInt8Mode" << std::endl;
    if (!builder->platformHasFastInt8())
      std::cout << "Notice: the platform do not has fast for int8" << std::endl;
    config->setFlag(BuilderFlag::kINT8);
      Int8EntropyCalibrator2 *calibrator2 = new Int8EntropyCalibrator2(1, 416, 416,
                                                                      "/media/ros/A666B94D66B91F4D/ros/coco_calib/",
                                                                      "/media/ros/A666B94D66B91F4D/ros/int8calib.table","data");
    config->setInt8Calibrator(calibrator2);
  } else if (mTrtRunMode == RUN_MODE::FLOAT16) {
    std::cout << "setFp16Mode" << std::endl;
    if (!builder->platformHasFastFp16())
      std::cout << "Notice: the platform do not has fast for fp16" << std::endl;
    config->setFlag(BuilderFlag::kFP16);
  }

  std::cout << "Begin building engine..." << std::endl;
  ICudaEngine * engine = builder->buildEngineWithConfig(*network, *config);
  if (!engine) RETURN_AND_LOG(nullptr, ERROR, "Unable to create engine");
  std::cout << "End building engine..." << std::endl;

  // We don't need the network any more, and we can destroy the parser.
  network->destroy();
  parser->destroy();

  // Serialize the engine, then close everything down.
  trtModelStream = engine->serialize();

  builder->destroy();
  config->destroy();
  shutdownProtobufLibrary();
  return engine;
}

void trtNet::doInference(const void * inputData, void * outputData)
{
  static const int batchSize = 1;
  assert(mTrtInputCount == 1);

  // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
  int inputIndex = 0;
  CUDA_CHECK(cudaMemcpyAsync(
    mTrtCudaBuffer[inputIndex], inputData, mTrtBindBufferSize[inputIndex], cudaMemcpyHostToDevice,
    mTrtCudaStream));
  auto t_start = std::chrono::high_resolution_clock::now();
  mTrtContext->execute(batchSize, &mTrtCudaBuffer[inputIndex]);
  auto t_end = std::chrono::high_resolution_clock::now();
  float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();

  // std::cout << "Time taken for inference is " << total << " ms." << std::endl;

  for (size_t bindingIdx = mTrtInputCount; bindingIdx < mTrtBindBufferSize.size(); ++bindingIdx) {
    auto size = mTrtBindBufferSize[bindingIdx];
    CUDA_CHECK(cudaMemcpyAsync(
      outputData, mTrtCudaBuffer[bindingIdx], size, cudaMemcpyDeviceToHost, mTrtCudaStream));
    outputData = (char *)outputData + size;
  }

  mTrtIterationTime++;
}
}  // namespace Tn