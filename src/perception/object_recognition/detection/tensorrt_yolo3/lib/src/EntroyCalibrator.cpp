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
#include "EntroyCalibrator.h"
#include <string.h>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iterator>
#include <opencv2/dnn/dnn.hpp>
#include "utils.h"
#include "cuda_utils.h"

namespace nvinfer1
{
Int8EntropyCalibrator::Int8EntropyCalibrator(
  int BatchSize, const std::vector<std::vector<float>> & data,
  const std::string & CalibDataName /*= ""*/, bool readCache /*= true*/)
: mCalibDataName(CalibDataName), mBatchSize(BatchSize), mReadCache(readCache)
{
  mDatas.reserve(data.size());
  mDatas = data;

  mInputCount = BatchSize * data[0].size();
  mCurBatchData = new float[mInputCount];
  mCurBatchIdx = 0;
  CUDA_CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
}

Int8EntropyCalibrator::~Int8EntropyCalibrator()
{
  CUDA_CHECK(cudaFree(mDeviceInput));
  if (mCurBatchData) delete[] mCurBatchData;
}

bool Int8EntropyCalibrator::getBatch(void * bindings[], const char * names[], int nbBindings)
{
  if (mCurBatchIdx + mBatchSize > int(mDatas.size())) return false;

  float * ptr = mCurBatchData;
  size_t imgSize = mInputCount / mBatchSize;
  auto iter = mDatas.begin() + mCurBatchIdx;

  std::for_each(iter, iter + mBatchSize, [=, &ptr](std::vector<float> & val) {
    assert(imgSize == val.size());
    memcpy(ptr, val.data(), imgSize * sizeof(float));

    ptr += imgSize;
  });

  CUDA_CHECK(
    cudaMemcpy(mDeviceInput, mCurBatchData, mInputCount * sizeof(float), cudaMemcpyHostToDevice));
  // std::cout << "input name " << names[0] << std::endl;
  bindings[0] = mDeviceInput;

  std::cout << "load batch " << mCurBatchIdx << " to " << mCurBatchIdx + mBatchSize - 1
            << std::endl;
  mCurBatchIdx += mBatchSize;
  return true;
}

const void * Int8EntropyCalibrator::readCalibrationCache(size_t & length)
{
  mCalibrationCache.clear();
  std::ifstream input(mCalibDataName + ".calib", std::ios::binary);
  input >> std::noskipws;
  if (mReadCache && input.good())
    std::copy(
      std::istream_iterator<char>(input), std::istream_iterator<char>(),
      std::back_inserter(mCalibrationCache));

  length = mCalibrationCache.size();
  return length ? &mCalibrationCache[0] : nullptr;
}

void Int8EntropyCalibrator::writeCalibrationCache(const void * cache, size_t length)
{
  std::ofstream output(mCalibDataName + ".calib", std::ios::binary);
  output.write(reinterpret_cast<const char *>(cache), length);
}




    Int8EntropyCalibrator2::Int8EntropyCalibrator2(int batchsize, int input_w, int input_h, const char* img_dir, const char* calib_table_name, const char* input_blob_name, bool read_cache)
            : batchsize_(batchsize)
            , input_w_(input_w)
            , input_h_(input_h)
            , img_idx_(0)
            , img_dir_(img_dir)
            , calib_table_name_(calib_table_name)
            , input_blob_name_(input_blob_name)
            , read_cache_(read_cache)
    {
        input_count_ = 3 * input_w * input_h * batchsize;
        CUDA_CHECK(cudaMalloc(&device_input_, input_count_ * sizeof(float)));
        read_files_in_dir(img_dir, img_files_);
    }

    Int8EntropyCalibrator2::~Int8EntropyCalibrator2()
    {
        CUDA_CHECK(cudaFree(device_input_));
    }

    int Int8EntropyCalibrator2::getBatchSize() const TRT_NOEXCEPT
    {
        return batchsize_;
    }

    bool Int8EntropyCalibrator2::getBatch(void* bindings[], const char* names[], int nbBindings) TRT_NOEXCEPT
    {
        if (img_idx_ + batchsize_ > (int)img_files_.size()) {
            return false;
        }

        std::vector<cv::Mat> input_imgs_;
        for (int i = img_idx_; i < img_idx_ + batchsize_; i++) {
            std::cout << img_files_[i] << "  " << i << std::endl;
            cv::Mat temp = cv::imread(img_dir_ + img_files_[i]);
            if (temp.empty()){
                std::cerr << "Fatal error: image cannot open!" << std::endl;
                return false;
            }
            cv::Mat pr_img = preprocess_img(temp, input_w_, input_h_);
            input_imgs_.push_back(pr_img);
        }
        img_idx_ += batchsize_;
        cv::Mat blob = cv::dnn::blobFromImages(input_imgs_, 1.0 / 255.0, cv::Size(input_w_, input_h_), cv::Scalar(0, 0, 0), true, false);

        CUDA_CHECK(cudaMemcpy(device_input_, blob.ptr<float>(0), input_count_ * sizeof(float), cudaMemcpyHostToDevice));
        assert(!strcmp(names[0], input_blob_name_));
        bindings[0] = device_input_;
        return true;
    }

    const void* Int8EntropyCalibrator2::readCalibrationCache(size_t& length) TRT_NOEXCEPT
    {
        std::cout << "reading calib cache: " << calib_table_name_ << std::endl;
        calib_cache_.clear();
        std::ifstream input(calib_table_name_, std::ios::binary);
        input >> std::noskipws;
        if (read_cache_ && input.good())
        {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(calib_cache_));
        }
        length = calib_cache_.size();
        return length ? calib_cache_.data() : nullptr;
    }

    void Int8EntropyCalibrator2::writeCalibrationCache(const void* cache, size_t length) TRT_NOEXCEPT
    {
        std::cout << "writing calib cache: " << calib_table_name_ << " size: " << length << std::endl;
        std::ofstream output(calib_table_name_, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }





}  // namespace nvinfer1