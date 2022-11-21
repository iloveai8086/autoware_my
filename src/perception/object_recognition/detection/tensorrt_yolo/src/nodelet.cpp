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

#include "tensorrt_yolo/nodelet.hpp"
#include <glob.h>

namespace {
    std::vector<std::string> getFilePath(const std::string &input_dir) {
        glob_t globbuf;
        std::vector<std::string> files;
        glob((input_dir + "*").c_str(), 0, NULL, &globbuf);
        for (size_t i = 0; i < globbuf.gl_pathc; i++) {
            files.push_back(globbuf.gl_pathv[i]);
        }
        globfree(&globbuf);
        return files;
    }
}  // namespace
namespace object_recognition {
    void TensorrtYoloNodelet::onInit() {
        nh_ = getNodeHandle();
        pnh_ = getPrivateNodeHandle();
        it_.reset(new image_transport::ImageTransport(pnh_));
        std::string onnx_file;
        std::string engine_file;
        std::string label_file;
        std::string calib_image_directory;
        std::string calib_cache_file;
        std::string mode;

        pnh_.param<std::string>("onnx_file", onnx_file, "");
        pnh_.param<std::string>("engine_file", engine_file, "");
        pnh_.param<std::string>("label_file", label_file, "");
        pnh_.param<std::string>("calib_image_directory", calib_image_directory, "");
        pnh_.param<std::string>("calib_cache_file", calib_cache_file, "");
        pnh_.param<std::string>("mode", mode, "FP32");
        pnh_.param<int>("num_anchors", yolo_config_.num_anchors, 3);
        if (!pnh_.getParam("anchors", yolo_config_.anchors)) {
            NODELET_WARN("Fail to load anchors");
            yolo_config_.anchors = {10, 13, 16, 30, 33, 23, 30, 61, 62,
                                    45, 59, 119, 116, 90, 156, 198, 373, 326};
        }
        if (!pnh_.getParam("scale_x_y", yolo_config_.scale_x_y)) {
            NODELET_WARN("Fail to load scale_x_y");
            yolo_config_.scale_x_y = {1.0, 1.0, 1.0};
        }
        pnh_.param<float>("score_thresh", yolo_config_.score_thresh, 0.1);
        pnh_.param<float>("iou_thresh", yolo_config_.iou_thresh, 0.45);
        pnh_.param<int>("detections_per_im", yolo_config_.detections_per_im, 100);
        pnh_.param<bool>("use_darknet_layer", yolo_config_.use_darknet_layer, true);
        pnh_.param<float>("ignore_thresh", yolo_config_.ignore_thresh, 0.5);

        if (!readLabelFile(label_file, &labels_)) {
            NODELET_ERROR("Could not find label file");
        }
        std::ifstream fs(engine_file);
        const auto calibration_images = getFilePath(calib_image_directory);
        if (fs.is_open()) {
            NODELET_INFO("Found %s", engine_file.c_str());
            net_ptr_.reset(new yolo::Net(engine_file, false));
            if (net_ptr_->getMaxBatchSize() != 1) {
                NODELET_INFO(
                        "Max batch size %d should be 1. Rebuild engine from file", net_ptr_->getMaxBatchSize());
                net_ptr_.reset(
                        new yolo::Net(onnx_file, mode, 1, yolo_config_, calibration_images, calib_cache_file));
                net_ptr_->save(engine_file);
            }
        } else {
            NODELET_INFO("Could not find %s, try making TensorRT engine from onnx", engine_file.c_str());
            net_ptr_.reset(
                    new yolo::Net(onnx_file, mode, 1, yolo_config_, calibration_images, calib_cache_file));
            net_ptr_->save(engine_file);
        }
        image_transport::SubscriberStatusCallback connect_cb =
                boost::bind(&TensorrtYoloNodelet::connectCb, this);
        std::lock_guard<std::mutex> lock(connect_mutex_);
        objects_pub_ = pnh_.advertise<autoware_perception_msgs::DynamicObjectWithFeatureArray>(
                "out/objects", 1, boost::bind(&TensorrtYoloNodelet::connectCb, this),
                boost::bind(&TensorrtYoloNodelet::connectCb, this));
        image_pub_ = it_->advertise("out/image", 1, connect_cb, connect_cb);
        out_scores_ =
                std::make_unique<float[]>(net_ptr_->getMaxBatchSize() * net_ptr_->getMaxDetections());
        out_boxes_ =
                std::make_unique<float[]>(net_ptr_->getMaxBatchSize() * net_ptr_->getMaxDetections() * 4);
        out_classes_ =
                std::make_unique<float[]>(net_ptr_->getMaxBatchSize() * net_ptr_->getMaxDetections());
        i2d[0] = scale;
        i2d[1] = 0;
        i2d[2] = (-scale * image_cols + input_width + scale - 1) * 0.5;
        i2d[3] = 0;
        i2d[4] = scale;
        i2d[5] = (-scale * image_rows + input_height + scale - 1) * 0.5; // 之前的公式的推导，计算我们的M矩阵，知道为什么是这么写的

        cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);  // image to dst(network), 2x3 matrix   正负变换的矩阵
        cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);  // dst to image, 2x3 matrix
        cv::invertAffineTransform(m2x3_i2d,
                                  m2x3_d2i);
    }

    void TensorrtYoloNodelet::connectCb() {
        std::lock_guard<std::mutex> lock(connect_mutex_);
        if (objects_pub_.getNumSubscribers() == 0 && image_pub_.getNumSubscribers() == 0)
            image_sub_.shutdown();
        else if (!image_sub_)
            image_sub_ = it_->subscribe("in/image", 1, &TensorrtYoloNodelet::callback, this);
    }

    void TensorrtYoloNodelet::callback(const sensor_msgs::Image::ConstPtr &in_image_msg) {
        autoware_perception_msgs::DynamicObjectWithFeatureArray out_objects;

        cv_bridge::CvImagePtr in_image_ptr;
        try {
            in_image_ptr = cv_bridge::toCvCopy(in_image_msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception &e) {
            NODELET_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        // -------------------------------------------------------------------------------------------------------------
        auto image = in_image_ptr->image;
        // int input_batch = 1;
        // int input_channel = 3;
        /*int input_height = 640;
        int input_width = 640;  // 输入是个固定的输入
        // std::cout << image.cols << ' ' << image.rows << std::endl;
        float scale_x = input_width / (float) image.cols;  // 960
        float scale_y = input_height / (float) image.rows;  // 540
        float scale = std::min(scale_x, scale_y);  // 等比缩放取一个最小的比例值
        float i2d[6], d2i[6];
        // resize图像，源图像和目标图像几何中心的对齐   等比缩放、长边对齐，上下左右填充 warpaffine
        i2d[0] = scale;
        i2d[1] = 0;
        i2d[2] = (-scale * image.cols + input_width + scale - 1) * 0.5;
        i2d[3] = 0;
        i2d[4] = scale;
        i2d[5] = (-scale * image.rows + input_height + scale - 1) * 0.5; // 之前的公式的推导，计算我们的M矩阵，知道为什么是这么写的

        cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);  // image to dst(network), 2x3 matrix   正负变换的矩阵
        cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);  // dst to image, 2x3 matrix
        cv::invertAffineTransform(m2x3_i2d,
                                  m2x3_d2i);  // 计算一个反仿射变换   得到dst2img的矩阵   正矩阵是做warpaffine把图像变成输入的过程，****逆矩阵是把框变成图像尺度的过程****
        */
        // -------------------------------------------------------------------------------------------------------------

        if (!net_ptr_->detect(
                in_image_ptr->image, out_scores_.get(), out_boxes_.get(), out_classes_.get())) {
            NODELET_WARN("Fail to inference");
            return;
        }
        // std::cout << "infer done!" << std::endl;
        const auto width = in_image_ptr->image.cols;
        const auto height = in_image_ptr->image.rows;
        for (int i = 0; i < yolo_config_.detections_per_im; ++i) {
            if (out_scores_[i] < yolo_config_.ignore_thresh) break;
            autoware_perception_msgs::DynamicObjectWithFeature object;

            // std::cout << out_boxes_[4 * i] << ' ' << out_boxes_[4 * i + 1] << ' '
            //           << out_boxes_[4 * i + 2] << ' ' << out_boxes_[4 * i + 3] << std::endl;

            object.feature.roi.x_offset = out_boxes_[4 * i] * width;
            object.feature.roi.y_offset = out_boxes_[4 * i + 1] * height;
            object.feature.roi.width = out_boxes_[4 * i + 2] * width;
            object.feature.roi.height = out_boxes_[4 * i + 3] * height;

            // 这地方是左上角和wh，很离谱
//            object.feature.roi.x_offset = out_boxes_[4 * i] * 640;
//            object.feature.roi.y_offset = out_boxes_[4 * i + 1] * 640;
//            object.feature.roi.width = out_boxes_[4 * i + 2] * 640;
//            object.feature.roi.height = out_boxes_[4 * i + 3] * 640;
//
//            float cx = object.feature.roi.x_offset + object.feature.roi.width / 2.0;
//            float cy = object.feature.roi.y_offset + object.feature.roi.height / 2.0;
//            float width  = object.feature.roi.width;
//            float height = object.feature.roi.height;
//
//            float left   = cx - width * 0.5;
//            float top    = cy - height * 0.5;
//            float right  = cx + width * 0.5;
//            float bottom = cy + height * 0.5;
//
//            // 对应图上的位置  反变换回原图 1920*1080   d2i   只用了025，因为只有缩放平移的时候，就只有三个有效自由度，缩放、平移 scale dx dy就是0 2 5
//            float image_base_left   = d2i[0] * left   + d2i[2];  // x的
//            float image_base_right  = d2i[0] * right  + d2i[2];
//            float image_base_top    = d2i[0] * top    + d2i[5];  // y的
//            float image_base_bottom = d2i[0] * bottom + d2i[5];

            object.object.semantic.confidence = out_scores_[i];
            const auto class_id = static_cast<int>(out_classes_[i]);


            if (labels_[class_id] == "car") {
                object.object.semantic.type = autoware_perception_msgs::Semantic::CAR;
            } else if (labels_[class_id] == "person") {
                object.object.semantic.type = autoware_perception_msgs::Semantic::PEDESTRIAN;
            } else if (labels_[class_id] == "bus") {
                object.object.semantic.type = autoware_perception_msgs::Semantic::BUS;
            } else if (labels_[class_id] == "truck") {
                object.object.semantic.type = autoware_perception_msgs::Semantic::TRUCK;
            } else if (labels_[class_id] == "bicycle") {
                object.object.semantic.type = autoware_perception_msgs::Semantic::BICYCLE;
            } else if (labels_[class_id] == "motorbike") {
                object.object.semantic.type = autoware_perception_msgs::Semantic::MOTORBIKE;
            } else {
                object.object.semantic.type = autoware_perception_msgs::Semantic::UNKNOWN;
            }
            out_objects.feature_objects.push_back(object);

            const auto left = std::max(0, static_cast<int>(object.feature.roi.x_offset));
            const auto top = std::max(0, static_cast<int>(object.feature.roi.y_offset));
            const auto right =
                    std::min(static_cast<int>(object.feature.roi.x_offset + object.feature.roi.width), width);
            const auto bottom =
                    std::min(static_cast<int>(object.feature.roi.y_offset + object.feature.roi.height), height);
//            cv::rectangle(
//                    in_image_ptr->image, cv::Point(image_base_left, image_base_top), cv::Point(image_base_right, image_base_bottom), cv::Scalar(0, 0, 255), 3,
//                    8, 0);
            cv::rectangle(
                    in_image_ptr->image, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 3,
                    8, 0);
        }
        image_pub_.publish(in_image_ptr->toImageMsg());

        out_objects.header = in_image_msg->header;
        objects_pub_.publish(out_objects);
    }

    bool TensorrtYoloNodelet::readLabelFile(
            const std::string &filepath, std::vector<std::string> *labels) {
        std::ifstream labelsFile(filepath);
        if (!labelsFile.is_open()) {
            NODELET_ERROR("Could not open label file. [%s]", filepath.c_str());
            return false;
        }
        std::string label;
        while (getline(labelsFile, label)) {
            labels->push_back(label);
        }
        return true;
    }

}  // namespace object_recognition

#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(object_recognition::TensorrtYoloNodelet, nodelet::Nodelet)
