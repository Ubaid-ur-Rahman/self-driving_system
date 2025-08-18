#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <fstream>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper.h"
#include "common_helper_cv.h"
#include "inference_helper.h"
#include "semantic_segmentation_engine.h"

/*** Macro ***/
#define TAG "SemanticSegmentationEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#if defined(ENABLE_TENSORRT)
#define MODEL_TYPE_ONNX
#else
#define MODEL_TYPE_TFLITE
#endif

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper.h"
#include "common_helper_cv.h"
#include "inference_helper.h"
#include "semantic_segmentation_engine.h"

/*** Macro ***/
#define TAG "SemanticSegmentationEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#if defined(ENABLE_TENSORRT)
#define MODEL_TYPE_ONNX
#else
#define MODEL_TYPE_TFLITE
#endif

#if defined(MODEL_TYPE_TFLITE)
#define MODEL_NAME  "road-segmentation-adas-0001.tflite"
#define TENSORTYPE  InferenceHelper::kTensorTypeFloat32
#define INPUT_NAME  "data"
#define INPUT_DIMS  { 1, 512, 896, 3 } // Updated to match actual model
#define IS_NCHW     false
#define IS_RGB      false
#define OUTPUT_NAME "Identity"
#define OUTPUT_DIMS { 1, 512, 896, 4 } // Updated to match actual output
#elif defined(MODEL_TYPE_ONNX)
#define MODEL_NAME  "road-segmentation-adas-0001.onnx"
#define TENSORTYPE  InferenceHelper::kTensorTypeFloat32
#define INPUT_NAME  "data"
#define INPUT_DIMS  { 1, 3, 512, 896 } // Updated to match actual model
#define IS_NCHW     true
#define IS_RGB      false
#define OUTPUT_NAME "tf.identity"
#define OUTPUT_DIMS { 1, 4, 512, 896 } // Updated to match actual output
#endif

/*** Function ***/
int32_t SemanticSegmentationEngine::Initialize(const std::string& work_dir, const int32_t num_threads) {
#ifndef ENABLE_SEGMENTATION
    return kRetOk;
#endif

    num_threads_ = num_threads;
    std::string model_filename = work_dir + "/pre_trained/" + MODEL_NAME;

    input_tensor_info_list_.clear();
    InferenceHelper::InputTensorInfo input_tensor_info(INPUT_NAME, TENSORTYPE, IS_NCHW);
    input_tensor_info.tensor_dims = INPUT_DIMS;
    input_tensor_info.data_type = InferenceHelper::InputTensorInfo::kDataTypeImage;
    input_tensor_info.normalize_mean[0] = 0;
    input_tensor_info.normalize_mean[1] = 0;
    input_tensor_info.normalize_mean[2] = 0;
    input_tensor_info.normalize_norm[0] = 1 / 255.f;
    input_tensor_info.normalize_norm[1] = 1 / 255.f;
    input_tensor_info.normalize_norm[2] = 1 / 255.f;
    input_tensor_info_list_.push_back(input_tensor_info);

    output_tensor_info_list_.clear();
    InferenceHelper::OutputTensorInfo output_tensor_info(OUTPUT_NAME, TENSORTYPE);
    output_tensor_info.tensor_dims = OUTPUT_DIMS;
    output_tensor_info_list_.push_back(output_tensor_info);

#if defined(MODEL_TYPE_TFLITE)
    inference_helper_ = std::unique_ptr<InferenceHelper>(InferenceHelper::Create(InferenceHelper::kTensorflowLiteXnnpack));
#elif defined(MODEL_TYPE_ONNX)
    inference_helper_ = std::unique_ptr<InferenceHelper>(InferenceHelper::Create(InferenceHelper::kTensorrt));
#endif

    if (!inference_helper_) {
        PRINT_E("Failed to create inference helper\n");
        return kRetErr;
    }
    if (inference_helper_->SetNumThreads(num_threads) != InferenceHelper::kRetOk) {
        inference_helper_.reset();
        return kRetErr;
    }
    if (inference_helper_->Initialize(model_filename, input_tensor_info_list_, output_tensor_info_list_) != InferenceHelper::kRetOk) {
        inference_helper_.reset();
        return kRetErr;
    }

    return kRetOk;
}

int32_t SemanticSegmentationEngine::Finalize() {
#ifndef ENABLE_SEGMENTATION
    return kRetOk;
#endif

    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    inference_helper_->Finalize();
    inference_helper_.reset();
    input_tensor_info_list_.clear();
    output_tensor_info_list_.clear();
    return kRetOk;
}

static cv::Vec3b GetColorForSegmentation[4] = {
    {70, 70, 70},  // Background
    {255, 0, 0},   // Road
    {0, 255, 0},   // Lane marking
    {0, 0, 255},   // Other
};

int32_t SemanticSegmentationEngine::Process(const cv::Mat& original_mat, Result& result) {
#ifndef ENABLE_SEGMENTATION
    return kRetOk;
#endif

    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    if (original_mat.empty() || original_mat.cols <= 0 || original_mat.rows <= 0) {
        PRINT_E("Invalid input image: empty or invalid dimensions\n");
        return kRetErr;
    }

    /*** PreProcess ***/
    const auto& t_pre_process0 = std::chrono::steady_clock::now();
    InferenceHelper::InputTensorInfo& input_tensor_info = input_tensor_info_list_[0];

    int32_t crop_x = 0, crop_y = 0, crop_w = original_mat.cols, crop_h = original_mat.rows;
    PRINT("Input image: width=%d, height=%d, channels=%d\n", original_mat.cols, original_mat.rows, original_mat.channels());

    cv::Mat img_src;
    cv::resize(original_mat, img_src, cv::Size(input_tensor_info.GetWidth(), input_tensor_info.GetHeight()), 0, 0, cv::INTER_AREA);
    PRINT("After resize: img_src width=%d, height=%d, channels=%d\n", img_src.cols, img_src.rows, img_src.channels());
    if (img_src.empty() || img_src.cols != input_tensor_info.GetWidth() || img_src.rows != input_tensor_info.GetHeight()) {
        PRINT_E("Invalid img_src after resize: width=%d, height=%d, expected=%dx%d\n",
                img_src.cols, img_src.rows, input_tensor_info.GetWidth(), input_tensor_info.GetHeight());
        img_src.release();
        return kRetErr;
    }

    input_tensor_info.data = img_src.data;
    input_tensor_info.data_type = InferenceHelper::InputTensorInfo::kDataTypeImage;
    input_tensor_info.image_info = { img_src.cols, img_src.rows, img_src.channels(), 0, 0, img_src.cols, img_src.rows, false, false };
    if (inference_helper_->PreProcess(input_tensor_info_list_) != InferenceHelper::kRetOk) {
        PRINT_E("PreProcess failed\n");
        img_src.release();
        return kRetErr;
    }
    img_src.release();
    const auto& t_pre_process1 = std::chrono::steady_clock::now();

    /*** Inference ***/
    const auto& t_inference0 = std::chrono::steady_clock::now();
    if (inference_helper_->Process(output_tensor_info_list_) != InferenceHelper::kRetOk) {
        PRINT_E("Inference failed\n");
        return kRetErr;
    }
    const auto& t_inference1 = std::chrono::steady_clock::now();

    /*** PostProcess ***/
    const auto& t_post_process0 = std::chrono::steady_clock::now();
    int32_t output_height = output_tensor_info_list_[0].tensor_dims[1];
    int32_t output_width = output_tensor_info_list_[0].tensor_dims[2];
    int32_t output_channel = output_tensor_info_list_[0].tensor_dims[3];
    float* output_raw_data = output_tensor_info_list_[0].GetDataAsFloat();
    PRINT("Output tensor: name=%s, width=%d, height=%d, channels=%d\n", output_tensor_info_list_[0].name.c_str(),
          output_width, output_height, output_channel);

    // Debug: Output tensor introspection
    PRINT("[DBG SEG] Output tensor dims=[%d %d %d %d]\n",
          output_tensor_info_list_[0].tensor_dims[0],
          output_tensor_info_list_[0].tensor_dims[1],
          output_tensor_info_list_[0].tensor_dims[2],
          output_tensor_info_list_[0].tensor_dims[3]);
    if (!output_raw_data) {
        PRINT_E("[DBG SEG] output_raw_data is null\n");
        return kRetErr;
    }
    if (output_channel != 4) {
        PRINT_E("[DBG SEG] Unexpected channel count: %d\n", output_channel);
        return kRetErr;
    }

    // Safe sampling function
    auto sample_score = [&](int y, int x) -> cv::Vec4f {
        y = std::max(0, std::min(y, output_height - 1));
        x = std::max(0, std::min(x, output_width - 1));
        cv::Vec4f* ptr = reinterpret_cast<cv::Vec4f*>(output_raw_data + (y * output_width + x) * output_channel * sizeof(float));
        return *ptr;
    };

    // Sample scores for debugging
    cv::Vec4f s_tl = sample_score(0, 0);
    cv::Vec4f s_mid = sample_score(output_height / 2, output_width / 2);
    cv::Vec4f s_br = sample_score(output_height - 1, output_width - 1);
    PRINT("[DBG SEG] Sample scores top-left=(%.3f %.3f %.3f %.3f) mid=(%.3f %.3f %.3f %.3f) bottom-right=(%.3f %.3f %.3f %.3f)\n",
          s_tl[0], s_tl[1], s_tl[2], s_tl[3],
          s_mid[0], s_mid[1], s_mid[2], s_mid[3],
          s_br[0], s_br[1], s_br[2], s_br[3]);

    // Wrap the raw output tensor as a float4 mat
    cv::Mat output_mat(output_height, output_width, CV_32FC4, output_raw_data);
    if (output_mat.empty()) {
        PRINT_E("Output matrix is empty\n");
        return kRetErr;
    }

    // Build combined visualization
    cv::Mat image_combined(output_height, output_width, CV_8UC3);
    for (int y = 0; y < output_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
            cv::Vec4f score = output_mat.at<cv::Vec4f>(y, x);
            // Normalize scores to ensure valid color values
            float bg = std::max(0.0f, std::min(score[0], 1.0f));
            float road = std::max(0.0f, std::min(score[1], 1.0f));
            float lane = std::max(0.0f, std::min(score[2], 1.0f));
            float other = std::max(0.0f, std::min(score[3], 1.0f));
            uint8_t b = static_cast<uint8_t>(bg * 70 + other * 255);
            uint8_t g = static_cast<uint8_t>(bg * 70 + lane * 255);
            uint8_t r = static_cast<uint8_t>(bg * 70 + road * 255);
            image_combined.at<cv::Vec3b>(y, x) = {b, g, r};
        }
    }

    PRINT("image_combined: width=%d, height=%d, type=%d\n", image_combined.cols, image_combined.rows, image_combined.type());
    const auto& t_post_process1 = std::chrono::steady_clock::now();

    result.image_list.clear();
    result.image_combined = image_combined;
    result.crop.x = std::max(0, crop_x);
    result.crop.y = std::max(0, crop_y);
    result.crop.w = std::min(crop_w, original_mat.cols - result.crop.x);
    result.crop.h = std::min(crop_h, original_mat.rows - result.crop.y);
    result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
    result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
    result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;

    PRINT("[DBG SEG] Returning segmentation result: combined=(%d,%d), crop=(%d,%d,%d,%d)\n",
          result.image_combined.cols, result.image_combined.rows,
          result.crop.x, result.crop.y, result.crop.w, result.crop.h);

    output_mat.release();
    return kRetOk;
}