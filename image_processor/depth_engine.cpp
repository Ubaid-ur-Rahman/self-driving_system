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
#include "depth_engine.h"

/*** Macro ***/
#define TAG "DepthEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#if defined(ENABLE_TENSORRT)
#define MODEL_TYPE_ONNX
#else
#define MODEL_TYPE_TFLITE
#endif

#if defined(MODEL_TYPE_TFLITE)
#define MODEL_NAME  "ldrn_kitti_resnext101_pretrained_data_grad_192x320.tflite"
#define INPUT_NAME  "input_1"
#define INPUT_DIMS  { 1, 192, 320, 3 }
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME "Identity"
#define TENSORTYPE  InferenceHelper::kTensorTypeFloat32
#elif defined(MODEL_TYPE_ONNX)
#define MODEL_NAME  "ldrn_kitti_resnext101_pretrained_data_grad_192x320.onnx"
#define INPUT_DIMS  { 1, 3, 192, 320 }
#define INPUT_NAME  "input.1"
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME "2499"
#define OUTPUT_DIMS { 1, 1, 192, 320 }
#define TENSORTYPE  InferenceHelper::kTensorTypeFloat32
#endif

/*** Function ***/
int32_t DepthEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
#ifndef ENABLE_DEPTH
    return kRetOk;
#endif

    /* Set model information */
    std::string model_filename = work_dir + "/pre_trained/" + MODEL_NAME;

    /* Set input tensor info */
    input_tensor_info_list_.clear();
    InferenceHelper::InputTensorInfo input_tensor_info(INPUT_NAME, TENSORTYPE, IS_NCHW);
    input_tensor_info.tensor_dims = INPUT_DIMS;
    input_tensor_info.data_type = InferenceHelper::InputTensorInfo::kDataTypeImage;
    input_tensor_info.normalize_mean[0] = 0.485f;
    input_tensor_info.normalize_mean[1] = 0.456f;
    input_tensor_info.normalize_mean[2] = 0.406f;
    input_tensor_info.normalize_norm[0] = 0.229f;
    input_tensor_info.normalize_norm[1] = 0.224f;
    input_tensor_info.normalize_norm[2] = 0.225f;
    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
    InferenceHelper::OutputTensorInfo output_tensor_info(OUTPUT_NAME, TENSORTYPE);
#ifdef OUTPUT_DIMS
    output_tensor_info.tensor_dims = OUTPUT_DIMS;
#endif
    output_tensor_info_list_.push_back(output_tensor_info);

    /* Create and Initialize Inference Helper */
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

    /* Log output tensor dimensions after initialization */
    if (!output_tensor_info_list_.empty()) {
        PRINT("Output tensor dims: [%d", output_tensor_info_list_[0].tensor_dims[0]);
        for (size_t i = 1; i < output_tensor_info_list_[0].tensor_dims.size(); ++i) {
            PRINT(",%d", output_tensor_info_list_[0].tensor_dims[i]);
        }
        PRINT("]\n");
    }

    PRINT("Initialized TensorRT model: %s\n", model_filename.c_str());
    return kRetOk;
}

int32_t DepthEngine::Finalize()
{
#ifndef ENABLE_DEPTH
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

int32_t DepthEngine::Process(const cv::Mat& original_mat, Result& result)
{
#ifndef ENABLE_DEPTH
    return kRetOk;
#endif

    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    /*** PreProcess ***/
    const auto& t_pre_process0 = std::chrono::steady_clock::now();
    InferenceHelper::InputTensorInfo& input_tensor_info = input_tensor_info_list_[0];
    /* Validate input image */
    if (original_mat.empty() || original_mat.cols <= 0 || original_mat.rows <= 0) {
        PRINT_E("Invalid input image: empty or invalid dimensions\n");
        return kRetErr;
    }
    PRINT("Input image: width=%d, height=%d, channels=%d, data=%p\n", 
          original_mat.cols, original_mat.rows, original_mat.channels(), original_mat.data);

    /* Resize and convert input */
    int32_t crop_x = 0;
    int32_t crop_y = 0;
    int32_t crop_w = original_mat.cols;
    int32_t crop_h = original_mat.rows;
    cv::Mat img_src;
    CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, 
                                CommonHelper::kCropTypeStretch, true, 
                                input_tensor_info.GetWidth(), input_tensor_info.GetHeight());
    PRINT("After CropResizeCvt: img_src width=%d, height=%d, channels=%d, data=%p\n", 
          img_src.cols, img_src.rows, img_src.channels(), img_src.data);
    
    /* Validate img_src dimensions */
    if (img_src.empty()) {
        PRINT_E("img_src is empty after CropResizeCvt\n");
        return kRetErr;
    }
    if (img_src.cols != input_tensor_info.GetWidth() || img_src.rows != input_tensor_info.GetHeight()) {
        PRINT_E("Invalid img_src after CropResizeCvt: width=%d, height=%d, expected=%dx%d\n",
                img_src.cols, img_src.rows, input_tensor_info.GetWidth(), input_tensor_info.GetHeight());
        img_src.release();
        return kRetErr;
    }
    if (img_src.channels() != 3) {
        PRINT_E("Invalid img_src channels: got %d, expected 3\n", img_src.channels());
        img_src.release();
        return kRetErr;
    }

    /* Save img_src for debugging */
    CommonHelper::SaveImage(img_src, "output/debug_img_src.png");
    PRINT("Saved debug_img_src.png\n");

    /* Set input tensor info */
    input_tensor_info.data = img_src.data;
    input_tensor_info.data_type = InferenceHelper::InputTensorInfo::kDataTypeImage;
    input_tensor_info.image_info.width = img_src.cols; // Should be 320
    input_tensor_info.image_info.height = img_src.rows; // Should be 192
    input_tensor_info.image_info.channel = img_src.channels();
    input_tensor_info.image_info.crop_x = 0;
    input_tensor_info.image_info.crop_y = 0;
    input_tensor_info.image_info.crop_width = img_src.cols;
    input_tensor_info.image_info.crop_height = img_src.rows;
    input_tensor_info.image_info.is_bgr = false;
    input_tensor_info.image_info.swap_color = false;
    PRINT("Input tensor: name=%s, dims=[%d,%d,%d,%d], image_info=[width=%d, height=%d, channels=%d], data=%p\n",
          input_tensor_info.name.c_str(),
          input_tensor_info.tensor_dims[0], input_tensor_info.tensor_dims[1],
          input_tensor_info.tensor_dims[2], input_tensor_info.tensor_dims[3],
          input_tensor_info.image_info.width, input_tensor_info.image_info.height,
          input_tensor_info.image_info.channel, input_tensor_info.data);

    /* Validate input buffer */
    if (!input_tensor_info.data) {
        PRINT_E("Input tensor data is null\n");
        img_src.release();
        return kRetErr;
    }

    /* Debug input data */
    uint8_t* input_data = static_cast<uint8_t*>(input_tensor_info.data);
    PRINT("Input data sample: [%.0f, %.0f, %.0f, %.0f]\n", 
          input_data[0] / 255.0f, input_data[1] / 255.0f, input_data[2] / 255.0f, input_data[3] / 255.0f);

    PRINT("Starting PreProcess\n");
    if (inference_helper_->PreProcess(input_tensor_info_list_) != InferenceHelper::kRetOk) {
        PRINT_E("PreProcess failed\n");
        img_src.release();
        return kRetErr;
    }
    PRINT("PreProcess completed\n");
    img_src.release();
    const auto& t_pre_process1 = std::chrono::steady_clock::now();

    /* Save input tensor for debugging */
    std::ofstream debug_file("output/debug_input_tensor.bin", std::ios::binary);
    if (debug_file.is_open()) {
        debug_file.write(static_cast<char*>(input_tensor_info.data), 
                        input_tensor_info.GetWidth() * input_tensor_info.GetHeight() * input_tensor_info.image_info.channel);
        debug_file.close();
        PRINT("Saved debug_input_tensor.bin\n");
    }

    /*** Inference ***/
    PRINT("Starting Inference\n");
    const auto& t_inference0 = std::chrono::steady_clock::now();
    if (inference_helper_->Process(output_tensor_info_list_) != InferenceHelper::kRetOk) {
        PRINT_E("Inference failed\n");
        return kRetErr;
    }
    PRINT("Inference completed\n");
    const auto& t_inference1 = std::chrono::steady_clock::now();


    /*** PostProcess ***/
    const auto& t_post_process0 = std::chrono::steady_clock::now();
    /* Retrieve the result */
    int32_t output_height = output_tensor_info_list_[0].GetHeight();
    int32_t output_width = output_tensor_info_list_[0].GetWidth();
    int32_t output_channels = output_tensor_info_list_[0].tensor_dims.size() > 1 ? output_tensor_info_list_[0].tensor_dims[1] : 1;
    float* values = output_tensor_info_list_[0].GetDataAsFloat();
    PRINT("Output tensor: name=%s, width=%d, height=%d, channels=%d, data=%p\n",
          output_tensor_info_list_[0].name.c_str(), output_width, output_height, output_channels, values);
    if (!values || output_height <= 0 || output_width <= 0 || output_channels != 1) {
        PRINT_E("Invalid output tensor: height=%d, width=%d, channels=%d, data=%p\n",
                output_height, output_width, output_channels, values);
        return kRetErr;
    }

    /* Debug: Log first few output values */
    if (values) {
        PRINT("Depth output sample values: [%.3f, %.3f, %.3f, %.3f]\n",
              values[0], values[1], values[2], values[3]);
    }

    /* Save output tensor for debugging */
    debug_file.open("output/debug_output_tensor.bin", std::ios::binary);
    if (debug_file.is_open()) {
        debug_file.write(reinterpret_cast<char*>(values), output_width * output_height * output_channels * sizeof(float));
        debug_file.close();
        PRINT("Saved debug_output_tensor.bin\n");
    }

    /* Validate output dimensions against expected */
#ifdef OUTPUT_DIMS
    const std::vector<int32_t> expected_dims = OUTPUT_DIMS;
    if (output_height != expected_dims[2] || output_width != expected_dims[3] || output_channels != expected_dims[1]) {
        PRINT_E("Output tensor dimension mismatch: got [%d,%d,%d,%d], expected [%d,%d,%d,%d]\n",
                output_tensor_info_list_[0].tensor_dims[0], output_channels, output_height, output_width,
                expected_dims[0], expected_dims[1], expected_dims[2], expected_dims[3]);
        return kRetErr;
    }
#endif

    PRINT("Creating mat_out\n");
    cv::Mat mat_out = cv::Mat(output_height, output_width, CV_32FC1, values);
    if (mat_out.empty()) {
        PRINT_E("Failed to create mat_out\n");
        return kRetErr;
    }
    PRINT("mat_out created: width=%d, height=%d, type=%d\n", mat_out.cols, mat_out.rows, mat_out.type());

    /* Convert and crop */
    cv::Mat mat_out_8u;
    double min_val, max_val;
    cv::minMaxLoc(mat_out, &min_val, &max_val);
    PRINT("Depth values: min=%.3f, max=%.3f\n", min_val, max_val);
    if (max_val == min_val) {
        PRINT_E("Invalid depth range: min=max=%.3f\n", min_val);
        mat_out.release();
        return kRetErr;
    }
    mat_out.convertTo(mat_out_8u, CV_8UC1, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));
    mat_out.release();
    PRINT("Converted to mat_out_8u: width=%d, height=%d, type=%d\n", mat_out_8u.cols, mat_out_8u.rows, mat_out_8u.type());

    /* Crop to remove top portion (experimentally determined) */
    if (mat_out_8u.rows > 0) {
        int32_t crop_height = static_cast<int32_t>(mat_out_8u.rows * (1.0 - 0.18));
        int32_t crop_y = static_cast<int32_t>(mat_out_8u.rows * 0.18);
        if (crop_y + crop_height <= mat_out_8u.rows && crop_height > 0) {
            mat_out_8u = mat_out_8u(cv::Rect(0, crop_y, mat_out_8u.cols, crop_height));
            PRINT("After crop: mat_out_8u width=%d, height=%d, type=%d\n", 
                  mat_out_8u.cols, mat_out_8u.rows, mat_out_8u.type());
        } else {
            PRINT_E("Invalid crop dimensions: crop_y=%d, crop_height=%d, mat_out_8u.rows=%d\n",
                    crop_y, crop_height, mat_out_8u.rows);
            mat_out_8u.release();
            return kRetErr;
        }
    } else {
        PRINT_E("mat_out_8u is empty after conversion\n");
        mat_out_8u.release();
        return kRetErr;
    }

    const auto& t_post_process1 = std::chrono::steady_clock::now();

    /* Return the results */
    result.mat_out = mat_out_8u.clone();
    result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
    result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
    result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;

    PRINT("Process completed successfully\n");
    return kRetOk;
}