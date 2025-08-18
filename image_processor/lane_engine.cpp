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
#include <numeric>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper.h"
#include "common_helper_cv.h"
#include "inference_helper.h"
#include "lane_engine.h"

/*** Macro ***/
#define TAG "LaneEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#if defined(ENABLE_TENSORRT)
#define MODEL_TYPE_ONNX
#else
#define MODEL_TYPE_TFLITE
#endif

#ifdef MODEL_TYPE_TFLITE
#define MODEL_NAME  "ultra_fast_lane_detection_culane_192x320.tflite"
#define TENSORTYPE  InferenceHelper::kTensorTypeFloat32
#define INPUT_NAME  "input_1"
#define INPUT_DIMS  { 1, 192, 320, 3 }
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME "Identity"
#elif defined(MODEL_TYPE_ONNX)
#define MODEL_NAME  "ultra_fast_lane_detection_culane_192x320.onnx"
#define TENSORTYPE  InferenceHelper::kTensorTypeFloat32
#define INPUT_NAME  "input.1"
#define INPUT_DIMS  { 1, 3, 192, 320 }
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME "200"
#endif

static constexpr int32_t culane_row_anchor[] = { 81, 87, 94, 100, 107, 113, 120, 126, 133, 139, 146, 152, 159, 165, 172, 178, 185, 191 }; // Scaled for 192
static constexpr int32_t kNumGriding = 201;
static constexpr int32_t kNumClassPerLine = 18;
static constexpr int32_t kNumLine = 4;
static constexpr int32_t kNumWidth = 320;
static constexpr int32_t kNumHeight = 192;
static constexpr float kDeltaWidth = ((kNumWidth - 1) - 0) / static_cast<float>((kNumGriding - 1) - 1);

/*** Function ***/
int32_t LaneEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
#ifndef ENABLE_LANE
    return kRetOk;
#endif
    std::string model_filename = work_dir + "/pre_trained/" + MODEL_NAME;

    input_tensor_info_list_.clear();
    InferenceHelper::InputTensorInfo input_tensor_info(INPUT_NAME, TENSORTYPE, IS_NCHW);
    input_tensor_info.tensor_dims = INPUT_DIMS;
    input_tensor_info.data_type = InferenceHelper::InputTensorInfo::kDataTypeImage;
    input_tensor_info.normalize_mean[0] = 0;
    input_tensor_info.normalize_mean[1] = 0;
    input_tensor_info.normalize_mean[2] = 0;
    input_tensor_info.normalize_norm[0] = 1.0f;
    input_tensor_info.normalize_norm[1] = 1.0f;
    input_tensor_info.normalize_norm[2] = 1.0f;
    input_tensor_info_list_.push_back(input_tensor_info);

    output_tensor_info_list_.clear();
    output_tensor_info_list_.push_back(InferenceHelper::OutputTensorInfo(OUTPUT_NAME, TENSORTYPE));

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

int32_t LaneEngine::Finalize()
{
#ifndef ENABLE_LANE
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

static inline void Flip_1(std::vector<float>& val_list, int32_t num_i, int32_t num_j, int32_t num_k)
{
    for (int32_t i = 0; i < num_i; i++) {
        for (int32_t j = 0; j < num_j / 2; j++) {
            for (int32_t k = 0; k < num_k; k++) {
                int32_t new_j = num_j - 1 - j;
                std::swap(val_list[i * (num_j * num_k) + j * num_k + k], val_list[i * (num_j * num_k) + new_j * num_k + k]);
            }
        }
    }
}

static void Softmax_0(std::vector<float>& val_list, int32_t num_i, int32_t num_j, int32_t num_k)
{
    // In-place softmax to avoid extra vector
    for (int32_t j = 0; j < num_j; j++) {
        for (int32_t k = 0; k < num_k; k++) {
            float sum = 0;
            for (int32_t i = 0; i < num_i; i++) {
                sum += std::exp(val_list[i * (num_j * num_k) + j * num_k + k]);
            }
            for (int32_t i = 0; i < num_i; i++) {
                val_list[i * (num_j * num_k) + j * num_k + k] = std::exp(val_list[i * (num_j * num_k) + j * num_k + k]) / sum;
            }
        }
    }
}

static void MulSum(std::vector<float>& res, const std::vector<float>& val_list_0, const std::vector<float>& val_list_1, int32_t num_i, int32_t num_j, int32_t num_k)
{
    // In-place result
    for (int32_t j = 0; j < num_j; j++) {
        for (int32_t k = 0; k < num_k; k++) {
            float sum = 0;
            for (int32_t i = 0; i < num_i; i++) {
                sum += val_list_0[i * (num_j * num_k) + j * num_k + k] * val_list_1[i];
            }
            res[j * num_k + k] = sum;
        }
    }
}

static inline std::vector<bool> CheckIfValid(const std::vector<float>& val_list, int32_t num_i, int32_t num_j, int32_t num_k)
{
    std::vector<bool> res(num_j * num_k, true);
    for (int32_t j = 0; j < num_j; j++) {
        for (int32_t k = 0; k < num_k; k++) {
            float max_val = -999;
            int32_t max_index = 0;
            for (int32_t i = 0; i < num_i; i++) {
                float val = val_list[i * (num_j * num_k) + j * num_k + k];
                if (val > max_val) {
                    max_val = val;
                    max_index = i;
                }
            }
            if (max_index == num_i - 1) {
                res[j * num_k + k] = false;
            }
        }
    }
    return res;
}

int32_t LaneEngine::Process(const cv::Mat& original_mat, Result& result)
{
#ifndef ENABLE_LANE
    return kRetOk;
#endif
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    /*** PreProcess ***/
    const auto& t_pre_process0 = std::chrono::steady_clock::now();
    InferenceHelper::InputTensorInfo& input_tensor_info = input_tensor_info_list_[0];

    int32_t crop_x = 0, crop_y = 0, crop_w = original_mat.cols, crop_h = original_mat.rows;
    PRINT("Input image: width=%d, height=%d, channels=%d\n", original_mat.cols, original_mat.rows, original_mat.channels());
    if (original_mat.empty() || original_mat.cols <= 0 || original_mat.rows <= 0) {
        PRINT_E("Invalid input image\n");
        return kRetErr;
    }
    cv::Mat img_src = cv::Mat::zeros(input_tensor_info.GetHeight(), input_tensor_info.GetWidth(), CV_8UC3);
    CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeStretch);
    PRINT("After CropResizeCvt: img_src width=%d, height=%d, channels=%d\n", img_src.cols, img_src.rows, img_src.channels());
    if (img_src.empty() || img_src.cols != input_tensor_info.GetWidth() || img_src.rows != input_tensor_info.GetHeight()) {
        PRINT_E("Invalid img_src after CropResizeCvt\n");
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

    int32_t output_size = kNumGriding * kNumClassPerLine * kNumLine;
    if (output_tensor_info_list_[0].GetElementNum() != output_size) {
        PRINT_E("Invalid output size=%d, expected=%d\n", output_tensor_info_list_[0].GetElementNum(), output_size);
        return kRetErr;
    }
    std::vector<float> output_raw_val(output_size);
    std::memcpy(output_raw_val.data(), output_tensor_info_list_[0].GetDataAsFloat(), output_size * sizeof(float));
    PRINT("Output tensor: griding=%d, class_per_line=%d, lines=%d\n", kNumGriding, kNumClassPerLine, kNumLine);

    Flip_1(output_raw_val, kNumGriding, kNumClassPerLine, kNumLine);
    Softmax_0(output_raw_val, kNumGriding - 1, kNumClassPerLine, kNumLine); // In-place softmax
    std::vector<float> idx(kNumGriding - 1);
    std::iota(idx.begin(), idx.end(), 1.0f);
    std::vector<float> loc(kNumClassPerLine * kNumLine);
    MulSum(loc, output_raw_val, idx, kNumGriding - 1, kNumClassPerLine, kNumLine);
    std::vector<bool> valid_map = CheckIfValid(output_raw_val, kNumGriding, kNumClassPerLine, kNumLine);

    result.line_list.clear();
    result.line_list.resize(kNumLine); // Pre-allocate
    for (int32_t k = 0; k < kNumLine; k++) {
        result.line_list[k].reserve(kNumClassPerLine); // Pre-allocate
        for (int32_t j = 0; j < kNumClassPerLine; j++) {
            int32_t index = j * kNumLine + k;
            float val = loc[index];
            if (valid_map[index] && val > 0) {
                int32_t x = static_cast<int32_t>(val * kDeltaWidth * crop_w / kNumWidth + crop_x);
                int32_t y = static_cast<int32_t>(culane_row_anchor[j] * crop_h / kNumHeight + crop_y);
                result.line_list[k].push_back({ x, y });
            }
        }
    }
    PRINT("Frame: line_list size=%zu, points per lane=[%zu,%zu,%zu,%zu]\n",
          result.line_list.size(),
          result.line_list[0].size(), result.line_list[1].size(), result.line_list[2].size(), result.line_list[3].size());

    const auto& t_post_process1 = std::chrono::steady_clock::now();

    result.crop.x = (std::max)(0, crop_x);
    result.crop.y = (std::max)(0, crop_y);
    result.crop.w = (std::min)(crop_w, original_mat.cols - result.crop.x);
    result.crop.h = (std::min)(crop_h, original_mat.rows - result.crop.y);
    result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
    result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
    result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;

    return kRetOk;
}