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
#include <memory>
#include <filesystem> // Added for directory creation

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper.h"
#include "common_helper_cv.h"
#include "camera_model.h"
#include "bounding_box.h"
#include "detection_engine.h"
#include "tracker.h"
#include "lane_detection.h"
#include "depth_engine.h"

#include "image_processor_if.h"
#include "image_processor.h"

/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

#define COLOR_BG  CommonHelper::CreateCvColor(70, 70, 70)
static constexpr float kTopViewSizeRatio = 1.0f;
static constexpr int32_t kMaxFrames = 1000; // Safeguard against infinite loops
static constexpr int32_t kModelInputWidth = 192;
static constexpr int32_t kModelInputHeight = 320;

/*** Global variable ***/

/*** Function ***/
ImageProcessor::ImageProcessor()
{
    frame_cnt_ = 0;
    vanishment_y_ = 1280 / 2;
}

ImageProcessor::~ImageProcessor()
{
    mat_transform_.release();
}

int32_t ImageProcessor::Initialize(const ImageProcessorIf::InputParam& input_param)
{

    if (object_detection_.Initialize(input_param.work_dir, input_param.num_threads) != ObjectDetection::kRetOk) {
        object_detection_.Finalize();
        return kRetErr;
    }
    if (lane_detection_.Initialize(input_param.work_dir, input_param.num_threads) != LaneDetection::kRetOk) {
        lane_detection_.Finalize();
        object_detection_.Finalize();
        return kRetErr;
    }
    if (segmentation_engine_.Initialize(input_param.work_dir, input_param.num_threads) != SemanticSegmentationEngine::kRetOk) {
        segmentation_engine_.Finalize();
        lane_detection_.Finalize();
        object_detection_.Finalize();
        return kRetErr;
    }
    if (depth_engine_.Initialize(input_param.work_dir, input_param.num_threads) != DepthEngine::kRetOk) {
        depth_engine_.Finalize();
        segmentation_engine_.Finalize();
        lane_detection_.Finalize();
        object_detection_.Finalize();
        return kRetErr;
    }
    frame_cnt_ = 0;
    vanishment_y_ = 1280 / 2;
    return kRetOk;
}

int32_t ImageProcessor::Finalize(void)
{
    mat_transform_.release();
    int32_t ret = kRetOk;
    if (object_detection_.Finalize() != ObjectDetection::kRetOk) ret = kRetErr;
    if (lane_detection_.Finalize() != LaneDetection::kRetOk) ret = kRetErr;
    if (segmentation_engine_.Finalize() != SemanticSegmentationEngine::kRetOk) ret = kRetErr;
    if (depth_engine_.Finalize() != DepthEngine::kRetOk) ret = kRetErr;
    return ret;
}

int32_t ImageProcessor::Command(int32_t cmd)
{
    switch (cmd) {
    case 0:
    default:
        PRINT_E("command(%d) is not supported\n", cmd);
        return kRetErr;
    }
    return kRetOk;
}

void ImageProcessor::SaveBoundingBoxes(const std::vector<BoundingBox>& bbox_list, const std::string& filename)
{
    PRINT("Saving %zu bounding boxes to %s\n", bbox_list.size(), filename.c_str());
    std::ofstream ofs(filename, std::ios::app);
    if (!ofs.is_open()) {
        PRINT_E("Failed to open %s for writing\n", filename.c_str());
        return;
    }
    for (const auto& bbox : bbox_list) {
        ofs << bbox.class_id << "," << bbox.label << "," << bbox.score << ","
            << bbox.x << "," << bbox.y << "," << bbox.w << "," << bbox.h << "\n";
    }
}

int32_t ImageProcessor::Process(const cv::Mat& mat_original, ImageProcessorIf::Result& result)
{
    if (frame_cnt_ >= kMaxFrames) {
        PRINT_E("Max frame limit (%d) reached, stopping processing\n", kMaxFrames);
        return kRetErr;
    }

    if (mat_original.empty()) {
        PRINT_E("Input image is empty\n");
        return kRetErr;
    }

    // Resize input to match model dimensions
    cv::Mat mat_resized;
    cv::resize(mat_original, mat_resized, cv::Size(kModelInputWidth, kModelInputHeight), 0, 0, cv::INTER_AREA);
    PRINT("[DEBUG] Frame %d: Resized input to %dx%d, type=%d\n", frame_cnt_, mat_resized.cols, mat_resized.rows, mat_resized.type());

    // Reset camera with resized dimensions
    if (frame_cnt_ == 0) {
        ResetCamera(mat_resized.cols, mat_resized.rows); // Use actual resized dimensions
    }

    // Run inference on resized image
    if (object_detection_.Process(mat_resized, mat_transform_, camera_real_) != ObjectDetection::kRetOk) {
        PRINT_E("Object detection failed\n");
        return kRetErr;
    }
    if (lane_detection_.Process(mat_resized, mat_transform_, camera_real_) != LaneDetection::kRetOk) {
        PRINT_E("Lane detection failed\n");
        return kRetErr;
    }
    SemanticSegmentationEngine::Result segmentation_result;
    if (segmentation_engine_.Process(mat_resized, segmentation_result) != SemanticSegmentationEngine::kRetOk) {
        PRINT_E("Segmentation failed\n");
        return kRetErr;
    }
    DepthEngine::Result depth_result;
    if (depth_engine_.Process(mat_resized, depth_result) != DepthEngine::kRetOk) {
        PRINT_E("Depth estimation failed\n");
        return kRetErr;
    }

    // Save results to disk
    std::string frame_id_str = std::to_string(frame_cnt_);
    CommonHelper::SaveImage(mat_original, "output/frame_" + frame_id_str + ".jpg"); // Save original for reference

    // Save object detection results
    const auto& bboxes = object_detection_.GetBoundingBoxes();
    SaveBoundingBoxes(bboxes, "output/bboxes_" + frame_id_str + ".csv");

    // Save lane detection results
    const auto& lane_points_list = lane_detection_.GetLanePoints();
    std::ofstream lane_file("output/lanes_" + frame_id_str + ".csv");
    if (!lane_file.is_open()) {
        PRINT_E("Failed to open lanes_%s.csv for writing\n", frame_id_str.c_str());
    } else {
        for (size_t lane_idx = 0; lane_idx < lane_points_list.size(); ++lane_idx) {
            for (const auto& pt : lane_points_list[lane_idx]) {
                lane_file << lane_idx << "," << pt.x << "," << pt.y << "\n";
            }
        }
    }

    // Save segmentation and depth results
    cv::Mat mat_segmentation;
    if (!segmentation_result.image_combined.empty()) {
        // Resize segmentation output to match model input size
        cv::resize(segmentation_result.image_combined, mat_segmentation, cv::Size(kModelInputWidth, kModelInputHeight), 0, 0, cv::INTER_NEAREST);
        PRINT("[DEBUG] Frame %d: mat_segmentation resized from image_combined size=%dx%d, type=%d\n", frame_cnt_, mat_segmentation.cols, mat_segmentation.rows, mat_segmentation.type());
    } else if (!segmentation_result.image_list.empty()) {
        DrawSegmentation(mat_segmentation, segmentation_result);
        PRINT("[DEBUG] Frame %d: mat_segmentation from DrawSegmentation size=%dx%d, type=%d\n", frame_cnt_, mat_segmentation.cols, mat_segmentation.rows, mat_segmentation.type());
    } else {
        PRINT_E("Segmentation result is empty\n");
        mat_segmentation = cv::Mat::zeros(cv::Size(kModelInputWidth, kModelInputHeight), CV_8UC3);
    }
    if (!mat_segmentation.empty()) {
        CommonHelper::SaveImage(mat_segmentation, "output/seg_" + frame_id_str + ".png");
        result.mat_output_segmentation = mat_segmentation.clone(); // Assign for main.cpp
    }

    // --- Depth output ---
    cv::Mat mat_depth;
    if (!depth_result.mat_out.empty()) {
        DrawDepth(mat_depth, depth_result);
        PRINT("[DEBUG] Frame %d: mat_depth size=%dx%d, type=%d\n", frame_cnt_, mat_depth.cols, mat_depth.rows, mat_depth.type());
        CommonHelper::SaveImage(mat_depth, "output/depth_" + frame_id_str + ".png");
        result.mat_output_depth = mat_depth.clone(); // Assign
    } else {
        PRINT_E("Depth result is empty\n");
        mat_depth = cv::Mat::zeros(cv::Size(kModelInputWidth, kModelInputHeight), CV_8UC3);
    }

    // --- Topview output ---
    cv::Mat mat_topview;
    if (!mat_segmentation.empty()) {
        // Use the resized segmentation visual for topview creation
        CreateTopViewMat(mat_segmentation, mat_topview);
        PRINT("[DEBUG] Frame %d: mat_topview after CreateTopViewMat size=%dx%d, type=%d\n", frame_cnt_, mat_topview.cols, mat_topview.rows, mat_topview.type());
    } else {
        // Fallback empty topview
        mat_topview = cv::Mat::zeros(cv::Size(kModelInputWidth, kModelInputHeight), CV_8UC3);
        PRINT("[DEBUG] Frame %d: mat_topview fallback size=%dx%d, type=%d\n", frame_cnt_, mat_topview.cols, mat_topview.rows, mat_topview.type());
    }
    if (!mat_topview.empty()) {
        CommonHelper::SaveImage(mat_topview, "output/topview_" + frame_id_str + ".png");
        result.mat_output_topview = mat_topview.clone();
    }

    // For simplicity, let result.mat_output be the original image resized
    result.mat_output = mat_resized.clone(); // or build a richer overlay if desired

    // Update internal status
    frame_cnt_++;

    // Return results
    result.time_pre_process = object_detection_.GetTimePreProcess() + lane_detection_.GetTimePreProcess() + segmentation_result.time_pre_process + depth_result.time_pre_process;
    result.time_inference = object_detection_.GetTimeInference() + lane_detection_.GetTimeInference() + segmentation_result.time_inference + depth_result.time_inference;
    result.time_post_process = object_detection_.GetTimePostProcess() + lane_detection_.GetTimePostProcess() + segmentation_result.time_post_process + depth_result.time_post_process;

    // Release engine results
    for (auto& img : segmentation_result.image_list) {
        img.release();
    }
    segmentation_result.image_combined.release();
    depth_result.mat_out.release();
    mat_resized.release();
    mat_segmentation.release();
    mat_depth.release();
    mat_topview.release();

    PRINT("[DEBUG] Frame %d: Returning result mats: output=(%d,%d) topview=(%d,%d) depth=(%d,%d) segmentation=(%d,%d)\n",
          frame_cnt_, result.mat_output.cols, result.mat_output.rows,
          result.mat_output_topview.cols, result.mat_output_topview.rows,
          result.mat_output_depth.cols, result.mat_output_depth.rows,
          result.mat_output_segmentation.cols, result.mat_output_segmentation.rows);
    PRINT("[DEBUG] Frame %d: Processing complete for frame %d\n", frame_cnt_, frame_cnt_);

    return kRetOk;
}

void ImageProcessor::DrawDepth(cv::Mat& mat, const DepthEngine::Result& depth_result)
{
    if (!depth_result.mat_out.empty()) {
        cv::applyColorMap(depth_result.mat_out, mat, cv::COLORMAP_PLASMA);
        PRINT("[DEBUG] Frame %d: DrawDepth applied, mat size=%dx%d, type=%d\n", frame_cnt_, mat.cols, mat.rows, mat.type());
    } else {
        PRINT("[DEBUG] Frame %d: DrawDepth skipped, depth_result.mat_out is empty\n", frame_cnt_);
        mat = cv::Mat::zeros(cv::Size(kModelInputWidth, kModelInputHeight), CV_8UC3);
    }
}

void ImageProcessor::DrawSegmentation(cv::Mat& mat_segmentation, const SemanticSegmentationEngine::Result& segmentation_result)
{
    if (!segmentation_result.image_combined.empty()) {
        cv::resize(segmentation_result.image_combined, mat_segmentation, cv::Size(kModelInputWidth, kModelInputHeight), 0, 0, cv::INTER_NEAREST);
        PRINT("[DEBUG] Frame %d: DrawSegmentation from image_combined size=%dx%d, type=%d\n", frame_cnt_, mat_segmentation.cols, mat_segmentation.rows, mat_segmentation.type());
    } else if (!segmentation_result.image_list.empty()) {
        std::vector<cv::Mat> mat_segmentation_list(4, cv::Mat());
#pragma omp parallel for
        for (int32_t i = 0; i < static_cast<int32_t>(segmentation_result.image_list.size()); i++) {
            cv::Mat mat_fp32_3;
            cv::cvtColor(segmentation_result.image_list[i], mat_fp32_3, cv::COLOR_GRAY2BGR);
            cv::multiply(mat_fp32_3, GetColorForSegmentation(i), mat_fp32_3);
            mat_fp32_3.convertTo(mat_fp32_3, CV_8UC3, 1, 0);
            cv::resize(mat_fp32_3, mat_segmentation_list[i], cv::Size(kModelInputWidth, kModelInputHeight), 0, 0, cv::INTER_NEAREST);
        }

        mat_segmentation = cv::Mat::zeros(cv::Size(kModelInputWidth, kModelInputHeight), CV_8UC3);
        PRINT("[DEBUG] Frame %d: DrawSegmentation initialized zero mat size=%dx%d, type=%d\n", frame_cnt_, mat_segmentation.cols, mat_segmentation.rows, mat_segmentation.type());
        for (int32_t i = 0; i < static_cast<int32_t>(mat_segmentation_list.size()); i++) {
            if (!mat_segmentation_list[i].empty()) {
                cv::add(mat_segmentation, mat_segmentation_list[i], mat_segmentation);
            }
        }
        for (auto& mat : mat_segmentation_list) {
            mat.release();
        }
        PRINT("[DEBUG] Frame %d: DrawSegmentation after add size=%dx%d, type=%d\n", frame_cnt_, mat_segmentation.cols, mat_segmentation.rows, mat_segmentation.type());
    } else {
        mat_segmentation = cv::Mat::zeros(cv::Size(kModelInputWidth, kModelInputHeight), CV_8UC3);
    }
}

void ImageProcessor::DrawFps(cv::Mat& mat, double time_inference, double time_draw, cv::Point pos, double font_scale, int32_t thickness, cv::Scalar color_front, cv::Scalar color_back, bool is_text_on_rect)
{
    char text[64];
    static auto time_previous = std::chrono::steady_clock::now();
    auto time_now = std::chrono::steady_clock::now();
    double fps = 1e9 / (time_now - time_previous).count();
    time_previous = time_now;
    snprintf(text, sizeof(text), "FPS: %.1f, Inference: %.1f [ms], Draw: %.1f [ms]", fps, time_inference, time_draw);
    CommonHelper::DrawText(mat, text, cv::Point(0, 0), 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(180, 180, 180), true);
}

cv::Scalar ImageProcessor::GetColorForSegmentation(int32_t id)
{
    switch (id) {
    default:
    case 0: /* BG */
        return COLOR_BG;
    case 1: /* road */
        return CommonHelper::CreateCvColor(255, 0, 0);
    case 2: /* curbs */
        return CommonHelper::CreateCvColor(0, 255, 0);
    case 3: /* marks */
        return CommonHelper::CreateCvColor(0, 0, 255);
    }
}

void ImageProcessor::ResetCamera(int32_t width, int32_t height, float fov_deg)
{
    if (width > 0 && height > 0 && fov_deg > 0) {
        camera_real_.SetIntrinsic(width, height, FocalLength(width, fov_deg));
        camera_top_.SetIntrinsic(static_cast<int32_t>(width * kTopViewSizeRatio), static_cast<int32_t>(height * kTopViewSizeRatio), FocalLength(static_cast<int32_t>(width * kTopViewSizeRatio), fov_deg));
    }
    camera_real_.SetExtrinsic(
        { 0.0f, 0.0f, 0.0f },
        { 0.0f, -1.5f, 0.0f }, true);
    camera_top_.SetExtrinsic(
        { 90.0f, 0.0f, 0.0f },
        { 0.0f, -8.0f, 11.0f }, true);
    CreateTransformMat();
    vanishment_y_ = std::max(0, std::min(height, camera_real_.EstimateVanishmentY()));
    PRINT("[DEBUG] Frame %d: ResetCamera with width=%d, height=%d, camera_real=%dx%d, camera_top=%dx%d\n",
          frame_cnt_, width, height, camera_real_.width, camera_real_.height, camera_top_.width, camera_top_.height);
}

void ImageProcessor::GetCameraParameter(float& focal_length, std::array<float, 3>& real_rvec, std::array<float, 3>& real_tvec, std::array<float, 3>& top_rvec, std::array<float, 3>& top_tvec)
{
    focal_length = camera_real_.fx();
    camera_real_.GetExtrinsic(real_rvec, real_tvec);
    camera_top_.GetExtrinsic(top_rvec, top_tvec);
}

void ImageProcessor::SetCameraParameter(float focal_length, const std::array<float, 3>& real_rvec, const std::array<float, 3>& real_tvec, const std::array<float, 3>& top_rvec, const std::array<float, 3>& top_tvec)
{
    camera_real_.fx() = focal_length;
    camera_real_.fy() = focal_length;
    camera_top_.fx() = focal_length * kTopViewSizeRatio;
    camera_top_.fy() = focal_length * kTopViewSizeRatio;
    camera_real_.SetExtrinsic(real_rvec, real_tvec);
    camera_top_.SetExtrinsic(top_rvec, top_tvec);
    CreateTransformMat();
    vanishment_y_ = std::max(0, std::min(camera_real_.height, camera_real_.EstimateVanishmentY()));
    PRINT("[DEBUG] Frame %d: SetCameraParameter with focal_length=%.1f, camera_real=%dx%d, camera_top=%dx%d\n",
          frame_cnt_, focal_length, camera_real_.width, camera_real_.height, camera_top_.width, camera_top_.height);
}

void ImageProcessor::CreateTransformMat()
{
    std::vector<cv::Point3f> object_point_list = {
        { -1.0f, 0, 10.0f },
        { 1.0f, 0, 10.0f },
        { -1.0f, 0, 3.0f },
        { 1.0f, 0, 3.0f },
    };
    std::vector<cv::Point2f> image_point_real_list;
    cv::projectPoints(object_point_list, camera_real_.rvec, camera_real_.tvec, camera_real_.K, camera_real_.dist_coeff, image_point_real_list);

    std::vector<cv::Point2f> image_point_top_list;
    cv::projectPoints(object_point_list, camera_top_.rvec, camera_top_.tvec, camera_top_.K, camera_top_.dist_coeff, image_point_top_list);

    mat_transform_ = cv::getPerspectiveTransform(&image_point_real_list[0], &image_point_top_list[0]);
    PRINT("[DEBUG] Frame %d: CreateTransformMat, mat_transform_ size=%dx%d\n", frame_cnt_, mat_transform_.cols, mat_transform_.rows);
}

void ImageProcessor::CreateTopViewMat(const cv::Mat& mat_original, cv::Mat& mat_topview)
{
    if (mat_original.empty() || mat_transform_.empty()) {
        PRINT_E("Invalid input or transform matrix for top view\n");
        mat_topview = cv::Mat::zeros(cv::Size(kModelInputWidth, kModelInputHeight), CV_8UC3);
        PRINT("[DEBUG] Frame %d: CreateTopViewMat fallback size=%dx%d, type=%d\n", frame_cnt_, mat_topview.cols, mat_topview.rows, mat_topview.type());
        return;
    }

    // Ensure input dimensions match expected
    cv::Mat mat_input = mat_original;
    if (mat_input.cols != kModelInputWidth || mat_input.rows != kModelInputHeight) {
        cv::resize(mat_input, mat_input, cv::Size(kModelInputWidth, kModelInputHeight), 0, 0, cv::INTER_NEAREST);
        PRINT("[DEBUG] Frame %d: Resized mat_input to %dx%d for top view\n", frame_cnt_, mat_input.cols, mat_input.rows);
    }

    // Determine the bounding box of the transformed points
    std::vector<cv::Point2f> corners(4);
    corners[0] = cv::Point2f(0, 0);
    corners[1] = cv::Point2f(mat_input.cols - 1, 0);
    corners[2] = cv::Point2f(0, mat_input.rows - 1);
    corners[3] = cv::Point2f(mat_input.cols - 1, mat_input.rows - 1);
    std::vector<cv::Point2f> transformed_corners(4);
    cv::perspectiveTransform(corners, transformed_corners, mat_transform_);

    // Find the bounding box of the transformed corners
    float min_x = std::min({transformed_corners[0].x, transformed_corners[1].x, transformed_corners[2].x, transformed_corners[3].x});
    float max_x = std::max({transformed_corners[0].x, transformed_corners[1].x, transformed_corners[2].x, transformed_corners[3].x});
    float min_y = std::min({transformed_corners[0].y, transformed_corners[1].y, transformed_corners[2].y, transformed_corners[3].y});
    float max_y = std::max({transformed_corners[0].y, transformed_corners[1].y, transformed_corners[2].y, transformed_corners[3].y});

    // Ensure valid bounds
    min_x = std::max(min_x, 0.0f);
    min_y = std::max(min_y, 0.0f);
    int topview_width = static_cast<int>(max_x - min_x + 10);
    int topview_height = static_cast<int>(max_y - min_y + 10);
    topview_width = std::max(topview_width, kModelInputWidth);
    topview_height = std::max(topview_height, kModelInputHeight);

    // Validate dimensions
    if (topview_width <= 0 || topview_height <= 0) {
        PRINT_E("Invalid topview dimensions: width=%d, height=%d\n", topview_width, topview_height);
        mat_topview = cv::Mat::zeros(cv::Size(kModelInputWidth, kModelInputHeight), CV_8UC3);
        return;
    }

    mat_topview = cv::Mat(cv::Size(topview_width, topview_height), CV_8UC3, COLOR_BG);
    PRINT("[DEBUG] Frame %d: CreateTopViewMat initialized mat_topview size=%dx%d, type=%d\n", frame_cnt_, mat_topview.cols, mat_topview.rows, mat_topview.type());

    // Adjust the transformation to map to the new origin
    cv::Mat translation = (cv::Mat_<float>(3, 3) << 1, 0, -min_x,
                                                 0, 1, -min_y,
                                                 0, 0, 1);
    cv::Mat adjusted_transform = translation * mat_transform_;

    // Perform the warp with the adjusted transform
    cv::warpPerspective(mat_input, mat_topview, adjusted_transform, mat_topview.size(), cv::INTER_NEAREST | cv::WARP_INVERSE_MAP);
    PRINT("[DEBUG] Frame %d: CreateTopViewMat after warpPerspective size=%dx%d, type=%d\n", frame_cnt_, mat_topview.cols, mat_topview.rows, mat_topview.type());

    // Draw depth lines
    static constexpr int32_t kDepthInterval = 5;
    static constexpr int32_t kHorizontalRange = 10;
    std::vector<cv::Point3f> object_point_list;
    for (float z = 0; z <= 30; z += kDepthInterval) {
        object_point_list.push_back(cv::Point3f(-kHorizontalRange, 0, z));
        object_point_list.push_back(cv::Point3f(kHorizontalRange, 0, z));
    }
    std::vector<cv::Point2f> image_point_list;
    cv::projectPoints(object_point_list, camera_top_.rvec, camera_top_.tvec, camera_top_.K, camera_top_.dist_coeff, image_point_list);
    for (int32_t i = 0; i < static_cast<int32_t>(image_point_list.size()); i++) {
        if (i % 2 != 0) {
            cv::Point2f p1 = image_point_list[i - 1];
            cv::Point2f p2 = image_point_list[i];
            // Clip points to mat_topview bounds
            p1.x = std::max(0.0f, std::min(p1.x, static_cast<float>(mat_topview.cols - 1)));
            p1.y = std::max(0.0f, std::min(p1.y, static_cast<float>(mat_topview.rows - 1)));
            p2.x = std::max(0.0f, std::min(p2.x, static_cast<float>(mat_topview.cols - 1)));
            p2.y = std::max(0.0f, std::min(p2.y, static_cast<float>(mat_topview.rows - 1)));
            cv::line(mat_topview, p1, p2, cv::Scalar(255, 255, 255));
        } else {
            cv::Point2f p = image_point_list[i];
            p.x = std::max(0.0f, std::min(p.x, static_cast<float>(mat_topview.cols - 1)));
            p.y = std::max(0.0f, std::min(p.y, static_cast<float>(mat_topview.rows - 1)));
            CommonHelper::DrawText(mat_topview, std::to_string(i / 2 * kDepthInterval) + "[m]", p, 0.5, 2, CommonHelper::CreateCvColor(0, 0, 0), CommonHelper::CreateCvColor(255, 255, 255), false);
        }
    }
}