cmake_minimum_required(VERSION 3.20)

set(LibraryName "ImageProcessor")

# Find CUDA and TensorRT
find_package(CUDA REQUIRED)
find_library(TENSORRT_LIBRARY nvinfer PATHS /usr/lib/x86_64-linux-gnu)
find_library(TENSORRT_ONNX_PARSER nvonnxparser PATHS /usr/lib/x86_64-linux-gnu)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS /usr/include/x86_64-linux-gnu)

# Create library
add_library(${LibraryName} STATIC
    image_processor.cpp
    image_processor.h
    image_processor_if.cpp
    image_processor_if.h
    detection_engine.cpp
    detection_engine.h
    lane_engine.cpp
    lane_engine.h
    semantic_segmentation_engine.cpp
    semantic_segmentation_engine.h
    depth_engine.cpp
    depth_engine.h
    lane_detection.cpp
    lane_detection.h
    object_detection.cpp
    object_detection.h
)

# Sub modules
## CommonHelper
target_include_directories(${LibraryName} PUBLIC ${PROJECT_SOURCE_DIR}/common_helper)
target_link_libraries(${LibraryName} PUBLIC CommonHelper)

## InferenceHelper
target_include_directories(${LibraryName} PUBLIC ${INFERENCE_HELPER_DIR})
target_link_libraries(${LibraryName} PUBLIC InferenceHelper)

## OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${LibraryName} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${LibraryName} PUBLIC ${OpenCV_LIBS})

## OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(${LibraryName} PUBLIC OpenMP::OpenMP_CXX)

## CUDA and TensorRT
if(INFERENCE_HELPER_ENABLE_TENSORRT)
    target_include_directories(${LibraryName} PUBLIC ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(${LibraryName} PUBLIC
        ${CUDA_LIBRARIES}
        cudart
        ${TENSORRT_LIBRARY}
        ${TENSORRT_ONNX_PARSER}
    )
endif()