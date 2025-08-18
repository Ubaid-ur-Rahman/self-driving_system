cmake_minimum_required(VERSION 3.10)
project(InferenceHelperProject)
set(LibraryName "InferenceHelper")

# Enable C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}, include dirs: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCV not found. Please ensure OpenCV is installed.")
endif()

# Find TensorFlow Lite
set(TFLITE_ROOT "/home/ubaid/tensorflow")
set(TFLITE_BUILD "${TFLITE_ROOT}/bazel-bin/tensorflow/lite")
find_library(TFLITE_LIBRARY tensorflowlite PATHS ${TFLITE_BUILD} NO_DEFAULT_PATH)
if(TFLITE_LIBRARY)
    message(STATUS "TensorFlow Lite found: ${TFLITE_LIBRARY}")
else()
    message(WARNING "TensorFlow Lite not found at ${TFLITE_BUILD}. TFLite support may be limited.")
endif()

# Set CUDA path manually
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.8)
set(CUDA_LIBRARIES /usr/local/cuda-12.8/lib64/libcudart.so)
link_directories(/usr/local/cuda-12.8/lib64)

# Find CUDA
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}, include dirs: ${CUDA_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "CUDA not found. Required for TensorRT.")
endif()

# Find TensorRT
find_library(TENSORRT_LIBRARY nvinfer PATHS /usr/lib/x86_64-linux-gnu)
find_library(TENSORRT_ONNX_PARSER nvonnxparser PATHS /usr/lib/x86_64-linux-gnu)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS /usr/include/x86_64-linux-gnu)

if(TENSORRT_LIBRARY AND TENSORRT_ONNX_PARSER AND TENSORRT_INCLUDE_DIR)
    set(TENSORRT_FOUND TRUE)
    message(STATUS "TensorRT found: ${TENSORRT_LIBRARY}, ONNX parser: ${TENSORRT_ONNX_PARSER}, include dir: ${TENSORRT_INCLUDE_DIR}")
else()
    set(TENSORRT_FOUND FALSE)
    message(WARNING "TensorRT not found. TensorRT support will be disabled.")
endif()

# Create the InferenceHelper static library
add_library(${LibraryName} STATIC
    inference_helper.cpp
)

# Include directories
target_include_directories(${LibraryName} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${TFLITE_ROOT}
    ${TFLITE_ROOT}/tensorflow/lite
    ${TFLITE_ROOT}/tensorflow/lite/tools/make/downloads
    ${TFLITE_ROOT}/tensorflow/lite/tools/make/downloads/flatbuffers/include
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(${LibraryName} PUBLIC
    ${OpenCV_LIBS}
)

# TensorFlow Lite support
if(INFERENCE_HELPER_ENABLE_TFLITE AND TFLITE_LIBRARY)
    target_link_libraries(${LibraryName} PUBLIC ${TFLITE_LIBRARY})
    target_compile_definitions(${LibraryName} PUBLIC -DUSE_TFLITE)
    if(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK)
        target_compile_definitions(${LibraryName} PUBLIC -DUSE_TFLITE_DELEGATE_XNNPACK)
    endif()
endif()

# TensorRT support
if(TENSORRT_FOUND AND INFERENCE_HELPER_ENABLE_TENSORRT)
    target_link_libraries(${LibraryName} PUBLIC
        ${TENSORRT_LIBRARY}
        ${TENSORRT_ONNX_PARSER}
        ${CUDA_LIBRARIES}
    )
    target_compile_definitions(${LibraryName} PUBLIC -DUSE_TENSORRT)
endif()

# Print include directories and linked libraries for debugging
get_target_property(INCLUDE_DIRS ${LibraryName} INCLUDE_DIRECTORIES)
message(STATUS "Include directories for ${LibraryName}: ${INCLUDE_DIRS}")
get_target_property(LINK_LIBS ${LibraryName} LINK_LIBRARIES)
message(STATUS "Linked libraries for ${LibraryName}: ${LINK_LIBS}")