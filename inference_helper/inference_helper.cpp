#include "inference_helper.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <map>

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        if (severity != nvinfer1::ILogger::Severity::kINFO) {
            std::cerr << "TensorRT: " << msg << std::endl;
        }
    }
};

// Placeholder for TensorFlow Lite implementation
class InferenceHelperTensorflowLite : public InferenceHelper {
public:
    int32_t SetNumThreads(int32_t num_threads) override {
        std::cout << "Setting threads for TFLite: " << num_threads << std::endl;
        return kRetOk;
    }

    int32_t Initialize(const std::string& model_filename,
                       std::vector<InputTensorInfo>& input_tensor_info_list,
                       std::vector<OutputTensorInfo>& output_tensor_info_list) override {
        std::cout << "Initializing TFLite model: " << model_filename << std::endl;
        return kRetOk;
    }

    int32_t PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list) override {
        std::cout << "Preprocessing for TFLite" << std::endl;
        return kRetOk;
    }

    int32_t Process(std::vector<OutputTensorInfo>& output_tensor_info_list) override {
        std::cout << "Running TFLite inference" << std::endl;
        return kRetOk;
    }

    int32_t Finalize() override {
        std::cout << "Finalizing TFLite" << std::endl;
        return kRetOk;
    }
};

// TensorRT implementation
class InferenceHelperTensorrt : public InferenceHelper {
public:
    InferenceHelperTensorrt() : m_engine(nullptr), m_context(nullptr), m_cuda_stream(nullptr) {}

    ~InferenceHelperTensorrt() override {
        Finalize();
    }

    int32_t SetNumThreads(int32_t num_threads) override {
        std::cout << "TensorRT: Number of threads not applicable, using CUDA streams" << std::endl;
        return kRetOk;
    }

    int32_t Initialize(const std::string& model_filename,
                       std::vector<InputTensorInfo>& input_tensor_info_list,
                       std::vector<OutputTensorInfo>& output_tensor_info_list) override {
        Logger logger;
        try {
            std::cout << "Initializing TensorRT model: " << model_filename << std::endl;
            // Read ONNX model file
            std::ifstream file(model_filename, std::ios::binary | std::ios::ate);
            if (!file.is_open()) {
                std::cerr << "Failed to open model file: " << model_filename << std::endl;
                return kRetErr;
            }
            size_t size = file.tellg();
            file.seekg(0, std::ios::beg);
            std::vector<char> model_data(size);
            file.read(model_data.data(), size);
            file.close();
            std::cout << "Read model file: size=" << size << " bytes" << std::endl;

            // Initialize TensorRT
            auto builder = std::unique_ptr<nvinfer1::IBuilder>(
                nvinfer1::createInferBuilder(logger));
            if (!builder) {
                std::cerr << "Failed to create TensorRT builder" << std::endl;
                return kRetErr;
            }

            // Create network with explicit batch size
            auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
                builder->createNetworkV2(0));
            if (!network) {
                std::cerr << "Failed to create TensorRT network" << std::endl;
                return kRetErr;
            }

            auto parser = std::unique_ptr<nvonnxparser::IParser>(
                nvonnxparser::createParser(*network, logger));
            if (!parser->parse(model_data.data(), model_data.size())) {
                std::cerr << "Failed to parse ONNX model" << std::endl;
                return kRetErr;
            }

            auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
                builder->createBuilderConfig());
            if (!config) {
                std::cerr << "Failed to create TensorRT config" << std::endl;
                return kRetErr;
            }

            // Enable FP16 precision if supported
            if (builder->platformHasFastFp16()) {
                config->setFlag(nvinfer1::BuilderFlag::kFP16);
                std::cout << "Enabled FP16 precision" << std::endl;
            }

            // Set memory pool limit
            config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30); // 1GB
            std::cout << "Set workspace memory limit to 1GB" << std::endl;

            // Build serialized engine
            auto plan = std::unique_ptr<nvinfer1::IHostMemory>(
                builder->buildSerializedNetwork(*network, *config));
            if (!plan) {
                std::cerr << "Failed to build TensorRT serialized network" << std::endl;
                return kRetErr;
            }
            std::cout << "Built serialized network: size=" << plan->size() << " bytes" << std::endl;

            // Create runtime and deserialize engine
            auto runtime = std::unique_ptr<nvinfer1::IRuntime>(
                nvinfer1::createInferRuntime(logger));
            if (!runtime) {
                std::cerr << "Failed to create TensorRT runtime" << std::endl;
                return kRetErr;
            }

            m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(
                runtime->deserializeCudaEngine(plan->data(), plan->size()),
                [](nvinfer1::ICudaEngine* engine) { delete engine; });
            if (!m_engine) {
                std::cerr << "Failed to deserialize TensorRT engine" << std::endl;
                return kRetErr;
            }
            std::cout << "Deserialized TensorRT engine" << std::endl;

            m_context = std::shared_ptr<nvinfer1::IExecutionContext>(
                m_engine->createExecutionContext(),
                [](nvinfer1::IExecutionContext* ctx) { delete ctx; });
            if (!m_context) {
                std::cerr << "Failed to create TensorRT execution context" << std::endl;
                return kRetErr;
            }
            std::cout << "Created TensorRT execution context" << std::endl;

            // Create CUDA stream
            if (cudaStreamCreate(&m_cuda_stream) != cudaSuccess) {
                std::cerr << "Failed to create CUDA stream" << std::endl;
                return kRetErr;
            }
            std::cout << "Created CUDA stream: " << m_cuda_stream << std::endl;

            // Store input/output tensor bindings
            m_input_tensor_info_list = input_tensor_info_list;
            m_output_tensor_info_list = output_tensor_info_list;

            // Map tensor names to indices and populate output tensor dimensions
            for (int i = 0; i < network->getNbInputs(); ++i) {
                auto input = network->getInput(i);
                m_tensor_name_to_index[input->getName()] = i;
                std::cout << "Input tensor: " << input->getName() << " at index " << i << std::endl;
            }
            for (int i = 0; i < network->getNbOutputs(); ++i) {
                auto output = network->getOutput(i);
                m_tensor_name_to_index[output->getName()] = i + network->getNbInputs();
                std::cout << "Output tensor: " << output->getName() << " at index " << (i + network->getNbInputs()) << std::endl;
                // Populate tensor_dims for each output
                for (auto& output_info : m_output_tensor_info_list) {
                    if (output_info.name == output->getName()) {
                        auto dims = output->getDimensions();
                        output_info.tensor_dims.clear();
                        for (int j = 0; j < dims.nbDims; ++j) {
                            output_info.tensor_dims.push_back(dims.d[j]);
                        }
                        std::cout << "Output tensor dims for " << output_info.name << ": [";
                        for (size_t j = 0; j < output_info.tensor_dims.size(); ++j) {
                            std::cout << output_info.tensor_dims[j] << (j < output_info.tensor_dims.size() - 1 ? "," : "");
                        }
                        std::cout << "]" << std::endl;
                    }
                }
            }

            m_buffers.resize(network->getNbInputs() + network->getNbOutputs(), nullptr);

            // Allocate GPU memory for inputs
            for (auto& input : m_input_tensor_info_list) {
                size_t size = 1;
                for (auto dim : input.tensor_dims) {
                    size *= dim;
                }
                size_t type_size = sizeof(float);
                if (input.tensor_type == kTensorTypeInt8) {
                    type_size = sizeof(int8_t);
                } else if (input.tensor_type == kTensorTypeFloat16) {
                    type_size = sizeof(float) / 2;
                } else if (input.tensor_type == kTensorTypeInt32) {
                    type_size = sizeof(int32_t);
                }
                cudaError_t cuda_status = cudaMalloc(&input.data, size * type_size);
                if (cuda_status != cudaSuccess || !input.data) {
                    std::cerr << "Failed to allocate CUDA memory for input: " << input.name 
                              << ", error: " << cudaGetErrorString(cuda_status) << std::endl;
                    return kRetErr;
                }
                std::cout << "Allocated CUDA memory for input: " << input.name << ", size=" << size * type_size << " bytes" << std::endl;
                auto it = m_tensor_name_to_index.find(input.name);
                if (it != m_tensor_name_to_index.end() && it->second < m_buffers.size()) {
                    m_buffers[it->second] = input.data;
                    std::cout << "Bound input tensor: " << input.name << " at index: " << it->second << std::endl;
                } else {
                    std::cerr << "Invalid input tensor name or index: " << input.name
                              << " (index: " << (it != m_tensor_name_to_index.end() ? std::to_string(it->second) : "not found")
                              << ", buffers size: " << m_buffers.size() << ")" << std::endl;
                    return kRetErr;
                }
            }

            // Allocate GPU memory for outputs
            for (auto& output : m_output_tensor_info_list) {
                size_t size = 1;
                auto it = m_tensor_name_to_index.find(output.name);
                if (it == m_tensor_name_to_index.end()) {
                    std::cerr << "Output tensor name not found: " << output.name << std::endl;
                    return kRetErr;
                }
                int binding_index = it->second;
                auto dims = m_engine->getTensorShape(output.name.c_str());
                for (int i = 0; i < dims.nbDims; ++i) {
                    size *= dims.d[i];
                }
                size_t type_size = sizeof(float);
                if (output.tensor_type == kTensorTypeInt8) {
                    type_size = sizeof(int8_t);
                } else if (output.tensor_type == kTensorTypeFloat16) {
                    type_size = sizeof(float) / 2;
                } else if (output.tensor_type == kTensorTypeInt32) {
                    type_size = sizeof(int32_t);
                }
                cudaError_t cuda_status = cudaMalloc(&output.data, size * type_size);
                if (cuda_status != cudaSuccess) {
                    std::cerr << "Failed to allocate CUDA memory for output: " << output.name 
                              << ", error: " << cudaGetErrorString(cuda_status) << std::endl;
                    return kRetErr;
                }
                m_buffers[binding_index] = output.data;
                std::cout << "Allocated CUDA memory for output: " << output.name << ", size=" << size * type_size << " bytes" << std::endl;
            }

            std::cout << "Initialized TensorRT model successfully" << std::endl;
            return kRetOk;
        } catch (const std::exception& e) {
            std::cerr << "TensorRT initialization error: " << e.what() << std::endl;
            return kRetErr;
        }
    }

    int32_t PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list) override {
        try {
            std::cout << "Starting TensorRT PreProcess" << std::endl;
            for (const auto& input : input_tensor_info_list) {
                std::cout << "Preprocessing input: " << input.name << ", dims=[";
                for (size_t i = 0; i < input.tensor_dims.size(); ++i) {
                    std::cout << input.tensor_dims[i] << (i < input.tensor_dims.size() - 1 ? "," : "");
                }
                std::cout << "], data=" << input.data << std::endl;

                if (input.data_type == InputTensorInfo::kDataTypeImage) {
                    // Preprocess image data
                    cv::Mat image;
                    if (input.image_info.is_bgr) {
                        image = cv::Mat(input.image_info.height, input.image_info.width, CV_8UC3, input.data);
                        std::cout << "Created image (BGR): width=" << input.image_info.width 
                                  << ", height=" << input.image_info.height << std::endl;
                    } else {
                        image = cv::Mat(input.image_info.height, input.image_info.width, CV_8UC3, input.data);
                        std::cout << "Created image (RGB): width=" << input.image_info.width 
                                  << ", height=" << input.image_info.height << std::endl;
                    }

                    // Debug: Save input image
                    cv::imwrite("output/debug_preprocess_input.png", image);
                    std::cout << "Saved debug_preprocess_input.png" << std::endl;

                    // Debug: Log first few pixel values
                    uint8_t* pixels = static_cast<uint8_t*>(input.data);
                    std::cout << "Input pixel sample: [" 
                              << static_cast<float>(pixels[0]) << ","
                              << static_cast<float>(pixels[1]) << ","
                              << static_cast<float>(pixels[2]) << ","
                              << static_cast<float>(pixels[3]) << "]" << std::endl;

                    // Crop
                    cv::Rect roi(input.image_info.crop_x, input.image_info.crop_y,
                                 input.image_info.crop_width, input.image_info.crop_height);
                    image = image(roi);
                    std::cout << "Cropped image: x=" << roi.x << ", y=" << roi.y 
                              << ", width=" << roi.width << ", height=" << roi.height << std::endl;

                    // Resize to tensor dimensions
                    cv::Mat resized;
                    cv::resize(image, resized, cv::Size(input.GetWidth(), input.GetHeight()));
                    std::cout << "Resized image: width=" << resized.cols << ", height=" << resized.rows << std::endl;

                    // Convert to float and normalize
                    cv::Mat float_image;
                    resized.convertTo(float_image, CV_32F);
                    std::cout << "Converted to float: type=" << float_image.type() << std::endl;

                    if (input.image_info.swap_color) {
                        cv::cvtColor(float_image, float_image, cv::COLOR_BGR2RGB);
                        std::cout << "Swapped color BGR to RGB" << std::endl;
                    }

                    // Apply normalization
                    for (int c = 0; c < 3; ++c) {
                        float_image.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int* position) -> void {
                            pixel[c] = (pixel[c] / 255.0f - input.normalize_mean[c]) / input.normalize_norm[c];
                        });
                    }
                    std::cout << "Applied normalization: mean=[" << input.normalize_mean[0] << ","
                              << input.normalize_mean[1] << "," << input.normalize_mean[2] << "], norm=["
                              << input.normalize_norm[0] << "," << input.normalize_norm[1] << ","
                              << input.normalize_norm[2] << "]" << std::endl;

                    // Debug: Log first few normalized values
                    std::cout << "Normalized pixel sample: [" 
                              << float_image.at<cv::Vec3f>(0,0)[0] << ","
                              << float_image.at<cv::Vec3f>(0,0)[1] << ","
                              << float_image.at<cv::Vec3f>(0,0)[2] << "]" << std::endl;

                    // Copy to GPU
                    auto it = m_tensor_name_to_index.find(input.name);
                    if (it == m_tensor_name_to_index.end()) {
                        std::cerr << "Input tensor name not found: " << input.name << std::endl;
                        return kRetErr;
                    }
                    void* input_buffer = m_buffers[it->second];
                    std::cout << "Copying to GPU buffer: index=" << it->second << ", buffer=" << input_buffer << std::endl;

                    if (input.is_nchw) {
                        // NCHW format
                        std::vector<float> nchw_data(input.GetHeight() * input.GetWidth() * 3);
                        for (int c = 0; c < 3; ++c) {
                            for (int h = 0; h < input.GetHeight(); ++h) {
                                for (int w = 0; w < input.GetWidth(); ++w) {
                                    nchw_data[c * input.GetHeight() * input.GetWidth() + h * input.GetWidth() + w] =
                                        float_image.at<cv::Vec3f>(h, w)[c];
                                }
                            }
                        }
                        std::cout << "Converted to NCHW, data size=" << nchw_data.size() * sizeof(float) << " bytes" << std::endl;
                        cudaError_t cuda_status = cudaMemcpyAsync(input_buffer, nchw_data.data(), 
                                                                nchw_data.size() * sizeof(float),
                                                                cudaMemcpyHostToDevice, m_cuda_stream);
                        if (cuda_status != cudaSuccess) {
                            std::cerr << "cudaMemcpyAsync failed for input: " << input.name 
                                      << ", error: " << cudaGetErrorString(cuda_status) << std::endl;
                            return kRetErr;
                        }
                        std::cout << "Copied NCHW data to GPU" << std::endl;
                    } else {
                        // NHWC format
                        cudaError_t cuda_status = cudaMemcpyAsync(input_buffer, float_image.data,
                                                                 input.GetHeight() * input.GetWidth() * 3 * sizeof(float),
                                                                 cudaMemcpyHostToDevice, m_cuda_stream);
                        if (cuda_status != cudaSuccess) {
                            std::cerr << "cudaMemcpyAsync failed for input: " << input.name 
                                      << ", error: " << cudaGetErrorString(cuda_status) << std::endl;
                            return kRetErr;
                        }
                        std::cout << "Copied NHWC data to GPU" << std::endl;
                    }
                } else {
                    // Handle blob data
                    auto it = m_tensor_name_to_index.find(input.name);
                    if (it == m_tensor_name_to_index.end()) {
                        std::cerr << "Input tensor name not found: " << input.name << std::endl;
                        return kRetErr;
                    }
                    size_t type_size = sizeof(float);
                    if (input.tensor_type == kTensorTypeInt8) {
                        type_size = sizeof(int8_t);
                    } else if (input.tensor_type == kTensorTypeFloat16) {
                        type_size = sizeof(float) / 2;
                    } else if (input.tensor_type == kTensorTypeInt32) {
                        type_size = sizeof(int32_t);
                    }
                    size_t size = 1;
                    for (auto dim : input.tensor_dims) {
                        size *= dim;
                    }
                    cudaError_t cuda_status = cudaMemcpyAsync(m_buffers[it->second], input.data,
                                                             size * type_size,
                                                             cudaMemcpyHostToDevice, m_cuda_stream);
                    if (cuda_status != cudaSuccess) {
                        std::cerr << "cudaMemcpyAsync failed for blob input: " << input.name 
                                  << ", error: " << cudaGetErrorString(cuda_status) << std::endl;
                        return kRetErr;
                    }
                    std::cout << "Copied blob data to GPU: size=" << size * type_size << " bytes" << std::endl;
                }
            }
            cudaError_t cuda_status = cudaStreamSynchronize(m_cuda_stream);
            if (cuda_status != cudaSuccess) {
                std::cerr << "cudaStreamSynchronize failed: " << cudaGetErrorString(cuda_status) << std::endl;
                return kRetErr;
            }
            std::cout << "TensorRT PreProcess completed" << std::endl;
            return kRetOk;
        } catch (const std::exception& e) {
            std::cerr << "TensorRT preprocessing error: " << e.what() << std::endl;
            return kRetErr;
        }
    }

    int32_t Process(std::vector<OutputTensorInfo>& output_tensor_info_list) override {
        try {
            std::cout << "Starting TensorRT inference" << std::endl;
            // Set input and output tensors for execution
            for (const auto& input : m_input_tensor_info_list) {
                auto it = m_tensor_name_to_index.find(input.name);
                if (it == m_tensor_name_to_index.end()) {
                    std::cerr << "Input tensor name not found: " << input.name << std::endl;
                    return kRetErr;
                }
                std::cout << "Setting input tensor address: " << input.name << " at index " << it->second 
                          << ", buffer=" << m_buffers[it->second] << std::endl;
                m_context->setInputTensorAddress(input.name.c_str(), m_buffers[it->second]);
            }
            for (const auto& output : m_output_tensor_info_list) {
                auto it = m_tensor_name_to_index.find(output.name);
                if (it == m_tensor_name_to_index.end()) {
                    std::cerr << "Output tensor name not found: " << output.name << std::endl;
                    return kRetErr;
                }
                std::cout << "Setting output tensor address: " << output.name << " at index " << it->second 
                          << ", buffer=" << m_buffers[it->second] << std::endl;
                m_context->setOutputTensorAddress(output.name.c_str(), m_buffers[it->second]);
            }

            // Execute inference
            std::cout << "Executing enqueueV3" << std::endl;
            if (!m_context->enqueueV3(m_cuda_stream)) {
                std::cerr << "TensorRT inference failed in enqueueV3" << std::endl;
                return kRetErr;
            }
            std::cout << "enqueueV3 completed" << std::endl;

            cudaError_t cuda_status = cudaStreamSynchronize(m_cuda_stream);
            if (cuda_status != cudaSuccess) {
                std::cerr << "cudaStreamSynchronize failed after enqueueV3: " << cudaGetErrorString(cuda_status) << std::endl;
                return kRetErr;
            }
            std::cout << "cudaStreamSynchronize completed after enqueueV3" << std::endl;

            // Copy outputs back to host
            for (auto& output : output_tensor_info_list) {
                auto it = m_tensor_name_to_index.find(output.name);
                if (it == m_tensor_name_to_index.end()) {
                    std::cerr << "Output tensor name not found: " << output.name << std::endl;
                    return kRetErr;
                }
                size_t size = 1;
                auto dims = m_engine->getTensorShape(output.name.c_str());
                for (int i = 0; i < dims.nbDims; ++i) {
                    size *= dims.d[i];
                }
                size_t type_size = sizeof(float);
                if (output.tensor_type == kTensorTypeInt8) {
                    type_size = sizeof(int8_t);
                } else if (output.tensor_type == kTensorTypeFloat16) {
                    type_size = sizeof(float) / 2;
                } else if (output.tensor_type == kTensorTypeInt32) {
                    type_size = sizeof(int32_t);
                }
                std::cout << "Copying output to host: " << output.name << ", size=" << size * type_size << " bytes" << std::endl;
                std::vector<char> host_output(size * type_size);
                cuda_status = cudaMemcpyAsync(host_output.data(), m_buffers[it->second],
                                             size * type_size, cudaMemcpyDeviceToHost, m_cuda_stream);
                if (cuda_status != cudaSuccess) {
                    std::cerr << "cudaMemcpyAsync failed for output: " << output.name 
                              << ", error: " << cudaGetErrorString(cuda_status) << std::endl;
                    return kRetErr;
                }
                cuda_status = cudaStreamSynchronize(m_cuda_stream);
                if (cuda_status != cudaSuccess) {
                    std::cerr << "cudaStreamSynchronize failed after output copy: " << cudaGetErrorString(cuda_status) << std::endl;
                    return kRetErr;
                }
                std::cout << "Copied output to host: " << output.name << std::endl;

                // Copy to output tensor data
                if (!output.data) {
                    output.data = malloc(size * type_size);
                    if (!output.data) {
                        std::cerr << "Failed to allocate host memory for output: " << output.name << std::endl;
                        return kRetErr;
                    }
                    std::cout << "Allocated host memory for output: " << output.name << std::endl;
                }
                memcpy(output.data, host_output.data(), size * type_size);
                // Update output tensor dimensions
                output.tensor_dims.clear();
                for (int i = 0; i < dims.nbDims; ++i) {
                    output.tensor_dims.push_back(dims.d[i]);
                }
                std::cout << "Updated output tensor dims for " << output.name << ": [";
                for (size_t i = 0; i < output.tensor_dims.size(); ++i) {
                    std::cout << output.tensor_dims[i] << (i < output.tensor_dims.size() - 1 ? "," : "");
                }
                std::cout << "]" << std::endl;
            }
            std::cout << "TensorRT inference completed" << std::endl;
            return kRetOk;
        } catch (const std::exception& e) {
            std::cerr << "TensorRT inference error: " << e.what() << std::endl;
            return kRetErr;
        }
    }

    int32_t Finalize() override {
        try {
            // Free CUDA memory
            for (auto& input : m_input_tensor_info_list) {
                if (input.data) {
                    cudaFree(input.data);
                    input.data = nullptr;
                    std::cout << "Freed CUDA memory for input: " << input.name << std::endl;
                }
            }
            for (auto& output : m_output_tensor_info_list) {
                if (output.data) {
                    cudaFree(output.data);
                    output.data = nullptr;
                    std::cout << "Freed CUDA memory for output: " << output.name << std::endl;
                }
            }
            m_buffers.clear();
            m_tensor_name_to_index.clear();

            // Destroy CUDA stream
            if (m_cuda_stream) {
                cudaStreamDestroy(m_cuda_stream);
                m_cuda_stream = nullptr;
                std::cout << "Destroyed CUDA stream" << std::endl;
            }

            // Destroy engine and context
            m_context.reset();
            m_engine.reset();

            std::cout << "Finalized TensorRT" << std::endl;
            return kRetOk;
        } catch (const std::exception& e) {
            std::cerr << "TensorRT finalization error: " << e.what() << std::endl;
            return kRetErr;
        }
    }

private:
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
    std::shared_ptr<nvinfer1::IExecutionContext> m_context;
    cudaStream_t m_cuda_stream;
    std::vector<InputTensorInfo> m_input_tensor_info_list;
    std::vector<OutputTensorInfo> m_output_tensor_info_list;
    std::map<std::string, int> m_tensor_name_to_index;
    std::vector<void*> m_buffers;
};

std::unique_ptr<InferenceHelper> InferenceHelper::Create(Backend backend) {
    switch (backend) {
        case kTensorflowLite:
        case kTensorflowLiteXnnpack:
            return std::make_unique<InferenceHelperTensorflowLite>();
        case kTensorrt:
            return std::make_unique<InferenceHelperTensorrt>();
        default:
            std::cerr << "Unsupported backend: " << backend << std::endl;
            return nullptr;
    }
}