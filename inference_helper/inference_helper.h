#ifndef INFERENCE_HELPER_H
#define INFERENCE_HELPER_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

class InferenceHelper {
public:
    enum {
        kRetOk = 0,
        kRetErr = -1,
    };

    enum Backend {
        kTensorflowLite,
        kTensorflowLiteXnnpack,
        kTensorflowLiteGpu,
        kTensorflowLiteEdgetpu,
        kTensorflowLiteNnapi,
        kTensorrt,
    };

    enum TensorType {
        kTensorTypeFloat32,
        kTensorTypeInt8,
        kTensorTypeFloat16,
        kTensorTypeInt32,
    };

    struct InputTensorInfo {
        enum DataType {
            kDataTypeImage,
            kDataTypeBlob,
        };
        std::string name;
        TensorType tensor_type;
        bool is_nchw;
        std::vector<int32_t> tensor_dims;
        DataType data_type;
        void* data;
        struct {
            int32_t width;
            int32_t height;
            int32_t channel;
            int32_t crop_x;
            int32_t crop_y;
            int32_t crop_width;
            int32_t crop_height;
            bool is_bgr;
            bool swap_color;
        } image_info;
        float normalize_mean[3];
        float normalize_norm[3];

        InputTensorInfo(const std::string& name_, TensorType type_, bool is_nchw_)
            : name(name_), tensor_type(type_), is_nchw(is_nchw_), data_type(kDataTypeImage) {}
        int32_t GetWidth() const { return tensor_dims[3]; }
        int32_t GetHeight() const { return tensor_dims[2]; }
    };

    struct OutputTensorInfo {
        std::string name;
        TensorType tensor_type;
        std::vector<int32_t> tensor_dims;
        void* data;

        OutputTensorInfo(const std::string& name_, TensorType type_)
            : name(name_), tensor_type(type_), data(nullptr) {}

        float* GetDataAsFloat() { return static_cast<float*>(data); }
        int32_t GetElementNum() const {
            int32_t size = 1;
            for (const auto& dim : tensor_dims) {
                size *= dim;
            }
            return size;
        }
        int32_t GetWidth() const { return tensor_dims[3]; }
        int32_t GetHeight() const { return tensor_dims[2]; }
    };

    static std::unique_ptr<InferenceHelper> Create(Backend backend);
    virtual ~InferenceHelper() = default;
    virtual int32_t SetNumThreads(int32_t num_threads) = 0;
    virtual int32_t Initialize(const std::string& model_filename,
                              std::vector<InputTensorInfo>& input_tensor_info_list,
                              std::vector<OutputTensorInfo>& output_tensor_info_list) = 0;
    virtual int32_t PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list) = 0;
    virtual int32_t Process(std::vector<OutputTensorInfo>& output_tensor_info_list) = 0;
    virtual int32_t Finalize() = 0;

protected:
    InferenceHelper() = default;
};

#endif // INFERENCE_HELPER_H