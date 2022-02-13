// OpenVINO C++ dll code for LabVIEW
// Paddledetection 模型输入节点信息：https://gitee.com/paddlepaddle/PaddleDetection/blob/release/2.3/deploy/EXPORT_MODEL.md
// ONLY support batchsize = 1

#include<string>
#include<vector>
#include<iostream>
#include<map>
#include<time.h>
#include<string.h>

#include<inference_engine.hpp>

#include "NIVisionExtLib.h"
#include "opencv2/opencv.hpp"

using namespace InferenceEngine;
using namespace std;

//将OpenCV Mat对象中的图像数据传给PPYOLOv2模型的"image"节点
void fillBlobImage(Blob::Ptr& inputBlob, const cv::Mat& frame);
//将图像信息传入PPYOLOv2模型的"im_shape"和"scale_factor"节点的数据
void fillBlobImInfo(Blob::Ptr& inputBlob, std::pair<float, float> image_info);

//定义结构体，存储与Inference Engine相关的变量
typedef struct lv_infer_engine {
    Core ie;                     //ie对象
    ExecutableNetwork exec_net; 
    InferRequest infer_request;
} InferEngineStruct;

// 创建指向InferEngine的指针，并反馈给LabVIEW
EXTERN_C  NI_EXPORT void* ppyolov2_init(char* model_xml_file, char* device_name, NIErrorHandle errorHandle) {

    InferEngineStruct* p = new InferEngineStruct();
    p->exec_net = p->ie.LoadNetwork(model_xml_file, device_name);
    p->infer_request = p->exec_net.CreateInferRequest();

    return (void*)p;        
}

EXTERN_C void NI_EXPORT ppyolov2_predict(NIImageHandle sourceHandle, void* pInferEngine, char* bbox_name, char* bbox_num_name, float* detections, NIErrorHandle errorHandle) {

    NIERROR error = NI_ERR_SUCCESS;
    ReturnOnPreviousError(errorHandle);

    try {

        NIImage source(sourceHandle);

        Mat sourceMat;

        InferEngineStruct* p = (InferEngineStruct*)pInferEngine;
        //从NIImage对象中浅拷贝图像数据到Mat对象
        ThrowNIError(source.ImageToMat(sourceMat));
        auto type = source.type;

        Blob::Ptr image_blob = p->infer_request.GetBlob("image");
        auto input_H = image_blob->getTensorDesc().getDims()[2]; //获得"image"节点的Height
        auto input_W = image_blob->getTensorDesc().getDims()[3]; //获得"image"节点的Width

        // 交换RB通道
        cv::Mat blob;
        cv::cvtColor(sourceMat, blob, cv::COLOR_BGRA2RGB); //Convert RGBA to RGB
        // 放缩图片到(input_H,input_W)
        cv::resize(blob, blob, cv::Size(input_H, input_W), 0, 0, cv::INTER_LINEAR);
        // 图像数据归一化，减均值mean，除以方差std
        // PaddleDetection模型使用imagenet数据集的 Mean = [0.485, 0.456, 0.406]和 std = [0.229, 0.224, 0.225]
        vector<float> mean_values{ 0.485 * 255, 0.456 * 255, 0.406 * 255 };
        vector<float> std_values{ 0.229 * 255, 0.224 * 255, 0.225 * 255 };
        vector<cv::Mat> rgbChannels(3);
        split(blob, rgbChannels);
        for (auto i = 0; i < rgbChannels.size(); i++)
        {
            rgbChannels[i].convertTo(rgbChannels[i], CV_32FC1, 1.0 / std_values[i], (0.0 - mean_values[i]) / std_values[i]);
        }
        merge(rgbChannels, blob);

        fillBlobImage(image_blob, blob);
       
        const float scale_h = float(input_H) / float(sourceMat.rows);
        const float scale_w = float(input_W) / float(sourceMat.cols);
        const pair<float, float> scale_factor(scale_h, scale_w);
        auto scale_factor_blob = p->infer_request.GetBlob("scale_factor");
        fillBlobImInfo(scale_factor_blob, scale_factor);  //scale_factor node's precision is float32
        
        const pair<float, float> im_shape(input_H, input_W);
        auto im_shape_blob = p->infer_request.GetBlob("im_shape");
        fillBlobImInfo(im_shape_blob, im_shape);  //im_shape node's precision is float32

        // --------------------------- 5. 执行推理计算 ----------------------------------------
        p->infer_request.Infer();        
        const float* infer_results = p->infer_request.GetBlob(bbox_name)->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();        
        const int* bbox_nums = p->infer_request.GetBlob(bbox_num_name)->buffer().as<PrecisionTrait<Precision::I32>::value_type*>();

        auto bbox_num = bbox_nums[0];
        for (int i = 0; i < bbox_num; i++) {
            detections[i * 6 + 0] = infer_results[i * 6 + 0];
            detections[i * 6 + 1] = infer_results[i * 6 + 1];
            detections[i * 6 + 2] = infer_results[i * 6 + 2];
            detections[i * 6 + 3] = infer_results[i * 6 + 3];
            detections[i * 6 + 4] = infer_results[i * 6 + 4];
            detections[i * 6 + 5] = infer_results[i * 6 + 5];
        }

    }
    catch (NIERROR _err) {
        error = _err;
    }
    catch (...) {
        error = NI_ERR_OCV_USER;
    }

    ProcessNIError(error, errorHandle);
    
}

EXTERN_C void NI_EXPORT ppyolov2_delete(void* pInferEngine, NIErrorHandle errorHandle) {

    NIERROR error = NI_ERR_SUCCESS;
    ReturnOnPreviousError(errorHandle);

    InferEngineStruct* p = (InferEngineStruct*)pInferEngine;
    delete p;
}
//将OpenCV Mat对象中的图像数据传给PPYOLOv2模型的"image"节点

void fillBlobImage(Blob::Ptr& inputBlob, const cv::Mat& frame) {
    SizeVector blobSize = inputBlob->getTensorDesc().getDims();
    const size_t width = blobSize[3];
    const size_t height = blobSize[2];
    const size_t channels = blobSize[1];
    MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
    if (!minput) {
        IE_THROW() << "We expect blob to be inherited from MemoryBlob in matU8ToBlob, "
            << "but by fact we were not able to cast inputBlob to MemoryBlob";
    }
    // locked memory holder should be alive all time while access to its buffer happens
    auto minputHolder = minput->wmap();

    float* inputBlobData = minputHolder.as<float*>();
    //hwc -> chw, Write image data into node's blob
    for (size_t c = 0; c < channels; c++) {
        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                inputBlobData[c * width * height + h * width + w] = frame.at<cv::Vec<float, 3>>(h, w)[c];
            }
        }
    }
}

//填写模型输入图像信息节点的数据
void fillBlobImInfo(Blob::Ptr& inputBlob, std::pair<float, float> image_info) {
    MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
    if (!minput) {
        IE_THROW() << "We expect inputBlob to be inherited from MemoryBlob in "
            "fillBlobImInfo, "
            << "but by fact we were not able to cast inputBlob to MemoryBlob";
    }
    // locked memory holder should be alive all time while access to its buffer
    // happens
    auto minputHolder = minput->wmap();

    auto inputBlobData = minputHolder.as<float*>();
    // Write image info into node's blob
    inputBlobData[0] = static_cast<float>(image_info.first);
    inputBlobData[1] = static_cast<float>(image_info.second);
}
