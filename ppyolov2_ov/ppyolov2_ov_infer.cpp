// OpenVINO C++ Sample code for PPYOLOv2
// Paddledetection 模型输入节点信息：https://gitee.com/paddlepaddle/PaddleDetection/blob/release/2.3/deploy/EXPORT_MODEL.md
// ONLY support batchsize = 1

#include<string>
#include<vector>
#include<iostream>
#include<map>
#include<time.h>

#include<inference_engine.hpp>

#include "opencv2/opencv.hpp"

using namespace InferenceEngine;
using namespace std;

//将OpenCV Mat对象中的图像数据传给PPYOLOv2模型的"image"节点
template <typename T>
void fillBlobImage(Blob::Ptr& inputBlob, const cv::Mat& frame);
//将图像信息传入PPYOLOv2模型的"im_shape"和"scale_factor"节点的数据
template <typename T>
void fillBlobImInfo(Blob::Ptr& inputBlob, std::pair<float, float> image_info);

//请手动配置推理计算设备，IR文件路径，图片路径，阈值和标签
string DEVICE = "CPU";
string IR_FileXML =  "D:/pd2060/ov_model/ppyolov2.xml";
string IMAGE_FILE = "D:/pd2060/ov_model/road554.png";
float CONF_THRESHOLD = 0.7; //取值0~1
vector<string> LABELS = { "speedlimit","crosswalk","trafficlight","stop" }; //标签输入

int main()
{
	// --------------------------- 1. 创建Core对象 --------------------------------------
    cout << "1.Create Core Object." << endl;
    Core ie;  // 创建Core对象

    // --------------------------- 2. 载入模型到AI推理计算设备----------------------------
    cout << "2.Load model into device..." << endl;
    ExecutableNetwork executable_network = ie.LoadNetwork(IR_FileXML, DEVICE); 

    // --------------------------- 3. 创建Infer Request------------------------------------
    cout << "3.Create infer request..." << endl;
    auto infer_request = executable_network.CreateInferRequest();

    // --------------------------- 4. 准备输入数据 ----------------------------------------
    // 由OpenCV完成数据预处理：RB交换、Resize和归一化
    // PPYOLOv2输入节点信息：https://gitee.com/paddlepaddle/PaddleDetection/blob/release/2.3/deploy/EXPORT_MODEL.md
    cout << "4.Prepare model's Input..." << endl;
    Blob::Ptr image_blob = infer_request.GetBlob("image");
    auto input_H = image_blob->getTensorDesc().getDims()[2]; //获得"image"节点的Height
    auto input_W = image_blob->getTensorDesc().getDims()[3]; //获得"image"节点的Width

    cv::Mat img = cv::imread(IMAGE_FILE,cv::IMREAD_COLOR); //从图像文件读入数据

    // 交换RB通道
    cv::Mat blob;
    cv::cvtColor(img, blob, cv::COLOR_BGR2RGB); //Convert BGR to RGB
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
    
    cout << "Write image data into image node..." << endl;
    fillBlobImage<float>(image_blob, blob);
 
    cout << "Write scale factor into scale_factor node..." << endl;
    const float scale_h = float(input_H) / float(img.rows);
    const float scale_w = float(input_W) / float(img.cols);
    const pair<float, float> scale_factor( scale_h, scale_w );
    auto scale_factor_blob = infer_request.GetBlob("scale_factor");
    fillBlobImInfo<float>(scale_factor_blob, scale_factor);  //scale_factor node's precision is float32

    cout << "Write image shape into im_shape node..." << endl;
    const pair<float, float> im_shape ( 640.0, 640.0 );
    auto im_shape_blob = infer_request.GetBlob("im_shape");
    fillBlobImInfo<float>(im_shape_blob, im_shape);  //im_shape node's precision is float32

    // --------------------------- 5. 执行推理计算 ----------------------------------------
    cout << "5.Start Inference..." << endl;
    clock_t begin, end;
    begin = std::clock();
    infer_request.Infer();
    end = std::clock();
    cout << "Infer Time:" << (float)(end - begin) << "ms" << endl;
    
    // --------------------------- 6. 处理推理结果 ----------------------------------------
    cout << "6. Process the Inference Results..." << endl;
    // PaddleDetection导出模型统一输出为：
    // 形状为[N, 6], 其中N为预测框的个数，6为[class_id, score, x1, y1, x2, y2]
    string bbox_name = "translated_layer/scale_0.tmp_0";
    string bbox_num_name = "translated_layer/scale_1.tmp_0";

    const float* detections = infer_request.GetBlob(bbox_name)->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
    const int* bbox_nums = infer_request.GetBlob(bbox_num_name)->buffer().as<PrecisionTrait<Precision::I32>::value_type*>();
    auto bbox_num = bbox_nums[0];
    cout << "bbox_num: " << bbox_num << endl;

    for (int i = 0; i < bbox_num; i++) {
        auto class_id = static_cast<int> (detections[i * 6 + 0]);
        float score = detections[i * 6 + 1];
        if (score > CONF_THRESHOLD){
            float x1 = detections[i * 6 + 2];
            float y1 = detections[i * 6 + 3];
            float x2 = detections[i * 6 + 4];
            float y2 = detections[i * 6 + 5];
            cout << "class_id:" << class_id <<"; score:" << score << "; x1:" << x1<<"; y1:"<<y1<<"; x2:"<<x2<<"; y2:"<<y2<<endl;
            std::ostringstream conf;
            conf << ":" << std::fixed << std::setprecision(3) << score;
            cv::putText(img, (LABELS[class_id] + conf.str()),
                cv::Point2f(x1, y1 - 5), cv::FONT_HERSHEY_COMPLEX, 0.5,
                cv::Scalar(255, 0, 0), 1);
            cv::rectangle(img, cv::Point2f(x1, y1), cv::Point2f(x2, y2), cv::Scalar(255, 0, 0));
        }
    }

    std::cout << "All is completed!" << std::endl;
    cv::imshow("Detection results", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
    
	return 0;
}

//将OpenCV Mat对象中的图像数据传给PPYOLOv2模型的"image"节点
template <typename T>
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

    T* inputBlobData = minputHolder.as<T*>();
    //hwc -> chw, Write image data into node's blob
    for (size_t c = 0; c < channels; c++) {
        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                inputBlobData[c * width * height + h * width + w] = frame.at<cv::Vec<T, 3>>(h, w)[c];
            }
        }
    }
}

//填写模型输入图像信息节点的数据
template <typename T>
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

    auto inputBlobData = minputHolder.as<T*>();
    // Write image info into node's blob
    inputBlobData[0] = static_cast<T>(image_info.first);
    inputBlobData[1] = static_cast<T>(image_info.second);
}
