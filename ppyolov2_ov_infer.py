# OpenVINO C++ Sample code for PPYOLOv2
# Paddledetection 模型输入节点信息：https://gitee.com/paddlepaddle/PaddleDetection/blob/release/2.3/deploy/EXPORT_MODEL.md
# ONLY support batchsize = 1
# https://docs.openvino.ai/latest/openvino_docs_IE_DG_Integrate_with_customer_application_new_API.html?sw_type=switcher-python

from openvino.inference_engine import IECore
import cv2
import numpy as np

# 请手动配置推理计算设备，IR文件路径，图片路径，阈值和标签
DEVICE = "CPU"
IR_FileXML = "ov_model/ppyolov2.xml"
IMAGE_FILE = "road554.png"
CONF_THRESHOLD = 0.7  #取值0~1
LABELS = [ "speedlimit","crosswalk","trafficlight","stop" ] #标签输入

# --------------------------- 1. 创建Core对象 --------------------------------------
print("1.Create Core Object.")
ie = IECore()

# --------------------------- 2. 载入模型到AI推理计算设备----------------------------
print("2.Load model into device...")
exec_net = ie.load_network(network=IR_FileXML, device_name=DEVICE)

# --------------------------- 3. 准备输入数据 --------------------------------------
# 由OpenCV完成数据预处理：RB交换、Resize，归一化和HWC->NCHW
print("3.Prepare the input data for the model...")
frame = cv2.imread(IMAGE_FILE)
if frame is None:
    raise Exception("Can not read image file: {} by cv2.imread".format(IMAGE_FILE))
# 交换RB通道
im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# 获得模型'image'节点要求的输入尺寸
_,_,input_h,input_w = exec_net.input_info["image"].tensor_desc.dims
# 计算模型要求的scale factor 并放缩图像
scale_h = input_h / float(im.shape[0])
scale_w = input_w / float(im.shape[1])
im = cv2.resize(im, None, None, fx=scale_w, fy=scale_h, interpolation=cv2.INTER_LINEAR)
# 图像数据归一化
im = im / 255.0
im -= [0.485, 0.456, 0.406]
im /= [0.229, 0.224, 0.225]
# 图像数据格式由HWC转为NCHW
im = im.transpose((2, 0, 1))
im = np.expand_dims(im, axis=0)
# 准备输入数据字典
inputs = dict()
inputs["image"] = im
inputs["im_shape"] = np.array(im.shape[2:]).astype("float32")
inputs["scale_factor"] = np.array([scale_h, scale_w]).astype("float32")

# --------------------------- 4. 执行推理并获得结果 ------------------------------------
print("4.Start Inference......")
import time
start = time.time()
outputs = exec_net.infer(inputs=inputs)
end = time.time()
print(f"infer time: {end - start} ms")

# --------------------------- 5. 处理推理计算结果 --------------------------------------
print("5.Postprocess the inference result......")
# PaddleDetection导出模型统一输出为：
# 形状为[N, 6], 其中N为预测框的个数，6为[class_id, score, x1, y1, x2, y2]
# 找出class_id >=0 且 score >= CONF_THRESHOLD的结果
boxes = None
for k, v in outputs.items():
    if len(v.shape) == 2 and v.shape[1] == 6:
        boxes = v
        break
# 过滤类别ID小于0的结果
filtered_boxes = boxes[boxes[:, 0] > -1e-06]
# 过滤低置信度结果
filtered_boxes = filtered_boxes[filtered_boxes[:, 1] >= CONF_THRESHOLD]

# 显示检测框,检测框格式 [class_id, score, x1, y1, x2, y2]
for box in filtered_boxes:
    cv2.rectangle(frame, (int(box[2]),int(box[3])),(int(box[4]),int(box[5])),(255,0,0))
    label = LABELS[int(box[0])]
    conf  = "{:.4f}".format(box[1])
    cv2.putText(frame, label+conf, (int(box[2]),int(box[3])-5), cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0))

cv2.imshow("Detection results", frame)
cv2.waitKey()
cv2.destroyAllWindows()
print("All is completed!")