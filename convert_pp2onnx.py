import paddle2onnx
import paddle
from openvino_ppdet import nms_mapper
# 通过上面的`nms_mapper`的import来启用插件，替换了paddle2onnx原始的nms_mapper

model_prefix = "D:\pd2060\PaddleDetection\inference_model\ppyolov2_r50vd_dcn_roadsign\model"
model = paddle.jit.load(model_prefix)
input_shape_dict = {
    "image": [1, 3, 640, 640],
    "scale_factor": [1, 2],
    "im_shape": [1, 2]
    }
paddle.enable_static()
onnx_model = paddle2onnx.run_convert(model, input_shape_dict=input_shape_dict, opset_version=11)

with open("./ppyolov2.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())