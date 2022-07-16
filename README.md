# yolov6n 的caffe、onnx和tensorRT部署版本

将pytorch版本的 yolov6n 转成caffe、onnx、tensorRT，用python语言对后处理进行了C++形式的重写，便于移植不同平台。

# 文件夹结构说明
yolov6n_caffe：去除维度变换层的prototxt、caffeModel、测试图像、测试结果、测试demo脚本

yolov6n_onnx：onnx模型、测试图像、测试结果、测试demo脚本

yolov6n_tensorRT：TensorRT版本模型、测试图像、测试结果、测试demo脚本、onnx模型、onnx2tensorRT脚本(tensorRT-7.2.3.4)

#  测试结果
![image]()
