# coding:utf-8
from ultralytics import YOLO
data = 'D:/yolov11-main/ultralytics-main/dataset/Data/seven_class.yaml'
# 模型配置文件
model_yaml_path = "ultralytics/cfg/models/11/MobileNetV4.yaml"
# 数据集配置文件

# 预训练模型
pre_model_name = 'yolo11s.pt'

if __name__ == '__main__':
    # 加载预训练模型
    model = YOLO(model_yaml_path, task='detect').load(pre_model_name)
    # 训练模型
    results = model.train(
        data=data,
        epochs=100,
        batch=16,
        name='train_emoKDEF_MobileNetV4_s',
    )