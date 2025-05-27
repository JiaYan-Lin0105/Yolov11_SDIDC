from ultralytics import YOLO
from ultralytics import settings
model_="IDC_v10"#IDC_v10.yaml是改良後,yolo11.yaml是原始的

# data_yaml_path = 'D:/yolov11-main/ultralytics-main/dataset/Data/seven_class.yaml'
data_yaml_path = r'D:\yolov11_custom\yolov11-main\ultralytics-main\datasets\Data\RDD2022.yaml'

# 预训练模型
# pre_model_name = f'./{model_}.pt'

# WandB 项目名称
wandb_project_name = "yolov11"
settings.update({"wandb": True})
name=(f'CoCo_yolov11n_BS16_e200_mosaic05_IDC_CSD_C2f')#輸出路徑
if __name__ == '__main__':
    # 初始化 WandB
    # wandb.init(project=wandb_project_name,entity="a201162011665",name=name)
    # 加载预训练模型
    model = YOLO(f"./ultralytics/cfg/models/11/{model_}.yaml")

    results = model.train(
        data=data_yaml_path,
        epochs=200,
        batch=16,
        imgsz=640,
        lr0=0.001,
        lrf=0.1,
        name=name,
        project=wandb_project_name,  # WandB 项目名称
        mosaic=0.5,
        weight_decay=0.0005
    )
    # wandb.finish()