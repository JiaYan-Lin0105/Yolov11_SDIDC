import torch
from ultralytics import YOLO

# 1. 讀取已訓練好的模型
#    以下示範先用 ultralytics 方式載入，再取出 model.model 將其轉成 torch.nn.Module
yolo_model = YOLO("yolov11/Road_seven_yolov11n_BS16_e200_mosaic05_IDC_CSD_C2f/weights/best.pt")
model = yolo_model.model

model.eval()  # 設定為推論模式

# 2. 建立一個假輸入 (dummy input)，大小與你訓練時相同
dummy_input = torch.randn(1, 3, 640, 640)  # [batch, channel, height, width]

# 3. 呼叫 torch.onnx.export 進行匯出
torch.onnx.export(
    model,                 # 要轉換的模型
    dummy_input,           # 假輸入
    "best.onnx",          # 輸出 onnx 檔案名稱
    input_names=["images"],
    output_names=["output"],
    export_params=True,
    opset_version=12,      # 建議使用較新的 opset，如 11 或 12
    do_constant_folding=True,
    dynamic_axes={
        "images": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)
