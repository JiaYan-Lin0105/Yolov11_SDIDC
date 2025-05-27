from ultralytics import YOLO
import os

# 模型權重路徑
model_weights_path = r'D:\yolov11_custom\yolov11-main\ultralytics-main\yolov11\Road_seven_yolov11n_BS16_e200_mosaic05_IDC_v10\weights/best.pt'
# model_weights_path = r'D:\yolov11-main\ultralytics-main\yolov11\Road_seven_yolov11_e200\weights/best.pt'

# 測試資料或影像資料夾
input_data_path = r'D:\road_damage\Produce_old.avi'

# 輸出資料夾
output_data_path = './output/test_data'

if __name__ == '__main__':
    # 確保輸出資料夾存在
    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)

    # 加載訓練好的模型
    model = YOLO(model_weights_path)

    # 執行推理
    results = model.predict(
        source=input_data_path,  # 測試資料來源
        save=True,               # 儲存結果
        save_txt=True,           # 儲存預測文字檔
        save_conf=False,          # 儲存置信度
        conf=0.1,
        project=f"{output_data_path}/Road_seven_yolov11n_BS16_e200_mosaic05_IDC_v10" # 結果輸出目錄

    )

    print("推理完成，結果已儲存於:", output_data_path)