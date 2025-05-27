# 專案說明

## predict_tensorrt_video.py

此腳本用於影片路徑的道路損壞自動偵測與分類，結合 YOLO 物件偵測、Lite-UNet 影像分割與 MobileNetV3 損壞嚴重度分類。

### 主要流程：
1. 讀取影片逐幀處理。
2. 先用 Lite-UNet 進行分割，取得 ROI。
3. 對 ROI 進行 YOLO 物件偵測與追蹤。
4. 對每個追蹤到的物件，使用 MobileNetV3 進行損壞嚴重度分類。
5. 將結果（類別、嚴重度、追蹤ID）繪製於畫面並儲存。

### 主要方法：
- `predict_mobilenet(model, image, transform, device)`：對單張影像進行嚴重度分類。
- `Unet_predict_frame(model, frame, device)`：對單幀影像進行分割，回傳 mask。
- `load_model(model_path, device)`：載入 Lite-UNet 分割模型。

---

## NEW_QT_v11.py

此腳本為 PyQt5 圖形化介面，整合 TensorRT 加速 YOLO 物件偵測、MobileNetV3 嚴重度分類，並可即時顯示辨識結果、統計數量、儲存影像與切換輸入來源。

### 主要功能：
- 支援攝影機或影片檔案輸入。
- 即時顯示辨識畫面、FPS、物件類別與嚴重度。
- 可點擊列表查看儲存影像。
- 支援一鍵開啟儲存資料夾、切換輸入來源、開始/停止辨識。

### 主要類別與方法：
- `DetectionApp(QMainWindow)`：主視窗類別，負責 UI 佈局與事件處理。
  - `start_detection()`：開始辨識與計時。
  - `stop_detection()`：暫停辨識。
  - `switch_input_source()`：切換攝影機/影片來源。
  - `update_frame()`：每幀進行推論、繪圖、儲存與 UI 更新。
  - `display_image(item)`：點擊列表顯示對應影像。
  - `open_directory()`：開啟儲存影像資料夾。

---

如需詳細參數與自訂功能，請參考原始碼註解。
