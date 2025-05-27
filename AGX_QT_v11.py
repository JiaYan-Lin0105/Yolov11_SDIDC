import cv2
from ultralytics import YOLO
import os
import cv2
from mobilenetv3_model import mobilenetv3_large_SimAM
import sys
import os
import tensorrt as trt
import torch
from torchvision import transforms
import subprocess  # 添加這行以匯入 subprocess
import numpy as np
from collections import OrderedDict, namedtuple
import sqlite3
import time
from PIL import Image
# import Jetson.GPIO as GPIO
import time as time
# import serial
from datetime import datetime
# from bytetrack import ByteTrack
import ultralytics
from ultralytics.trackers.byte_tracker import BYTETracker
import argparse
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout, \
    QTextEdit, QListWidget, QListWidgetItem, QLabel, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt



def predict_mobilenet(model, image, transform, device):
    image = Image.fromarray(image).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # 增加 batch 維度

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # 計算每個類別的概率
        confidence, predicted = probabilities.max(1)  # 最大概率值及其索引

    return predicted.item(), confidence.item()


class_dict = {
    0: "longitudinal cracks",
    1: "transverse cracks",
    2: "alligator cracks",
    3: "potholes",
}

cls_map = {
    0: 'transverse cracks',
    1: 'longitudinal cracks',
    2: 'alligator cracks',
    3: 'potholes',
    4: 'manhole',
    5: 'speed bump',
    6: 'expansion joint'
}
color_map = {
    	0: (0, 255, 0),	# 綠色
    	1: (0, 255, 255),  # 黃色
    	2: (0, 0, 255) 	# 紅色
	}
mobilenet_class_dict = {
    	0: "A",
    	1: "B",
    	2: "C",
	}

# 類別對應顏色 (BGR 格式)
cls_colors = {
    0: (255, 0, 0),   # 藍色 transverse cracks
    1: (0, 255, 0),   # 綠色 longitudinal cracks
    2: (0, 0, 255),   # 紅色 alligator cracks
    3: (255, 255, 0), # 青色 potholes
    4: (255, 0, 255), # 粉色 manhole
    5: (0, 255, 255), # 黃色 speed bump
    6: (128, 0, 128)  # 紫色 expansion joint
}
class DetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("辨識系統")
        self.setGeometry(100, 100, 1000, 500)
        self.count_track_id = []  # 用於追蹤唯一的 ID
        self.previous_value = None

        # 建立 QLabel 來顯示攝像頭畫面
        self.video_label = QLabel(self)
        self.video_label.resize(420, 240)

        # 建立開始與停止按鈕
        self.start_button = QPushButton("開始辨識", self)
        self.start_button.clicked.connect(self.start_detection)

        self.stop_button = QPushButton("停止辨識", self)
        self.stop_button.clicked.connect(self.stop_detection)

        # 建立結束程式按鈕
        self.exit_button = QPushButton("結束程式", self)
        self.exit_button.clicked.connect(self.close_program)

        self.open_directory_button = QPushButton("打開儲存路徑", self)
        self.open_directory_button.clicked.connect(self.open_directory)

        self.switch_input_button = QPushButton("切換輸入來源", self)
        self.switch_input_button.clicked.connect(self.switch_input_source)

        button_size = 80  # 假設方形邊長為 100 像素
        button_width = 80
        self.start_button.setFixedSize(button_width, button_size)
        self.stop_button.setFixedSize(button_width, button_size)
        self.exit_button.setFixedSize(button_width, button_size)
        self.open_directory_button.setFixedSize(button_width, button_size)
        self.switch_input_button.setFixedSize(button_width, button_size)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.exit_button)
        button_layout.addWidget(self.open_directory_button)
        button_layout.addWidget(self.switch_input_button)

        self.info_list = QListWidget(self)
        self.info_list.setFixedSize(150, 400)  # 設定大小
        self.info_list.itemClicked.connect(self.display_image)

        self.counter_label = QLabel(self)
        self.counter_label.setFixedSize(200, 50)

        self.update_counter()

        font = QFont()
        font.setPointSize(14)
        self.counter_label.setFont(font)

        self.image_display = QLabel(self)
        self.image_display.setFixedSize(300, 400)
        self.image_display.setAlignment(Qt.AlignCenter)

        # 建立佈局
        main_layout = QHBoxLayout()
        video_and_buttons_layout = QVBoxLayout()
        video_and_buttons_layout.addWidget(self.video_label)
        video_and_buttons_layout.addLayout(button_layout)

        info_and_image_layout = QVBoxLayout()
        info_and_image_layout.addWidget(self.info_list)
        info_and_image_layout.addWidget(self.counter_label)

        main_layout.addLayout(video_and_buttons_layout)
        main_layout.addWidget(self.info_list)
        main_layout.addWidget(self.image_display)

        # 設定中央 widget
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # 使用 QTimer 每 30 毫秒更新一次影像
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = cv2.VideoCapture(0)  # 攝像頭
        if not self.cap.isOpened():
            print("無法打開攝像頭")
            return

        # self.trt_engine = TRT_engine("./new_best_fp16.engine")  # 確保 TRT_engine 被正確初始化
        self.detection_active = False
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # self.output_video = cv2.VideoWriter('output.mp4', fourcc, 30.0, (500, 300))

        # 用於 FPS 計算的變數
        self.fps_text = "FPS: 0"

    def open_directory(self):
        # 打開影像的儲存路徑
        output_dir = r"D:\yolov11_custom\yolov11-main\ultralytics-main\RDD_image"
        if os.path.exists(output_dir):
            if sys.platform == "win32":
                os.startfile(output_dir)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", output_dir])
            else:
                subprocess.Popen(["xdg-open", output_dir])
        else:
            print(f"路徑 {output_dir} 不存在")

    def update_counter(self):
        # 更新統計辨識數量
        total_items = self.info_list.count()
        self.counter_label.setText(f"辨識數量: {total_items}")

    def start_detection(self):
        if not self.detection_active:
            print("開始辨識")
            self.detection_active = True
            self.timer.start(10)

    def stop_detection(self):
        if self.detection_active:
            print("停止辨識")
            self.detection_active = False
            self.timer.stop()

    def switch_input_source(self):
        # 切換輸入來源（攝像頭或影片文件）
        if self.detection_active:
            print("請先停止辨識再切換輸入來源")
            return

        file_dialog = QFileDialog()
        video_file, _ = file_dialog.getOpenFileName(self, "選擇影片文件", "",
                                                    "影片文件 (*.mp4 *.avi *.mov);;所有文件 (*)")
        if video_file:
            # 如果選擇了影片文件
            self.video_source = video_file
        else:
            # 如果沒有選擇影片，切換回攝像頭
            self.video_source = 0

        # 更新視頻捕獲
        self.cap.release()  # 釋放原來的視頻資源
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            print(f"無法打開視頻來源: {self.video_source}")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret and self.detection_active:
            # 開始計時
            start_time = time.time()

            # 預測與辨識
            frame = cv2.resize(frame, (500, 380))
            y_offset = 60  # 設定文字起始位置

            results = trt_engine.track(
                source=frame,
                conf=0.2,  # 置信度閾值
                tracker="bytetrack.yaml",
                persist=True  # 保持追蹤 ID
                )
            for cls_id, cls_name in cls_map.items():
                color = cls_colors.get(cls_id, (255, 255, 255))  # 預設白色
                cv2.putText(frame, f"{cls_name}", (10, y_offset),cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 25  # 調整垂直間距
            for result in results:
                boxes = result.boxes  # 取得偵測框
                for i in range(len(boxes)):
                    box_values = boxes.data[i].tolist()
                    if len(box_values) == 7:  # 確保有 track_id
                        x1, y1, x2, y2, track_id, conf, cls_id = box_values
                    else:
                        continue  # 忽略不合規輸出

                    cropped_img = frame[int(y1):int(y2), int(x1):int(x2)]
                    mobilenet_predicted_class, mobilenet_confidence = predict_mobilenet(mobilenet_model, cropped_img,
                                                                                        transform, device)
                    mobilenet_class_name = mobilenet_class_dict[mobilenet_predicted_class]
                    # 類別名稱
                    cls_name = cls_map.get(int(cls_id), "unknown")
                    # 取得顏色
                    color = cls_colors.get(int(cls_id), (255, 255, 255))  # 預設白色

                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    # 繪製邊界框（使用該類別的顏色）
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'ID {track_id} -{mobilenet_class_name}- {cls_name}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # 結束計時，計算從預測到辨識完成的時間
                    elapsed_time = time.time() - start_time

                    if elapsed_time > 0:
                        fps = 1.0 / elapsed_time
                        self.fps_text = f"FPS: {fps:.2f}"

                    # 將 FPS 顯示在左上角
                    cv2.putText(frame, self.fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    # self.output_video.write(frame)

                    # 追蹤 ID 的處理
                    track_id_trigger = False  # 默認設置為 False
                    if self.previous_value is None or track_id != self.previous_value:
                        if track_id not in self.count_track_id:
                            self.count_track_id.append(track_id)
                            track_id_trigger = True
                            print(track_id_trigger)
                    self.previous_value = track_id

                    if track_id_trigger and results:
                        class_id = cls_id
                        if class_id in class_dict:
                            cls_names = class_dict[class_id]
                            currentDateAndTime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            output_dir = f"./RDD_image/{cls_names}"
                            image_path = f"{output_dir}/{cls_names}_{currentDateAndTime}.png"
                            print(cls_names)
                            # 確保目錄存在
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)

                            # 儲存影像
                            cv2.imwrite(image_path, frame)

                            # 檢查影像是否成功儲存
                            if os.path.exists(image_path):
                                print(f"影像已儲存: {image_path}")
                            else:
                                print(f"影像未成功儲存: {image_path}")

                            # 設定 ListWidgetItem
                            list_item_text = f"影像名稱: {os.path.basename(image_path)}\n類別名稱: {cls_names}\n時間: {currentDateAndTime}\n---------------"
                            list_item = QListWidgetItem(list_item_text)
                            list_item.setData(Qt.UserRole, image_path)  # 儲存影像路徑

                            # 加入 QListWidget
                            self.info_list.addItem(list_item)
                            print(f"加入 list_item: {list_item_text}")
                            # 滾動到最新項目
                            self.info_list.scrollToBottom()
                            self.info_list.update()

                            # 更新計數
                            self.update_counter()
                            print(f"當前辨識數量: {self.info_list.count()}")
            # 將影像轉換為 QImage 格式，並顯示在 QLabel 上
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 更新 QLabel 的顯示畫面
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def display_image(self, item):
        # 根據點擊的項目顯示相應的圖片
        image_path = item.data(Qt.UserRole)
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # 使用 scaled() 方法等比例縮小影像以適應 QLabel 大小
                scaled_pixmap = pixmap.scaled(self.image_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_display.setPixmap(scaled_pixmap)
                print(f"顯示影像: {image_path}")
            else:
                print(f"加載影像失敗: {image_path}")
        else:
            print(f"影像文件不存在: {image_path}")

    def close_program(self):
        print("結束程式")
        QApplication.instance().quit()

    def closeEvent(self, event):
        # 釋放攝像頭資源
        self.cap.release()



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 圖像的轉換
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    num_classes = 3  # 修改為你的分類數
    mobilenet_model = mobilenetv3_large_SimAM(num_classes=num_classes, width_mult=0.5).to(device)

    # 設定要加載的模型檔案
    model_path = './best_mobilenet_epoch_255.pth'
    mobilenet_model.load_state_dict(torch.load(model_path))
    mobilenet_model.eval()
    trt_engine = YOLO("best.engine")
    app = QApplication(sys.argv)
    window = DetectionApp()
    window.show()
    sys.exit(app.exec_())

