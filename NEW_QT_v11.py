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
    QTextEdit, QListWidget, QListWidgetItem, QLabel, QFileDialog, QStackedWidget
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
    0: (0, 0, 255),   # 藍色 transverse cracks
    1: (0, 255, 0),   # 綠色 longitudinal cracks
    2: (255, 0, 0),   # 紅色 alligator cracks
    3: (0, 165, 255), # 青色 potholes
    4: (128, 0, 128), # 粉色 manhole
    5: (255, 0, 255), # 黃色 speed bump
    6: (42, 42, 165)  # 紫色 expansion joint
}


class DetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RoadScan Pro")
        self.setGeometry(100, 100, 1000, 500)  # 調整主視窗大小
        self.setStyleSheet("background-color: white;")
        self.class_counts = {key: 0 for key in cls_map.values()}
        self.count_track_id = []
        self.previous_value = None
        self.detection_active = False

        # 建立 QLabel 來顯示攝像頭畫面
        self.video_label = QLabel(self)
        self.video_label.setGeometry(220, 50, 650, 350)  # 調整視訊顯示區域
        self.video_label.setStyleSheet("""
            border: 3px solid black;
        """)

        # Logo 標籤
        self.Logo_label = QLabel(self)
        self.Logo_label.setGeometry(420, 5, 260, 35)  # 調整 Logo 位置
        self.Logo_label.setText("""
                        <b><font color='black'>RoadScan Pro</font></b>
                """)
        font = QFont()
        font.setPointSize(20)  # 調整字體大小
        self.Logo_label.setFont(font)
        self.Logo_label.setStyleSheet("""font-family: "Verdana";""")

        # 類別標籤
        self.class_label = QLabel(self)
        self.class_label.setGeometry(10, 5, 170, 220)  # 調整類別標籤區域
        self.class_label.setText(f"""
            <table border='1' cellspacing='0' cellpadding='2' style='border-collapse: collapse; border: 2px solid black; text-align: left;'>
                <tr><td><b><font color='red'>Transverse cracks</font></b></td></tr>
                <tr><td><b><font color='green'>Longitudinal cracks</font></b></td></tr>
                <tr><td><b><font color='blue'>Alligator cracks</font></b></td></tr>
                <tr><td><b><font color='orange'>Potholes</font></b></td></tr>
                <tr><td><b><font color='purple'>Manhole</font></b></td></tr>
                <tr><td><b><font color='magenta'>Speed bump</font></b></td></tr>
                <tr><td><b><font color='brown'>Expansion joint</font></b></td></tr>
                <tr><td><b><font color='black'>辨識數量</font></b></td></tr>
            </table>
        """)
        font = QFont()
        font.setPointSize(12)  # 調整字體大小
        self.class_label.setFont(font)

        # 計數標籤
        self.count_label = QLabel(self)
        self.count_label.setGeometry(160, 5, 40, 220)  # 調整計數標籤位置和大小
        self.count_label.setText(f"""
                    <table border='1' cellspacing='0' cellpadding='2' style='border-collapse: collapse; border: 100px solid black; text-align: center;'>
                        <tr><td style="width: 40px;"><b><font color='black'>{0}</font></b></td></tr>
                        <tr><td style="width: 40px;"><b><font color='black'>{0}</font></b></td></tr>
                        <tr><td style="width: 40px;"><b><font color='black'>{0}</font></b></td></tr>
                        <tr><td style="width: 40px;"><b><font color='black'>{0}</font></b></td></tr>
                        <tr><td style="width: 40px;"><b><font color='black'>{0}</font></b></td></tr>
                        <tr><td style="width: 40px;"><b><font color='black'>{0}</font></b></td></tr>
                        <tr><td style="width: 40px;"><b><font color='black'>{0}</font></b></td></tr>
                        <tr><td style="width: 40px;"><b><font color='black'>{0}</font></b></td></tr>
                    </table>
                """)
        font = QFont()
        font.setPointSize(12)  # 調整字體大小
        self.count_label.setFont(font)

        # 按鈕區域
        button_y = 410  # 調整按鈕的 Y 座標
        button_height = 80  # 調整按鈕高度
        button_width = 110  # 調整按鈕寬度
        button_spacing = 25  # 調整按鈕間距

        # 開始辨識按鈕
        self.start_button = QPushButton("開始辨識", self)
        self.start_button.setGeometry(220, button_y, button_width, button_height)
        self.start_button.clicked.connect(self.start_detection)
        self.apply_button_style(self.start_button)

        # 停止辨識按鈕
        self.stop_button = QPushButton("停止辨識", self)
        self.stop_button.setGeometry(220 + button_width + button_spacing, button_y, button_width, button_height)
        self.stop_button.clicked.connect(self.stop_detection)
        self.apply_button_style(self.stop_button)

        # 結束程式按鈕
        self.exit_button = QPushButton("結束程式", self)
        self.exit_button.setGeometry(220 + (button_width + button_spacing) * 2, button_y, button_width, button_height)
        self.exit_button.clicked.connect(self.close_program)
        self.apply_button_style(self.exit_button)

        # 打開儲存路徑按鈕
        self.open_directory_button = QPushButton("打開儲存路徑", self)
        self.open_directory_button.setGeometry(220 + (button_width + button_spacing) * 3, button_y, button_width, button_height)
        self.open_directory_button.clicked.connect(self.open_directory)
        self.apply_button_style(self.open_directory_button)

        # 切換輸入來源按鈕
        self.switch_input_button = QPushButton("切換輸入來源", self)
        self.switch_input_button.setGeometry(220 + (button_width + button_spacing) * 4, button_y, button_width, button_height)
        self.switch_input_button.clicked.connect(self.switch_input_source)
        self.apply_button_style(self.switch_input_button)

        # 資訊列表
        self.info_list = QListWidget(self)
        self.info_list.setGeometry(10, 230, 200, 260)  # 調整資訊列表大小
        self.info_list.itemClicked.connect(self.display_image)

        # 圖片顯示標籤
        self.image_display = QLabel(self)
        self.image_display.setGeometry(270, 150, 400, 220)  # 調整圖片顯示區域
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setAttribute(Qt.WA_TranslucentBackground)

        # 設定計時器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("無法打開攝像頭")
            return

    def apply_button_style(self, button):
        button.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: black;
                font-size: 14px;
                font-family: "Microsoft JhengHei";
                font-weight: bold;
                border-radius: 20px;
                border: 2px solid black;
            }
            QPushButton:hover {
                background-color: gray;
                color: white;
            }
            QPushButton:pressed {
                background-color: lightgray;
                color: white;
            }
        """)

    def new_button_clicked(self):
        """新按鈕的點擊事件"""
        print("新按鈕被點擊了!")
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
        self.count_label.setText(f"""
                <table border='1' cellspacing='0' cellpadding='2' 
                style='border-collapse: collapse; border: 100px solid black; text-align: left;'>
                {''.join(f'<tr><td style="width: 50px;"><b><font color="black">{count}</font></b></td></tr>'
                         for cls_name, count in self.class_counts.items())}
                <tr><td style="width: 30px;"><b><font color='black'>{total_items}</font></b></td></tr>
                </table>
            """)
        # self.count_label.setText(f"""
        #                     <table border='1' cellspacing='0' cellpadding='2' style='border-collapse: collapse; border: 100px solid black; text-align: left;'>
        #                         <tr><td style="width: 30px;"><b><font color='black'>{0}</font></b></td></tr>
        #                         <tr><td style="width: 30px;"><b><font color='black'>{0}</font></b></td></tr>
        #                         <tr><td style="width: 30px;"><b><font color='black'>{0}</font></b></td></tr>
        #                         <tr><td style="width: 30px;"><b><font color='black'>{0}</font></b></td></tr>
        #                         <tr><td style="width: 30px;"><b><font color='black'>{0}</font></b></td></tr>
        #                         <tr><td style="width: 30px;"><b><font color='black'>{0}</font></b></td></tr>
        #                         <tr><td style="width: 30px;"><b><font color='black'>{0}</font></b></td></tr>
        #                         <tr><td style="width: 30px;"><b><font color='black'>{total_items}</font></b></td></tr>
        #                     </table>
        #                 """)

    def start_detection(self):
        if not self.detection_active:
            print("開始辨識")
            self.detection_active = True
            self.timer.start(10)

    def stop_detection(self):
        if self.detection_active:
            print("暫停辨識")
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
            frame = cv2.resize(frame, (650, 350))

            results = trt_engine.track(
                source=frame,
                conf=0.2,  # 置信度閾值
                tracker="bytetrack.yaml",
                persist=True  # 保持追蹤 ID
                )

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

                            # list_item_text = f"影像名稱: {os.path.basename(image_path)}\n類別名稱: {cls_names}\n時間: {currentDateAndTime}\n嚴重程度: {mobilenet_class_name}\n-------------------"
                            list_item_text = f"類別名稱: {cls_names}\n時間: {currentDateAndTime}\n嚴重程度: {mobilenet_class_name}\n-------------------"
                            list_item = QListWidgetItem(list_item_text)
                            list_item.setData(Qt.UserRole, image_path)  # 儲存影像路徑

                            # 加入 QListWidget
                            self.info_list.addItem(list_item)
                            print(f"加入 list_item: {list_item_text}")
                            # 滾動到最新項目
                            self.info_list.scrollToBottom()
                            self.info_list.update()
                            if cls_name in self.class_counts:
                                self.class_counts[cls_name] += 1  # 類別數量 +1
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
        # 取得點擊的影像路徑
        image_path = item.data(Qt.UserRole)

        # 如果當前顯示的影像與點擊的影像相同，就關閉顯示
        if hasattr(self, "current_image_path") and self.current_image_path == image_path:
            self.image_display.clear()  # 清除 QLabel 顯示的影像
            self.current_image_path = None  # 重置記錄的影像路徑
            print("關閉影像顯示")
            return

        # 檢查影像是否存在
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # 使用 scaled() 方法等比例縮小影像以適應 QLabel 大小
                scaled_pixmap = pixmap.scaled(self.image_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_display.setPixmap(scaled_pixmap)
                self.current_image_path = image_path  # 記錄當前顯示的影像
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

