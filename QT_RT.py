import sys
import os
import cv2
import torch
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QTextEdit
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
from mobilenetv3_model import mobilenetv3_large_SimAM
from torchvision import transforms
from PIL import Image

# 影片路徑
input_data_path = r'D:\road_damage\Produce_old.avi'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mobilenet_class_dict = {
    	0: "A",
    	1: "B",
    	2: "C",
	}
# 類別對應表 (物件分類)
cls_map = {
    0: 'transverse cracks',
    1: 'longitudinal cracks',
    2: 'alligator cracks',
    3: 'potholes',
    4: 'manhole',
    5: 'speed bump',
    6: 'expansion joint'
}

# 類別對應顏色 (BGR格式)
cls_colors = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (255, 0, 255),
    5: (0, 255, 255),
    6: (128, 0, 128)
}

# 影像預處理 (適用於 MobileNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class YOLOThread(QThread):
    # 訊號傳遞: 更新畫面與文字顯示
    update_frame = pyqtSignal(QImage)
    update_text = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.paused = False
        # 初始化 YOLO 模型
        # 初始化 MobileNetV3 模型
        self.cap = cv2.VideoCapture(input_data_path)

    def predict_mobilenet(self, image):
        # 使用 MobileNetV3 進行分類預測
        image = Image.fromarray(image).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        outputs = mobilenet_model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)
        return predicted.item(), confidence.item()

    def run(self):
        # 執行偵測與追蹤
        self.running = True
        while self.cap.isOpened() and self.running:
            if self.paused:
                continue
            ret, frame = self.cap.read()
            if not ret:
                break

            # YOLO 物件偵測與追蹤
            results = model.track(source=frame, conf=0.2, tracker="bytetrack.yaml", persist=True)
            try:
                inference_time = results[0].speed['inference']
                FPS = 1000 / inference_time
            except:
                FPS = 0

            # 顯示 FPS
            cv2.putText(frame, f'FPS: {FPS:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

            detected_texts = []
            for result in results:
                boxes = result.boxes
                for i in range(len(boxes)):
                    box_values = boxes.data[i].tolist()
                    if len(box_values) == 7:
                        x1, y1, x2, y2, track_id, conf, cls_id = map(int, box_values)
                    else:
                        continue

                    # 進行 MobileNetV3 分類
                    cropped_img = frame[y1:y2, x1:x2]
                    mobilenet_predicted_class, mobilenet_confidence = self.predict_mobilenet(cropped_img)
                    mobilenet_class_name = mobilenet_class_dict[mobilenet_predicted_class]

                    cls_name = cls_map.get(cls_id, "unknown")
                    color = cls_colors.get(cls_id, (255, 255, 255))

                    # 畫出偵測框與標籤
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'ID {track_id} - {cls_name}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    # detected_texts.append(f'ID {track_id}: {cls_name}')

            # 轉換影像格式，傳遞至 UI
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            qt_img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
            self.update_frame.emit(qt_img)
            self.update_text.emit('\n'.join(detected_texts))

        self.cap.release()

    def pause(self):
        # 暫停執行
        self.paused = True

    def resume(self):
        # 恢復執行
        self.paused = False

    def stop(self):
        # 停止偵測，完全關閉
        self.running = False
        self.quit()
        self.wait()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Detection UI")
        self.setGeometry(100, 100, 800, 600)

        # 建立 UI 元件
        self.image_label = QLabel(self)
        self.text_output = QTextEdit(self)
        self.text_output.setReadOnly(True)
        self.start_button = QPushButton("Start Detection", self)
        self.stop_button = QPushButton("Pause Detection", self)
        self.close_button = QPushButton("Close Application", self)

        # 設定 UI 佈局
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.text_output)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.close_button)
        self.setLayout(layout)

        # 設定按鈕事件
        self.yolo_thread = YOLOThread()
        self.yolo_thread.update_frame.connect(self.display_image)
        self.yolo_thread.update_text.connect(self.update_text)
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.pause_detection)
        self.close_button.clicked.connect(self.close_application)

    def display_image(self, qt_img):
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def update_text(self, text):
        self.text_output.setText(text)

    def start_detection(self):
        if self.yolo_thread.paused:
            self.yolo_thread.resume()
        elif not self.yolo_thread.isRunning():
            self.yolo_thread.start()

    def pause_detection(self):
        self.yolo_thread.pause()

    def close_application(self):
        self.yolo_thread.stop()
        self.close()


if __name__ == '__main__':
    model = YOLO("best.engine")
    mobilenet_model =mobilenetv3_large_SimAM(num_classes=3,width_mult=0.5).to(device="cuda")
    model_path = './best_mobilenet_epoch_255.pth'
    mobilenet_model.load_state_dict(torch.load(model_path))
    mobilenet_model.eval()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
