import cv2
import sys
import os
import tensorrt as trt
import torch
from torchvision import transforms
from mobilenetv3_model import mobilenetv3_large_SimAM
import subprocess  # 添加這行以匯入 subprocess
import numpy as np
from collections import OrderedDict,namedtuple
import sqlite3
import time
from PIL import Image
import Jetson.GPIO as GPIO
import time as time
import serial
from datetime import datetime
#from bytetrack import ByteTrack
import ultralytics
from ultralytics.trackers.byte_tracker import BYTETracker
import argparse
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget,QHBoxLayout,QTextEdit,QListWidget,QListWidgetItem,QLabel,QFileDialog
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap,QFont
from PyQt5.QtCore import QTimer, Qt
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
def predict_mobilenet(model, image, transform, device):
    image = Image.fromarray(image).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # 增加 batch 維度

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # 計算每個類別的概率
        confidence, predicted = probabilities.max(1)  # 最大概率值及其索引

    return predicted.item(), confidence.item()

class_dict = {
    0:"longitudinal cracks",
    1:"transverse cracks",
    2:"alligator cracks",
    3:"potholes",
    }
global ser 
parser = argparse.ArgumentParser()

parser.add_argument("--track_high_thresh", type=float, default=0.5, help="tracking confidence threshold")
parser.add_argument("--track_low_thresh", type=float, default=0.1, help="tracking confidence threshold")
parser.add_argument("--new_track_thresh", type=float, default=0.5, help="tracking confidence threshold")
parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
parser.add_argument("--match_thresh", type=float, default=0.7, help="matching threshold for tracking")
parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
args = parser.parse_args()
tracker=BYTETracker(args,frame_rate=30)
ser = serial.Serial('/dev/ttyTHS0', 9600,timeout=0.5)
def recv(serial):
	global date
	while True:
		date = serial.readline()
		print(date)
		if date == "":
			continue
		else:
			break
		sleep(0.02)
	return date
def parse_gpgga(gpgga_string):
    date = gpgga_string.split(',')
    if date[2] and date[4]:
        latitude = float(date[2][:2]) + float(date[2][2:]) / 60.0
        longitude = float(date[4][:3]) + float(date[4][3:]) / 60.0
        if date[3] == 'S':
            latitude = -latitude
        if date[5] == 'W':
            longitude = -longitude
        return latitude, longitude
    else:
        return None
def record_database(class_names,time):
	print(f'record_datebase_{class_names}')

	date = recv(ser)

	date = date.decode("utf-8")
	print(date)
	if "GPGGA" not in date:
		print("gps_save_failure")
	if "GPGGA" in date:
		latitude, longitude = parse_gpgga(date)
		print(latitude, longitude)
		insert_coordinates_to_db(latitude,longitude,class_names,time)
def insert_coordinates_to_db(latitude, longitude,defect,time):
    print(latitude, longitude)
    conn = sqlite3.connect('coordinates_20240419.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL,
            longitude REAL,
            defect REAL,
            time REAL
        )
    ''')
    cursor.execute('INSERT INTO locations (latitude, longitude,defect,time) VALUES (?, ?, ?, ?)', (latitude, longitude,defect,time))
    conn.commit()
    cursor.close()
    conn.close()
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
       self.trt_engine = TRT_engine("./new_best_fp16.engine")  # 確保 TRT_engine 被正確初始化
       self.detection_active = False
       fourcc = cv2.VideoWriter_fourcc(*'XVID')
       self.output_video = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))

       # 用於 FPS 計算的變數
       self.fps_text = "FPS: 0"
   def open_directory(self):
        # 打開影像的儲存路徑
        output_dir = "./RDD_image"
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
        video_file, _ = file_dialog.getOpenFileName(self, "選擇影片文件", "", "影片文件 (*.mp4 *.avi *.mov);;所有文件 (*)")
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
        frame = cv2.resize(frame, (1280, 720))
        results = self.trt_engine.predict(frame, threshold=0.5)
        frame, track_id = visualize(frame, results, tracker)

        # 結束計時，計算從預測到辨識完成的時間
        elapsed_time = time.time() - start_time

        if elapsed_time > 0:
            fps = 1.0 / elapsed_time
            self.fps_text = f"FPS: {fps:.2f}"

        # 將 FPS 顯示在左上角
        cv2.putText(frame, self.fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 追蹤 ID 的處理
        track_id_trigger = False  # 默認設置為 False
        if self.previous_value is None or track_id != self.previous_value:
            if track_id not in self.count_track_id:
                self.count_track_id.append(track_id)
                track_id_trigger = True
        self.previous_value = track_id

        if track_id_trigger and results:
        	class_id = results[0][0]
        	if class_id in class_dict:
        		cls_names = class_dict[class_id]
        		currentDateAndTime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        		output_dir = f"./RDD_image/{cls_names}"
        		image_path = f"{output_dir}/{cls_names}_{currentDateAndTime}.png"
        		if not os.path.exists(output_dir):
        			os.makedirs(output_dir)
        		cv2.imwrite(image_path, frame)
        		
        		
        		list_item_text = f"影像名稱: {os.path.basename(image_path)}\n類別名稱: {cls_names}\n時間: {currentDateAndTime}\n---------------"
        		list_item = QListWidgetItem(list_item_text)
        		list_item.setData(Qt.UserRole, image_path)  # 將影像路徑存儲在項目中
        		
        		self.info_list.addItem(list_item)
        		self.update_counter()
        self.output_video.write(frame)
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

class TRT_engine():
    def __init__(self, weight) -> None:
        self.imgsz = [640,640]
        self.weight = weight
        self.device = torch.device('cuda:0')
        self.init_engine()

    def init_engine(self):
        # Infer TensorRT Engine
        self.Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        self.logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(self.weight, 'rb') as self.f, trt.Runtime(self.logger) as self.runtime:
            self.model = self.runtime.deserialize_cuda_engine(self.f.read())
        self.bindings = OrderedDict()
        self.fp16 = False
        for index in range(self.model.num_bindings):
            self.name = self.model.get_binding_name(index)
            self.dtype = trt.nptype(self.model.get_binding_dtype(index))
            self.shape = tuple(self.model.get_binding_shape(index))
            self.data = torch.from_numpy(np.empty(self.shape, dtype=np.dtype(self.dtype))).to(self.device)
            self.bindings[self.name] = self.Binding(self.name, self.dtype, self.shape, self.data, int(self.data.data_ptr()))
            if self.model.binding_is_input(index) and self.dtype == np.float16:
                self.fp16 = True
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = self.model.create_execution_context()

    def letterbox(self,im,color=(114, 114, 114), auto=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        new_shape = self.imgsz
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # Scale ratio (new / old)
        self.r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            self.r = min(self.r, 1.0)
        # Compute padding
        new_unpad = int(round(shape[1] * self.r)), int(round(shape[0] * self.r))
        self.dw, self.dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            self.dw, self.dh = np.mod(self.dw, stride), np.mod(self.dh, stride)  # wh padding
        self.dw /= 2  # divide padding into 2 sides
        self.dh /= 2
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(self.dh - 0.1)), int(round(self.dh + 0.1))
        left, right = int(round(self.dw - 0.1)), int(round(self.dw + 0.1))
        self.img = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return self.img,self.r,self.dw,self.dh

    def preprocess(self,image):
        self.img,self.r,self.dw,self.dh = self.letterbox(image)
        self.img = self.img.transpose((2, 0, 1))
        self.img = np.expand_dims(self.img,0)
        self.img = np.ascontiguousarray(self.img)
        self.img = torch.from_numpy(self.img).to(self.device)
        self.img = self.img.float()
        return self.img

    def predict(self,img,threshold):
        img = self.preprocess(img)
        self.binding_addrs['images'] = int(img.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        nums = self.bindings['num_dets'].data[0].tolist()
        boxes = self.bindings['det_boxes'].data[0].tolist()
        scores =self.bindings['det_scores'].data[0].tolist()
        classes = self.bindings['det_classes'].data[0].tolist()

        num = int(nums[0])
        new_bboxes = []
        for i in range(num):
            if(scores[i] < threshold):
                continue
            xmin = (boxes[i][0] - self.dw)/self.r
            ymin = (boxes[i][1] - self.dh)/self.r
            xmax = (boxes[i][2] - self.dw)/self.r
            ymax = (boxes[i][3] - self.dh)/self.r
            new_bboxes.append([classes[i],scores[i],xmin,ymin,xmax,ymax])
        return new_bboxes

def visualize(img,bbox_array,tracker):

    track_id=0
    color_map = {
        0: (0, 255, 0),    # 綠色
        1: (0, 255, 255),  # 黃色
        2: (0, 0, 255)     # 紅色
    }
    mobilenet_class_dict = {
        0: "A",
        1: "B",
        2: "C",
    }

    for temp in bbox_array:
        xmin = int(temp[2])
        ymin = int(temp[3])
        xmax = int(temp[4])
        ymax = int(temp[5])
        clas = int(temp[0])
        score = temp[1]
        if temp != []:
        	yolo_output=np.array([[temp[2],temp[3],temp[4],temp[5],score,clas]])
        	trackre_data = ultralytics.engine.results.Boxes(yolo_output,(img.shape[0],img.shape[1]))
        	track_result = tracker.update(trackre_data)
        	
        	cropped_img = img[ymin:ymax, xmin:xmax]
        	mobilenet_predicted_class, mobilenet_confidence = predict_mobilenet(mobilenet_model, cropped_img, transform, device)
        	
        	mobilenet_class_name = mobilenet_class_dict[mobilenet_predicted_class]
        	box_color = color_map.get(mobilenet_predicted_class, (255, 255, 255)) 
        	cv2.rectangle(img,(xmin,ymin),(xmax,ymax), box_color, 2)
        	
        	for result in track_result:
        		track_id=int(result[4])
        		img = cv2.putText(img, "id:"+str(track_id), (xmin-50,int(ymin)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        		#cv2.rectangle(img,(xmin,ymin),(xmax,ymax), box_color, 2)
        if clas in class_dict:

        	label = f"{class_dict[clas]} ({mobilenet_class_name})"
        img = cv2.putText(img, " " +str(label)+" "+str(round(score,2)), (xmin,int(ymin)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (105, 237, 249), 2)
    return img,track_id
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DetectionApp()
    window.show()
    sys.exit(app.exec_())

