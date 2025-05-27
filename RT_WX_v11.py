import wx
import cv2
import os
import sys
import time
import subprocess
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import torch
from torchvision import transforms
from mobilenetv3_model import mobilenetv3_large_SimAM


def predict_mobilenet(model, image, transform, device):
    from PIL import Image
    image = Image.fromarray(image).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)
    return predicted.item(), confidence.item()


# 類別與顏色設定
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
mobilenet_class_dict = {
    0: "A",
    1: "B",
    2: "C",
}
cls_colors = {
    0: (0, 0, 255),  # 藍色 - transverse cracks
    1: (0, 255, 0),  # 綠色 - longitudinal cracks
    2: (255, 0, 0),  # 紅色 - alligator cracks
    3: (0, 165, 255),  # 青色 - potholes
    4: (128, 0, 128),  # 粉色 - manhole
    5: (255, 0, 255),  # 黃色 - speed bump
    6: (42, 42, 165)  # 紫色 - expansion joint
}


class DetectionFrame(wx.Frame):
    def __init__(self, parent=None):
        wx.Frame.__init__(self, parent, title="RoadScan Pro", size=(900, 630))
        self.panel = wx.Panel(self)
        self.panel.SetBackgroundColour("white")

        # 用來記錄各類別統計、追蹤ID等
        self.class_counts = {v: 0 for v in cls_map.values()}
        self.count_track_id = []
        self.previous_value = None
        self.detection_active = False
        self.current_image_path = None
        self.fps_text = "FPS: 0"

        # 攝影畫面顯示區（使用 wx.StaticBitmap）
        self.video_display = wx.StaticBitmap(self.panel, pos=(220, 70), size=(600, 385))

        # 標題區：Logo
        self.logo_label = wx.StaticText(self.panel, label="RoadScan Pro", pos=(400, 10), size=(260, 40))
        logo_font = wx.Font(25, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, "Verdana")
        self.logo_label.SetFont(logo_font)

        # 左上角類別標籤（以多行文字方式呈現）
        class_label_text = (
            "Transverse cracks (red)\n"
            "Longitudinal cracks (green)\n"
            "Alligator cracks (blue)\n"
            "Potholes (orange)\n"
            "Manhole (purple)\n"
            "Speed bump (magenta)\n"
            "Expansion joint (brown)\n"
            "辨識數量"
        )
        self.class_label = wx.StaticText(self.panel, label=class_label_text, pos=(10, 10), size=(180, 250))
        font_small = wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.class_label.SetFont(font_small)

        # 顯示各類別統計與總數的區域
        self.count_label = wx.StaticText(self.panel, label="0", pos=(180, 10), size=(100, 250))
        self.count_label.SetFont(font_small)

        # 建立按鈕區：開始、停止、結束、打開儲存路徑、切換輸入來源
        self.start_button = wx.Button(self.panel, label="開始辨識", pos=(220, 470), size=(100, 66))
        self.stop_button = wx.Button(self.panel, label="停止辨識", pos=(345, 470), size=(100, 66))
        self.exit_button = wx.Button(self.panel, label="結束程式", pos=(470, 470), size=(100, 66))
        self.open_directory_button = wx.Button(self.panel, label="打開儲存路徑", pos=(595, 470), size=(100, 66))
        self.switch_input_button = wx.Button(self.panel, label="切換輸入來源", pos=(720, 470), size=(100, 66))

        self.start_button.Bind(wx.EVT_BUTTON, self.OnStart)
        self.stop_button.Bind(wx.EVT_BUTTON, self.OnStop)
        self.exit_button.Bind(wx.EVT_BUTTON, self.OnExit)
        self.open_directory_button.Bind(wx.EVT_BUTTON, self.OnOpenDirectory)
        self.switch_input_button.Bind(wx.EVT_BUTTON, self.OnSwitchInput)

        # 建立一個捲動視窗，用來動態加入代表偵測項目的按鈕
        self.info_panel = wx.ScrolledWindow(self.panel, pos=(10, 250), size=(200, 315), style=wx.VSCROLL)
        self.info_panel.SetScrollRate(5, 5)
        self.info_sizer = wx.BoxSizer(wx.VERTICAL)
        self.info_panel.SetSizer(self.info_sizer)

        # 影像預覽區（用於顯示按鈕點選後的影像）
        # self.image_display = wx.StaticBitmap(self.panel, pos=(250, 100), size=(420, 240))

        # 設定定時器，每10毫秒更新一次影像
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.UpdateFrame, self.timer)

        # 初始化攝影機
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("無法打開攝像頭")

        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def OnStart(self, event):
        if not self.detection_active:
            print("開始辨識")
            self.detection_active = True
            self.timer.Start(10)

    def OnStop(self, event):
        if self.detection_active:
            print("暫停辨識")
            self.detection_active = False
            self.timer.Stop()

    def OnExit(self, event):
        print("結束程式")
        self.Close()

    def OnOpenDirectory(self, event):
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

    def OnSwitchInput(self, event):
        if self.detection_active:
            print("請先停止辨識再切換輸入來源")
            return
        wildcard = "影片文件 (*.mp4;*.avi;*.mov)|*.mp4;*.avi;*.mov|所有文件 (*)|*.*"
        with wx.FileDialog(self, "選擇影片文件", wildcard=wildcard,
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                self.video_source = 0
            else:
                self.video_source = fileDialog.GetPath()
        self.cap.release()
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            print(f"無法打開視頻來源: {self.video_source}")

    def UpdateFrame(self, event):
        ret, frame = self.cap.read()
        if ret and self.detection_active:
            start_time = time.time()
            frame = cv2.resize(frame, (600, 385))

            results = trt_engine.track(
                source=frame,
                conf=0.2,
                tracker="bytetrack.yaml",
                persist=True
            )

            for result in results:
                boxes = result.boxes
                for i in range(len(boxes)):
                    box_values = boxes.data[i].tolist()
                    if len(box_values) == 7:
                        x1, y1, x2, y2, track_id, conf, cls_id = box_values
                    else:
                        continue

                    cropped_img = frame[int(y1):int(y2), int(x1):int(x2)]
                    mobilenet_predicted_class, mobilenet_confidence = predict_mobilenet(
                        mobilenet_model, cropped_img, transform, device)
                    mobilenet_class_name = mobilenet_class_dict[mobilenet_predicted_class]
                    cls_name = cls_map.get(int(cls_id), "unknown")
                    color = cls_colors.get(int(cls_id), (255, 255, 255))
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'ID {track_id} -{mobilenet_class_name}- {cls_name}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                        fps = 1.0 / elapsed_time
                        self.fps_text = f"FPS: {fps:.2f}"
                    cv2.putText(frame, self.fps_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    # 追蹤ID處理：避免重複儲存同一物件
                    track_id_trigger = False
                    if self.previous_value is None or track_id != self.previous_value:
                        if track_id not in self.count_track_id:
                            self.count_track_id.append(track_id)
                            track_id_trigger = True
                            print(track_id_trigger)
                    self.previous_value = track_id

                    if track_id_trigger and results:
                        if int(cls_id) in class_dict:
                            cls_names = class_dict[int(cls_id)]
                            currentDateAndTime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            output_dir = f"./RDD_image/{cls_names}"
                            image_path = f"{output_dir}/{cls_names}_{currentDateAndTime}.png"
                            print(cls_names)
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            cv2.imwrite(image_path, frame)
                            if os.path.exists(image_path):
                                print(f"影像已儲存: {image_path}")
                            else:
                                print(f"影像未成功儲存: {image_path}")

                            # 將偵測資訊以按鈕形式新增到 info_panel 中
                            list_item_text = (
                                f"類別名稱: {cls_names}\n"
                                f"時間: {currentDateAndTime}\n"
                                f"嚴重程度: {mobilenet_class_name}\n"
                                "-------------------"
                            )
                            btn = wx.Button(self.info_panel, label=list_item_text)
                            # 將對應影像路徑存入按鈕屬性
                            btn.image_path = image_path
                            btn.Bind(wx.EVT_BUTTON, self.OnDetectionButtonClick)
                            self.info_sizer.Add(btn, 0, wx.ALL | wx.EXPAND, 5)
                            self.info_panel.Layout()
                            self.info_panel.FitInside()
                            if cls_name in self.class_counts:
                                self.class_counts[cls_name] += 1
                            self.UpdateCounter()
                            print(f"當前偵測項目數: {self.info_sizer.GetItemCount()}")

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = rgb_image.shape[:2]
            img = wx.Image(width, height)
            img.SetData(rgb_image.tobytes())
            bitmap = wx.Bitmap(img)
            self.video_display.SetBitmap(bitmap)
            self.panel.Refresh()

    def UpdateCounter(self):
        total_items = self.info_sizer.GetItemCount()
        counts_str = "\n".join(f"{cls}: {count}" for cls, count in self.class_counts.items())
        self.count_label.SetLabel(f"{counts_str}\n總數: {total_items}")

    def OnDetectionButtonClick(self, event):
        btn = event.GetEventObject()
        image_path = btn.image_path
        if hasattr(self, "current_image_path") and self.current_image_path == image_path:
            self.image_display.SetBitmap(wx.NullBitmap)
            self.current_image_path = None
            print("關閉影像顯示")
            return
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                wx_img = wx.Image(w, h)
                wx_img.SetData(img.tobytes())
                bitmap = wx.Bitmap(wx_img)
                self.image_display.SetBitmap(bitmap)
                self.current_image_path = image_path
                print(f"顯示影像: {image_path}")
            else:
                print(f"加載影像失敗: {image_path}")
        else:
            print(f"影像文件不存在: {image_path}")

    def OnClose(self, event):
        self.cap.release()
        self.Destroy()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    num_classes = 3
    mobilenet_model = mobilenetv3_large_SimAM(num_classes=num_classes, width_mult=0.5).to(device)
    model_path = './best_mobilenet_epoch_255.pth'
    mobilenet_model.load_state_dict(torch.load(model_path))
    mobilenet_model.eval()

    trt_engine = YOLO("best.engine")

    app = wx.App(False)
    frame = DetectionFrame()
    frame.Show()
    app.MainLoop()
