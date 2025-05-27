import cv2
import os
import subprocess
import time
from datetime import datetime
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from ultralytics import YOLO
from mobilenetv3_model import mobilenetv3_large_SimAM
import customtkinter as ctk
from tkinter import filedialog
import tkinter as tk  # 用於 Listbox


# 定義預測函式
def predict_mobilenet(model, image, transform, device):
    image = Image.fromarray(image).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # 增加 batch 維度
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)
    return predicted.item(), confidence.item()


# 定義類別對應資訊
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
    0: (0, 0, 255),  # 藍色 (transverse cracks)
    1: (0, 255, 0),  # 綠色 (longitudinal cracks)
    2: (255, 0, 0),  # 紅色 (alligator cracks)
    3: (0, 165, 255),  # 青色 (potholes)
    4: (128, 0, 128),  # 粉色 (manhole)
    5: (255, 0, 255),  # 黃色 (speed bump)
    6: (42, 42, 165)  # 紫色 (expansion joint)
}


# 使用 customtkinter 改寫的介面
class DetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RoadScan Pro")
        self.root.geometry("900x630")
        self.group_info = []  # 每個元素為 (start_index, end_index, image_path)


        # 儲存辨識記錄影像的路徑與已開啟的視窗
        self.info_items = []
        self.open_windows = {}  # 鍵值：影像路徑，值：對應的 Toplevel 視窗
        self.class_counts = {value: 0 for value in cls_map.values()}
        self.count_track_id = []
        self.previous_value = None
        self.detection_active = False
        self.video_source = 0  # 初始使用攝像頭

        # 建立用來顯示即時影像的 Label
        self.video_label = ctk.CTkLabel(self.root, text="", fg_color="black", width=600, height=385)
        self.video_label.place(x=220, y=70)

        # Logo 顯示
        self.logo_label = ctk.CTkLabel(self.root, text="RoadScan Pro", font=("Verdana", 25),
                                       fg_color="white", text_color="black", width=260, height=40)
        self.logo_label.place(x=400, y=10)

        # 顯示類別名稱（以純文字呈現）
        class_text = ("Transverse cracks\n"
                      "Longitudinal cracks\n"
                      "Alligator cracks\n"
                      "Potholes\n"
                      "Manhole\n"
                      "Speed bump\n"
                      "Expansion joint\n"
                      "辨識數量")
        self.class_label = ctk.CTkLabel(self.root, text=class_text, font=("Arial", 14),
                                        fg_color="white", text_color="black", width=180, height=250, anchor="w")
        self.class_label.place(x=10, y=10)

        # 顯示各類別辨識數量
        self.count_label = ctk.CTkLabel(self.root, font=("Arial", 14),
                                        fg_color="white", text_color="black", width=80, height=250, anchor="nw")
        self.count_label.place(x=180, y=10)
        self.update_counter()

        # 建立各按鈕
        self.start_button = ctk.CTkButton(self.root, text="開始辨識", command=self.start_detection,
                                          font=("Microsoft JhengHei", 12, "bold"), width=100, height=66)
        self.start_button.place(x=220, y=470)

        self.stop_button = ctk.CTkButton(self.root, text="停止辨識", command=self.stop_detection,
                                         font=("Microsoft JhengHei", 12, "bold"), width=100, height=66)
        self.stop_button.place(x=345, y=470)

        self.exit_button = ctk.CTkButton(self.root, text="結束程式", command=self.close_program,
                                         font=("Microsoft JhengHei", 12, "bold"), width=100, height=66)
        self.exit_button.place(x=470, y=470)

        self.open_directory_button = ctk.CTkButton(self.root, text="打開儲存路徑", command=self.open_directory,
                                                   font=("Microsoft JhengHei", 12, "bold"), width=100, height=66)
        self.open_directory_button.place(x=595, y=470)

        self.switch_input_button = ctk.CTkButton(self.root, text="切換輸入來源", command=self.switch_input_source,
                                                 font=("Microsoft JhengHei", 12, "bold"), width=100, height=66)
        self.switch_input_button.place(x=720, y=470)

        # 建立 Listbox 顯示辨識記錄（採用 tkinter 原生 Listbox）
        self.info_list = tk.Listbox(self.root, font=("Arial", 12))
        self.info_list.place(x=10, y=250, width=200, height=315)
        # 使用 <ButtonRelease-1> 避免重複觸發
        self.info_list.bind("<ButtonRelease-1>", self.display_image)

        # 開啟攝像頭
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            print("無法打開攝像頭")
            return

        self.fps_text = ""

    def update_counter(self):
        total_items = len(self.info_items)
        text = ""
        for cls_name in ["Transverse cracks", "Longitudinal cracks", "Alligator cracks",
                         "Potholes", "Manhole", "Speed bump", "Expansion joint"]:
            count = self.class_counts.get(cls_name, 0)
            text += f"{cls_name}: {count}\n"
        text += f"Total: {total_items}"
        self.count_label.configure(text=text)

    def start_detection(self):
        if not self.detection_active:
            print("開始辨識")
            self.detection_active = True
            self.update_frame()

    def stop_detection(self):
        if self.detection_active:
            print("暫停辨識")
            self.detection_active = False

    def open_directory(self):
        output_dir = r"D:\yolov11_custom\yolov11-main\ultralytics-main\RDD_image"
        if os.path.exists(output_dir):
            if os.name == "nt":
                os.startfile(output_dir)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", output_dir])
            else:
                subprocess.Popen(["xdg-open", output_dir])
        else:
            print(f"路徑 {output_dir} 不存在")

    def switch_input_source(self):
        if self.detection_active:
            print("請先停止辨識再切換輸入來源")
            return
        file_path = filedialog.askopenfilename(title="選擇影片文件",
                                               filetypes=[("影片文件", "*.mp4;*.avi;*.mov"), ("所有文件", "*.*")])
        if file_path:
            self.video_source = file_path
        else:
            self.video_source = 0
        self.cap.release()
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            print(f"無法打開視頻來源: {self.video_source}")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret and self.detection_active:
            start_time = time.time()
            frame = cv2.resize(frame, (600, 385))
            # 呼叫 YOLO 進行追蹤與偵測
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
                    mobilenet_predicted_class, mobilenet_confidence = predict_mobilenet(mobilenet_model, cropped_img,
                                                                                        transform, device)
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
                    cv2.putText(frame, self.fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                cv2.LINE_AA)
                    # 追蹤 ID 處理
                    track_id_trigger = False
                    if self.previous_value is None or track_id != self.previous_value:
                        if track_id not in self.count_track_id:
                            self.count_track_id.append(track_id)
                            track_id_trigger = True
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

                            # 將每筆資訊分成多個 Listbox 項目插入
                            start_index = self.info_list.size()  # 記錄起始索引
                            self.info_list.insert(tk.END, f"類別名稱: {cls_names}")
                            self.info_list.insert(tk.END, f"時間: {currentDateAndTime}")
                            self.info_list.insert(tk.END, f"嚴重程度: {mobilenet_class_name}")
                            self.info_list.insert(tk.END, "-------------------")
                            end_index = self.info_list.size() - 1  # 取得最後一行的索引

                            # 將該筆資訊的索引範圍及影像路徑記錄到群組清單中
                            self.group_info.append((start_index, end_index, image_path))

                            self.info_items.append(image_path)
                            if cls_names in self.class_counts:
                                self.class_counts[cls_names] += 1
                            self.update_counter()
                            print(f"當前辨識數量: {len(self.info_items)}")

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(rgb_image)
            imgtk = ImageTk.PhotoImage(image_pil)
            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk
        if self.detection_active:
            self.root.after(10, self.update_frame)

    def display_image(self, event):
        # 取得點擊的項目索引
        selection = event.widget.curselection()
        if not selection:
            return
        clicked_index = selection[0]

        # 檢查點擊的索引屬於哪一個群組
        for group in self.group_info:
            start, end, image_path = group
            if start <= clicked_index <= end:
                # 如果該影像視窗已經開啟，就關閉
                if image_path in self.open_windows:
                    self.open_windows[image_path].destroy()
                    del self.open_windows[image_path]
                    print(f"關閉影像視窗: {image_path}")
                else:
                    # 否則開啟新的影像視窗
                    top = ctk.CTkToplevel(self.root)
                    top.title("影像顯示")

                    def on_close():
                        if image_path in self.open_windows:
                            del self.open_windows[image_path]
                        top.destroy()

                    top.protocol("WM_DELETE_WINDOW", on_close)
                    image = Image.open(image_path)
                    imgtk = ImageTk.PhotoImage(image)
                    label = ctk.CTkLabel(top, text="", image=imgtk)
                    label.image = imgtk  # 保持參考
                    label.pack(padx=10, pady=10)
                    self.open_windows[image_path] = top
                    print(f"打開影像視窗: {image_path}")
                break

    def close_program(self):
        print("結束程式")
        self.detection_active = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.quit()


if __name__ == "__main__":
    # 全域參數設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    num_classes = 3  # 根據實際分類數設定
    mobilenet_model = mobilenetv3_large_SimAM(num_classes=num_classes, width_mult=0.5).to(device)
    model_path = './best_mobilenet_epoch_255.pth'
    mobilenet_model.load_state_dict(torch.load(model_path))
    mobilenet_model.eval()
    trt_engine = YOLO("best.engine")

    root = ctk.CTk()
    app = DetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close_program)
    root.mainloop()
