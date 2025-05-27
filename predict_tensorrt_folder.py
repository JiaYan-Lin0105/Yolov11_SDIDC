from ultralytics import YOLO
import os
import cv2
from mobilenetv3_model import mobilenetv3_large_SimAM
import torch
from torchvision import transforms
from PIL import Image
import time
import numpy as np
from Lite_Unet import LiteUNet
import glob

# 輸入／輸出資料夾
input_folder = r'D:\road_damage\frames\20240904_800_600'       # 放所有待處理圖片的資料夾
output_folder = r'./output/frame'
os.makedirs(output_folder, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 類別對應表
cls_map = {
    0: 'longitudinal cracks',
    1: 'transverse cracks',
    2: 'alligator cracks',
    3: 'potholes',
    4: 'manhole',
    5: 'speed bump',
    6: 'expansion joint'
}
severity_colors = {
    0: (0, 255, 0),
    1: (0, 255, 255),
    2: (0, 0, 255)
}
mobilenet_class_dict = {0: "A", 1: "B", 2: "C"}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_mobilenet(model, image, transform, device):
    image = Image.fromarray(image).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    outputs = model(image)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    conf, pred = probs.max(1)
    return pred.item(), conf.item()

def Unet_predict_frame(model, frame, device):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    img_tensor = t(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_tensor)
    prob_map = out.squeeze().cpu().numpy()
    mask = (prob_map > 0.5).astype(np.uint8) * 255
    orig = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    disp = (prob_map * 255).astype(np.uint8)
    disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
    return orig, disp, mask

def load_unet(path, device):
    model = LiteUNet(num_classes=1)
    sd = torch.load(path, map_location=device)
    sd = {k:v for k,v in sd.items() if 'total_ops' not in k and 'total_params' not in k}
    model.load_state_dict(sd)
    model.to(device).eval()
    return model

if __name__ == '__main__':
    # 載入模型
    yolo_model       = YOLO("best.engine", task='detect')
    mobilenet_model  = mobilenetv3_large_SimAM(num_classes=3, width_mult=0.5).to(device)
    mobilenet_model.load_state_dict(torch.load('./best_mobilenet_epoch_255.pth', map_location=device))
    mobilenet_model.eval()
    # unet_model = load_unet("Lite_Unet_custom_v1_E200.pth", device)

    # 讀取所有圖片檔（支援 jpg, png, bmp）
    img_paths = sorted(glob.glob(os.path.join(input_folder, '*.*')))
    start = time.time()

    track_results = {}  # 若想跨圖追蹤同一目標可保留；不需要則可每張圖都重置

    for idx, img_path in enumerate(img_paths, 1):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"無法讀取：{img_path}, 跳過。")
            continue

        print(f"[{idx}/{len(img_paths)}] 處理：{os.path.basename(img_path)}")

        # 1. Unet 分割 + 裁切
        # _, _, mask = Unet_predict_frame(unet_model, frame, device)
        # mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        # cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # if cnts:
        #     c = max(cnts, key=cv2.contourArea)
        #     x,y,w,h = cv2.boundingRect(c)
        #     frame = frame[y:y+h, x:x+w]

        # 2. YOLO 偵測 + Bytetrack
        results = yolo_model.track(source=frame, conf=0.1,iou=0.1, tracker="bytetrack.yaml", persist=True)
        for res in results:
            for box in res.boxes.data.tolist():
                if len(box) != 7:
                    continue
                x1,y1,x2,y2,tid,conf,cls_id = box
                x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
                # MobileNet 分類（只第一次遇到該 track_id 時做）
                if tid not in track_results:
                    crop = frame[y1:y2, x1:x2]
                    pred_cls, pred_conf = predict_mobilenet(mobilenet_model, crop, transform, device)
                    track_results[tid] = (pred_cls, pred_conf)
                else:
                    pred_cls, pred_conf = track_results[tid]

                # 畫框 + 標籤
                clr = severity_colors[pred_cls]
                cv2.rectangle(frame, (x1,y1), (x2,y2), clr, 2)
                label = f"ID:{int(tid)}-{cls_map[int(cls_id)]}-{mobilenet_class_dict[pred_cls]}"
                tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                lx = x1 + (x2-x1-tw)//2
                ly = y1 - 10
                cv2.putText(frame, label, (lx,ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                cv2.putText(frame, label, (lx,ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # 3. 存檔
        out_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_folder, out_name), frame)

    elapsed = time.time() - start
    print(f"全部圖片處理完成，用時：{elapsed:.2f} 秒")
