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

# 影片路徑
input_video_path = r'D:\road_damage\Produce_old.avi'
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("無法開啟影片，請確認路徑與檔案格式是否正確。")
    exit()

# 輸出資料夾設定，若不存在則建立
output_path = r'./output/frame'
if not os.path.exists(output_path):
    os.makedirs(output_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 類別對應表 (注意此處 longitudinal/transverse 的定義依需求調整)
cls_map = {
    0: 'longitudinal cracks',
    1: 'transverse cracks',
    2: 'alligator cracks',
    3: 'potholes',
    4: 'manhole',
    5: 'speed bump',
    6: 'expansion joint'
}

# 損壞嚴重度顏色 (綠：輕度, 黃：中度, 紅：重度)
severity_colors = {
    0: (0, 255, 0),
    1: (0, 255, 255),
    2: (0, 0, 255)
}

# 類別顏色 (BGR 格式) - 若需要可另外使用
cls_colors = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (255, 0, 255),
    5: (0, 255, 255),
    6: (128, 0, 128)
}

mobilenet_class_dict = {
    0: "A",
    1: "B",
    2: "C",
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_mobilenet(model, image, transform, device):
    # 將 numpy image 轉換為 PIL Image
    image = Image.fromarray(image).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    outputs = model(image)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted = probabilities.max(1)
    return predicted.item(), confidence.item()

def Unet_predict_frame(model, frame, device):
    """
    將影片中的 frame (BGR) 轉換為 PIL Image 處理後，
    回傳原始影像 (BGR)、概率圖 (COLORMAP_JET) 與二值 mask。
    """
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transform_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform_img(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
    pred_prob = output.squeeze().cpu().numpy()
    pred_mask = (pred_prob > 0.5).astype(np.uint8) * 255
    orig_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    prob_disp = (pred_prob * 255).astype(np.uint8)
    prob_disp = cv2.applyColorMap(prob_disp, cv2.COLORMAP_JET)
    return orig_img, prob_disp, pred_mask

def load_model(model_path, device):
    model = LiteUNet(num_classes=1)
    state_dict = torch.load(model_path, map_location=device)
    filtered_state_dict = {k: v for k, v in state_dict.items() if 'total_ops' not in k and 'total_params' not in k}
    model.load_state_dict(filtered_state_dict)
    model.to(device)
    model.eval()
    return model

if __name__ == '__main__':
    # 載入模型
    yolo_model = YOLO("best.engine", task='detect')
    mobilenet_model = mobilenetv3_large_SimAM(num_classes=3, width_mult=0.5).to(device)
    mobilenet_model_path = './best_mobilenet_epoch_255.pth'
    mobilenet_model.load_state_dict(torch.load(mobilenet_model_path, map_location=device))
    mobilenet_model.eval()

    FAST_unet_model_path = "Lite_Unet_custom_v1_E200.pth"
    FAST_unet_model = load_model(FAST_unet_model_path, device)

    # 用於記錄每個物件的 mobilenet 分類結果，key 為 track_id
    track_results = {}

    ID = 1
    start_time = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        print(f"處理幀數: {frame_idx}")

        # 使用 FAST_unet 模型取得分割 mask
        #第一步
        _, _, mask_img = Unet_predict_frame(FAST_unet_model, frame, device)
        mask_img = cv2.resize(mask_img, (frame.shape[1], frame.shape[0]))
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            frame = frame[y:y + h, x:x + w]

        # YOLO 偵測物件
        #第二步
        yolo_results = yolo_model.track(source=frame, conf=0.1, tracker="bytetrack.yaml", persist=True)

        for result in yolo_results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box_values = boxes.data[i].tolist()
                if len(box_values) == 7:
                    x1, y1, x2, y2, track_id, conf, cls_id = box_values
                else:
                    continue

                if track_id is True:
                    track_id = ID
                # 轉換座標
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                # 如果該 track_id 尚未進行分類，則進行 mobilenet 分類，否則直接使用記錄的結果
                if track_id not in track_results:
                    cropped_img = frame[y1:y2, x1:x2]
                    mobilenet_predicted_class, mobilenet_confidence = predict_mobilenet(
                        mobilenet_model, cropped_img, transform, device)
                    track_results[track_id] = (mobilenet_predicted_class, mobilenet_confidence)
                else:
                    mobilenet_predicted_class, mobilenet_confidence = track_results[track_id]

                cls_name = cls_map.get(int(cls_id), "unknown")
                # 這裡以 mobilenet 的分類結果對應損壞嚴重度
                color = severity_colors.get(int(mobilenet_predicted_class), (255, 255, 255))

                # 繪製偵測框與文字
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                mobilenet_class_name = mobilenet_class_dict[mobilenet_predicted_class]
                label = f'ID:{int(track_id)}-{cls_name}-{mobilenet_class_name}'
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_x = x1 + (x2 - x1 - label_width) // 2
                label_y = y1 - 15
                font_scale = 0.5
                thickness = 1
                cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (0, 0, 0), thickness + 2)
                cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (255, 255, 255), thickness)

        # 儲存處理後的每一幀
        out_file = os.path.join(output_path, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(out_file, frame)
        # cv2.imshow("Result", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()
    processing_time = time.time() - start_time
    print("所有幀處理完成！總處理時間: {:.2f} 秒".format(processing_time))
