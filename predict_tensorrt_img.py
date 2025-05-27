from ultralytics import YOLO
import os
import cv2
from mobilenetv3_model import mobilenetv3_large_SimAM
import torch
from torchvision import transforms
from PIL import Image
import time
import base64

# 影片輸入與輸出路徑
input_video_path = r'D:\road_damage\Produce_old.avi'

output_video_path = r'./output/output_video2.mp4'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 類別對應表
cls_map = {
    0: 'transverse cracks',
    1: 'longitudinal cracks',
    2: 'alligator cracks',
    3: 'potholes',
    4: 'manhole',
    5: 'speed bump',
    6: 'expansion joint'
}

# 類別顏色 (BGR 格式)
cls_colors = {
    0: (255, 0, 0),  # 藍色 transverse cracks
    1: (0, 255, 0),  # 綠色 longitudinal cracks
    2: (0, 0, 255),  # 紅色 alligator cracks
    3: (255, 255, 0),  # 青色 potholes
    4: (255, 0, 255),  # 粉色 manhole
    5: (0, 255, 255),  # 黃色 speed bump
    6: (128, 0, 128)  # 紫色 expansion joint
}

mobilenet_class_dict = {
    0: "A",
    1: "B",
    2: "C",
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict_mobilenet(model, image, transform, device):
    image = Image.fromarray(image).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    outputs = model(image)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted = probabilities.max(1)
    return predicted.item(), confidence.item()


if __name__ == '__main__':
    # 加載 YOLOv11 模型
    model = YOLO("best.engine", task='detect')
    mobilenet_model = mobilenetv3_large_SimAM(num_classes=3, width_mult=0.5).to(device)
    model_path = './best_mobilenet_epoch_255.pth'
    mobilenet_model.load_state_dict(torch.load(model_path))
    mobilenet_model.eval()
    ID = 1

    # 讀取影片
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    start_time = time.time()
    detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 進行 YOLOv11 偵測 + BoT-SORT 追蹤
        results = model.track(
            source=frame,
            conf=0.2,
            tracker="bytetrack.yaml",
            persist=True
        )

        # 解析 YOLO 追蹤結果
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box_values = boxes.data[i].tolist()
                if len(box_values) == 7:
                    x1, y1, x2, y2, track_id, conf, cls_id = box_values
                else:
                    continue
                if track_id is True:
                    track_id = ID
                ID += 1
                print(ID)
                cropped_img = frame[int(y1):int(y2), int(x1):int(x2)]
                mobilenet_predicted_class, mobilenet_confidence = predict_mobilenet(mobilenet_model, cropped_img,
                                                                                    transform, device)
                cls_name = cls_map.get(int(cls_id), "unknown")
                color = cls_colors.get(int(cls_id), (255, 255, 255))

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'ID {track_id}-{mobilenet_predicted_class}-{cls_name}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                detections.append({
                    "ID": int(track_id),
                    "bbox": [x1, y1, x2, y2],
                    "class": cls_name,
                    "confidence": mobilenet_confidence
                })
        print(detections)
        out.write(frame)

    cap.release()
    out.release()

    processing_time = time.time() - start_time

    final_result = {
        "output_video": output_video_path,
        "detections": detections,
        "message": "偵測完成",
        "processing_time": processing_time,
        "status": "success"
    }
    print(final_result)
