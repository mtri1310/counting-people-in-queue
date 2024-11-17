import cv2
import pandas as pd
import torch
import pathlib
import cvzone
import numpy as np
from pathlib import Path

# Fix lỗi PosixPath trên Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


# Tải mô hình YOLOv5 đã huấn luyện
model_path = "best.pt"  # Đường dẫn tới mô hình YOLOv5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Sử dụng GPU nếu có
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, device=device)


# Hàm phát hiện và đếm người
def detect_and_count_human(frame, region_coords):
    # Chuyển khung hình sang định dạng PyTorch
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # Lấy kết quả phát hiện

    human_count = 0
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        class_id = int(class_id)

        # Kiểm tra nếu đối tượng là 'person' (class_id = 0)
        if class_id == 0:
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Kiểm tra nếu đối tượng nằm trong vùng xác định
            if (region_coords[0] < center_x < region_coords[2] and
                    region_coords[1] < center_y < region_coords[3]):
                human_count += 1

            # Vẽ bounding box và nhãn
            cvzone.cornerRect(frame, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)), l=20, t=2, colorR=(0, 255, 0))
            cv2.putText(frame, f"Person {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

    return frame, human_count


# Khởi tạo video capture
cap = cv2.VideoCapture("queue.mp4")  # Đường dẫn video
# region_coords = (200, 150, 400, 300)  # Xác định vùng giữa màn hình (x1, y1, x2, y2)
region_coords = (100, 50, 300, 200)  # Vùng dịch lên trên và sang trái một chút


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Xử lý phát hiện và đếm
    frame, human_count = detect_and_count_human(frame, region_coords)

    # Vẽ vùng giữa màn hình
    cv2.rectangle(frame, (region_coords[0], region_coords[1]), (region_coords[2], region_coords[3]), (0, 0, 255), 2)
    cv2.putText(frame, f"Count: {human_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Hiển thị khung hình
    cv2.imshow("Human Detection", frame)

    # Thoát khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
