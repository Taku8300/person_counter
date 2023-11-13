from ultralytics import YOLO
import cv2

def yoloPredict():
    cap = cv2.VideoCapture(0)

    # カウンター初期化
    people_count = 0

    while True:
        _, frame = cap.read()

        frame = cv2.resize(frame, (640, 370))
        cv2.imshow('class_people', frame)

        # YOLOv8による予測
        results = model(frame)

        detected_people = [det for det in results.xyxy[0] if int(det[5]) == 0]

        # 検出された人物の周りに枠を描画
        for det in detected_people:
            x1, y1, x2, y2 = map(int, det[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 人数カウント
        people_count = len(detected_people)

        # 画面に人数表示
        cv2.putText(frame, f'People Count: {people_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Predict the model
        model.predict(frame, save=True, conf=0.5, box=True)

        k = cv2.waitKey(1)

        if k == 27 or k == 13:
            break

    cap.release()

if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8n.pt')
    yoloPredict()
