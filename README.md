# Open-Source Term Project

참고한곳 :   
https://github.com/spmallick/learnopencv

### 20211125
----------------------

캠키는 파이썬 코드

![image](https://user-images.githubusercontent.com/39877181/143380941-11fd1985-ae12-46fc-82cf-9069538889c3.png)
```
import cv2

def webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        cv2.imshow('webcam', img)
        
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()

def main():
    webcam(mirror=True)

if __name__ == '__main__':
    main()
```

얼굴 인식

![image](https://user-images.githubusercontent.com/39877181/143382813-d9be22a3-3f7e-469e-bf70-6174a13b3ecf.png)

```
import numpy as np
import cv2
 
detector = cv2.CascadeClassifier('C:\\Users\\CV_LAB\\OpenSource_TermProject\\test\\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
 
while (True):
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
 
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
```



###20211126
----------------------------
가위바위보에서 바위를 엄지만 핀 상태를 인식하도록 수정
=> 가위에서 바위로 가는 것을 방아쇠를 당기는 것으로 표현 가능

가위 (총을 쏘기 전 모습)
![image](https://user-images.githubusercontent.com/39877181/143528965-c4f51fe2-4d68-48f1-97cb-7ebf58db1b23.png)

바위(방아쇠를 당기는 모습)
![image](https://user-images.githubusercontent.com/39877181/143529010-b7540ab1-5775-49ca-94ca-d0e87417a360.png)

가위 -> 바위 (총을 쏘는 행위)

https://user-images.githubusercontent.com/39877181/143529127-ae3c52f7-7919-4a91-b5d8-ba7979a05b42.mp4

코드 수정
```
import cv2
import mediapipe as mp
import numpy as np

max_num_hands = 1
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}
rps_gesture = {6:'rock', 5:'paper', 9:'scissors'}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition model
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            # Draw gesture result
            if idx in rps_gesture.keys():
                cv2.putText(img, text=rps_gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(125,125,125), thickness=2)

            # Other gestures
            # cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(125,125,125), thickness=2)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Game', img)
    if cv2.waitKey(1) == ord('q'):
        break
```


