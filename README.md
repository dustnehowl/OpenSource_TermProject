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
