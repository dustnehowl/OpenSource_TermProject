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
