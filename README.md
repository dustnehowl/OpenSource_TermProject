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

#### 얼굴 인식

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



### 20211126
----------------------------
가위바위보에서 바위를 엄지만 핀 상태를 인식하도록 수정
=> 가위에서 바위로 가는 것을 방아쇠를 당기는 것으로 표현 가능

가위 (총을 쏘기 전 모습)

![image](https://user-images.githubusercontent.com/39877181/143528965-c4f51fe2-4d68-48f1-97cb-7ebf58db1b23.png)

바위(방아쇠를 당기는 모습)

![image](https://user-images.githubusercontent.com/39877181/143529010-b7540ab1-5775-49ca-94ca-d0e87417a360.png)

가위 -> 바위 (총을 쏘는 행위)

https://user-images.githubusercontent.com/39877181/143529127-ae3c52f7-7919-4a91-b5d8-ba7979a05b42.mp4

#### 코드 수정
```
rps_gesture = {6:'rock', 5:'paper', 9:'scissors'}
```

손가락 위치에 에임 가져다 놓기   **방아쇠를 당겨도 에임이 움직이지 않도록 약지의 첫번째 마디를 에임으로 선정함**


![image](https://user-images.githubusercontent.com/39877181/143799128-e48c3edd-4922-4486-8d42-1433de4ff7e0.png)
![image](https://user-images.githubusercontent.com/39877181/143799160-65246cd8-0e8f-417d-9f11-9ebe7151e858.png)

코드 추가 (추후에 캠슐화 해야함)

#### 다른 보드 생성 ( 에임보드)
```
pygame.init()

WINDOW_SIZE_WIDTH = 500
WINDOW_SIZE_HEIGHT = 500

BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)

aim_color = {6 : BLUE, 5 : GREEN, 9 : RED, 1: RED, 0 : BLUE, 3 : RED}

FONT = pygame.font.SysFont(None, 48)
FONT2 = pygame.font.SysFont(None, 20)

windowSurface = pygame.display.set_mode((WINDOW_SIZE_WIDTH,WINDOW_SIZE_HEIGHT))
pygame.display.set_caption("sniper")

def drawText(text, surface, x, y, font = FONT, color = RED):
    textObject = font.render(text, 1, color)
    textRect = textObject.get_rect()
    textRect.topleft = (x,y)
    surface.blit(textObject, textRect)
```

#### 손가락 번호 추출 후 원그리기
```
px = joint[14][0] * WINDOW_SIZE_WIDTH + 0
py = joint[14][1] * WINDOW_SIZE_HEIGHT + 0
jointpos = [px,py]
if idx in aim_color.keys():
    pygame.draw.circle(windowSurface, aim_color[idx], jointpos,10)
```

#### 인식률을 높이기 위해 모든 각도에서 인식할 수 있는 주먹가위보 추가
```
rps_gesture = {6:'rock', 5:'paper', 9:'scissors', 1:'scissors', 0: 'rock', 3:'scissors'}
```
