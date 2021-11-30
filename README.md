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

코드 추가 (추후에 캡슐화 해야함)

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

### 20211129

#### 에임 그림 추가
```
if idx in aim_color.keys():
                pygame.draw.circle(windowSurface, aim_color[idx], jointpos,5)

                pygame.draw.circle(windowSurface, BLACK, (px,py),
                           20, 2)

                pygame.draw.line(windowSurface, BLACK, (px, py + 20),
                        (px, py - 20), 2)
                pygame.draw.line(windowSurface, BLACK, (px + 20, py),
                        (px - 20, py), 2) 
```

### 20211130

#### 타겟 gif처럼 보이게 설정

##### 전역 변수 추가
```
frameNum = 0
TARGET_SIZE = 80
```

##### 프레임별로 추가
```
while total_target < 2:        
        targets.append((random.randrange(0,WINDOW_SIZE_WIDTH-TARGET_SIZE), random.randrange(0,WINDOW_SIZE_HEIGHT-TARGET_SIZE)))
        total_target += 1

    if frameNum < 24 :
        frameNum = frameNum+1
    else:
        frameNum = 0

    if frameNum < 4:
        targetImage = pygame.image.load("0.png")
    elif frameNum < 8:
        targetImage = pygame.image.load("1.png")
    elif frameNum < 12:
        targetImage = pygame.image.load("2.png")
    elif frameNum < 16:
        targetImage = pygame.image.load("3.png")
    elif frameNum < 20:
        targetImage = pygame.image.load("4.png")
    elif frameNum < 24:
        targetImage = pygame.image.load("5.png")

    for target in targets: 
        targetImage = pygame.transform.scale(targetImage, (TARGET_SIZE,TARGET_SIZE))
        windowSurface.blit(targetImage, target)
```

#### 게임시작전 메뉴 화면
```
def menuSetup():

    windowSurface.fill(WHITE)
    difficultyRects = []
    difficultyRects.append(pygame.Rect(50, 180, 100, 50)) # difficultyRects.append(pygame.Rect(5, 450, random.randrange(100,200), 100))
    difficultyRects.append(pygame.Rect(200, 180, 100, 50))
    difficultyRects.append(pygame.Rect(350, 180, 100, 50))
    for rect in difficultyRects:
        pygame.draw.rect(windowSurface, GREEN, rect)
    drawText("Easy", windowSurface, 75, 195, FONT2 , BLACK)
    drawText("Medium", windowSurface, 210, 195,FONT2 , BLACK)
    drawText("Hard", windowSurface, 375, 195 ,FONT2 , BLACK)
    global START_GAME
    if bang() == True:
        if difficultyRects[0].collidepoint(jointpos):
            START_GAME = 1
        if difficultyRects[1].collidepoint(jointpos):
            START_GAME = 2
        if difficultyRects[2].collidepoint(jointpos):
            START_GAME = 3

def Menu():
    menuSetup()
```

#### 총쏘는거 감지하는 bang 함수
```
def bang():
    global buffer
    if (buffer == 9 or buffer == 1 or buffer == 3) and (idx == 6 or idx == 0):
        return True
    else:
        buffer = idx
        return False
```

#### 메뉴화면 만듦에 따라 타켓 코드 수정 + 뒤로가기 버튼
```
if START_GAME > 0:
        windowSurface.blit(backImage, [465, 465]) # 뒤로가기 버튼
        makeTarget(GAME_MODE[START_GAME])
        if total_target == 0:
            while total_target < START_GAME * 2:       # 난이도에 따라 target 생성 수 변화 
                #targets.append((random.randrange(0,WINDOW_SIZE_WIDTH-TARGET_SIZE)
                #, random.randrange(0,WINDOW_SIZE_HEIGHT-TARGET_SIZE)))
                targets.append(pygame.Rect(random.randrange(0,WINDOW_SIZE_WIDTH - TARGET_SIZE),
                random.randrange(0,WINDOW_SIZE_WIDTH - TARGET_SIZE),TARGET_SIZE,TARGET_SIZE))
                total_target = total_target + 1

        backRect = pygame.Rect(465, 465, BACK_SIZE, BACK_SIZE)

        if bang() == True:
            if backRect.collidepoint(jointpos):
                targets.clear()
                total_target = 0
                START_GAME = 0
            for target in targets:
                if target.collidepoint(jointpos):
                    targets.remove(target)
                    total_target = total_target - 1

        frameNum = drawTarget(frameNum)
    
    else:
        #게임 시작이 아니면 메뉴 화면 만들어야함
        Menu()
```

#### 211130 결과


https://user-images.githubusercontent.com/39877181/144035644-32c8fd6f-f8f3-497b-bc71-118e114a1ac4.mp4

