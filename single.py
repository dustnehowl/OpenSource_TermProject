import cv2
import mediapipe as mp
import numpy as np
import pygame, random, sys, os
from pygame.locals import *


######################################################## pygame 필요한것들 #211126
#########################################################################
pygame.init()

POINTER_SIZE = 20
pointerImage = pygame.image.load("pointer.png")
pointerImage = pygame.transform.scale(pointerImage, (POINTER_SIZE,POINTER_SIZE))

BACK_SIZE = 35
backImage = pygame.image.load("back.png")
backImage = pygame.transform.scale(backImage, (BACK_SIZE,BACK_SIZE))

GAME_MODE = {1: 'easy', 2:'normal', 3:'hard'} # 게임 난이도 고를 수 있도록
START_GAME = 0

total_target = 0
targets = []

idx = 15

jointpos = [0,0]

frameNum = 0
TARGET_SIZE = 80

WINDOW_SIZE_WIDTH = 500
WINDOW_SIZE_HEIGHT = 500

BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)

aim_color = {6 : BLUE, 5 : GREEN, 9 : RED, 1: RED, 0 : BLUE, 3 : RED}

FONT = pygame.font.SysFont(None, 48)
FONT2 = pygame.font.SysFont(None, 30)

windowSurface = pygame.display.set_mode((WINDOW_SIZE_WIDTH,WINDOW_SIZE_HEIGHT))
pygame.display.set_caption("Hand_Aim")

def drawText(text, surface, x, y, font = FONT, color = RED):
    textObject = font.render(text, 1, color)
    textRect = textObject.get_rect()
    textRect.topleft = (x,y)
    surface.blit(textObject, textRect)
#########################################################################
#########################################################################


# draw Target 211130
def drawTarget(fN):

    if fN < 4:
        targetImage = pygame.image.load("0.png")
    elif fN < 8:
        targetImage = pygame.image.load("1.png")
    elif fN < 12:
        targetImage = pygame.image.load("2.png")
    elif fN < 16:
        targetImage = pygame.image.load("3.png")
    elif fN < 20:
        targetImage = pygame.image.load("4.png")
    elif fN < 24:
        targetImage = pygame.image.load("5.png")

    for target in targets: 
        targetImage = pygame.transform.scale(targetImage, (TARGET_SIZE,TARGET_SIZE))
        windowSurface.blit(targetImage, target)

    return (fN + 1) % 24

def makeTarget(difficulty):
    global TARGET_SIZE
    if difficulty == 'easy':
        TARGET_SIZE = 90
    elif difficulty == 'normal':
        TARGET_SIZE = 70
    else:
        TARGET_SIZE = 50

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

buffer = 0

# 총쏘는 걸 감지하는 bang함수
def bang():
    global buffer
    if (buffer == 9 or buffer == 1 or buffer == 3) and (idx == 6 or idx == 0):
        return True
    else:
        buffer = idx
        return False


max_num_hands = 1
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}

# 특정한 제스쳐들만 사용하므로 인식률을 높이기 위해 다른 모양들도 주먹,가위에 추가해줌 #211126
rps_gesture = {6:'rock', 5:'paper', 9:'scissors', 1:'scissors', 0: 'rock', 3:'scissors'}

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

    windowSurface.fill(WHITE)
    # px = random.randrange(100,200)
    # py = random.randrange(100,200)
    # randpos = [px,py]

    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#####################################target######################################  #211130 수정
    
    if START_GAME > 0:
        windowSurface.blit(backImage, [465, 465])
        makeTarget(GAME_MODE[START_GAME])
        if total_target == 0:
            while total_target < START_GAME * 2:        
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

#################################################################################


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

            # 제스쳐들이 어떤 숫자로 인식하는지 모두 검사하기 위해 key값을 출력
            # drawText(str(idx), windowSurface, 230, 230, FONT , BLACK)

            #############create aim############### #211126
            px = joint[14][0] * (WINDOW_SIZE_WIDTH + 100) + 0
            py = joint[14][1] * (WINDOW_SIZE_HEIGHT + 100) + 0
            jointpos = [px,py]
            if START_GAME > 0:
                if idx in aim_color.keys():
                    pygame.draw.circle(windowSurface, aim_color[idx], jointpos,5)

                    pygame.draw.circle(windowSurface, BLACK, (px,py),
                            20, 2)

                    pygame.draw.line(windowSurface, BLACK, (px, py + 20),
                            (px, py - 20), 2)
                    pygame.draw.line(windowSurface, BLACK, (px + 20, py),
                            (px - 20, py), 2) 
            else:
                windowSurface.blit(pointerImage, jointpos)
            #######################################

            # Other gestures
            # cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(125,125,125), thickness=2)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    #cv2.imshow('Game', img)


    pygame.display.update()

    #나가기 추가
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()


    if cv2.waitKey(1) == ord('q'):
        break


