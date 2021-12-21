import cv2
import mediapipe as mp
import numpy as np
import pygame, random, sys, os
from pygame.locals import *


######################################################## pygame 필요한것들 #211126
#########################################################################
pygame.init()

score = 0
finalScore = 0

POINTER_SIZE = 20
pointerImage = pygame.image.load("images/pointer.png")
pointerImage = pygame.transform.scale(pointerImage, (POINTER_SIZE,POINTER_SIZE))

BACK_SIZE = 35
backImage = pygame.image.load("images/back.png")
backImage = pygame.transform.scale(backImage, (BACK_SIZE,BACK_SIZE))

bg = pygame.image.load("images/background.png")
bg_game = pygame.image.load("images/game_back.png")

GAME_MODE = {1: ' Easy', 2:'Normal', 3:' Hard', 4: 'ScoreBoard'} # 게임 난이도 고를 수 있도록
START_GAME = 0
MODE = 0
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

FONT = pygame.font.Font('font/Maplestory Bold.ttf', 40)
UIFONT = pygame.font.Font('font/Maplestory Bold.ttf', 25)
UIFONT2 = pygame.font.Font('font/Maplestory Bold.ttf', 28)
DIFFIFONT = pygame.font.Font('font/Maplestory Bold.ttf', 32)
DIFFIFONT2 = pygame.font.Font('font/Maplestory Bold.ttf', 35)

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
        targetImage = pygame.image.load("images/0.png")
    elif fN < 8:
        targetImage = pygame.image.load("images/1.png")
    elif fN < 12:
        targetImage = pygame.image.load("images/2.png")
    elif fN < 16:
        targetImage = pygame.image.load("images/3.png")
    elif fN < 20:
        targetImage = pygame.image.load("images/4.png")
    elif fN < 24:
        targetImage = pygame.image.load("images/5.png")

    for target in targets: 
        targetImage = pygame.transform.scale(targetImage, (TARGET_SIZE,TARGET_SIZE))
        windowSurface.blit(targetImage, target)

    return (fN + 1) % 24

def makeTarget(difficulty):
    global TARGET_SIZE
    if difficulty == ' Easy':
        TARGET_SIZE = 90
    elif difficulty == 'Normal':
        TARGET_SIZE = 70
    else:
        TARGET_SIZE = 50

def Menu():
    windowSurface.fill(WHITE)
    global MODE
    bg_trans = pygame.transform.scale(bg, (WINDOW_SIZE_WIDTH,WINDOW_SIZE_HEIGHT))
    bg_trans.set_alpha(100)
    windowSurface.blit(bg_trans, [0,0])
    if MODE == 0:
        menuUI = []
        menuUI.append(pygame.Rect(320,300,100,40))
        menuUI.append(pygame.Rect(320,350,100,40))
        menuUI.append(pygame.Rect(320,400,100,40))

        # for rect in menuUI: #굳이 그릴필요 없음
        #     pygame.draw.rect(windowSurface, WHITE, rect)

        drawText("Eagle Hunt", windowSurface, 150, 130, FONT , BLACK) # 게임 이름

        uiname = ["Start!", "Creator", "Exit"]
        ui_y = [310,360,410]
        for idx in range(0,3):
            if menuUI[idx].collidepoint(jointpos):
                drawText(uiname[idx], windowSurface, 357,  ui_y[idx]-3, UIFONT2 , BLACK)
                if bang() == True:
                    MODE = idx+1
                    print(MODE)
            else:
                drawText(uiname[idx], windowSurface, 360, ui_y[idx], UIFONT , BLACK)
    if MODE == 1:
        SetDiffi()
    elif MODE == 2:
        Creator()
    elif MODE == 3:
        sys.exit()

def Make_back():
    backRect = pygame.Rect(0, 465, BACK_SIZE, BACK_SIZE)
    windowSurface.blit(backImage, [0, 465])
    global MODE
    if bang() == True:
        if backRect.collidepoint(jointpos):
            MODE = 0

def Creator(): # 
    drawText("18101192 김동경", windowSurface, 100,100, FONT,BLACK)
    drawText("18101201 김연수", windowSurface, 100,150, FONT,BLACK)
    Make_back()

def SetDiffi():
    difficultyRects = []
    difficultyRects.append(pygame.Rect(195, 95, 100, 40)) # difficultyRects.append(pygame.Rect(5, 450, random.randrange(100,200), 100))
    difficultyRects.append(pygame.Rect(195, 205, 120, 40))
    difficultyRects.append(pygame.Rect(195, 330, 100, 40))
    # for rect in difficultyRects:
    #    pygame.draw.rect(windowSurface, GREEN, rect)
    global START_GAME
    diffi_y = [95,210,335] # 난이도에 에임올려져 있을 시 강조 효과 추가
    for idx in range(1,4):
        if difficultyRects[idx-1].collidepoint(jointpos):
            drawText(GAME_MODE[idx], windowSurface, 197, diffi_y[idx-1]-3, DIFFIFONT2 , BLACK)
            if bang() == True:
                START_GAME = idx
        else:
            drawText(GAME_MODE[idx], windowSurface, 200, diffi_y[idx-1], DIFFIFONT , BLACK)

    Make_back()

buffer = 0
# 총쏘는 걸 감지하는 bang함수
def bang():
    global buffer
    if (buffer == 9 or buffer == 1 or buffer == 3) and (idx == 6 or idx == 0):
        return True
    else:
        buffer = idx
        return False


def game(): #211130 수정
    
    bg_trans = pygame.transform.scale(bg_game, (WINDOW_SIZE_WIDTH,WINDOW_SIZE_HEIGHT))
    bg_trans.set_alpha(100)
    windowSurface.blit(bg_trans, [0,0])
    
    global START_GAME,total_target,score,frameNum
    if START_GAME == 4:
        ScoreBoard()
    elif START_GAME > 0:
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
        drawText("SCORE : ", windowSurface, 10,10, UIFONT , BLACK)
        drawText(str(score), windowSurface, 120,10, UIFONT , BLACK)
        


        if bang() == True:
            if backRect.collidepoint(jointpos):
                global finalScore
                finalScore = score
                targets.clear()
                total_target = 0
                score = 0
                START_GAME = 4
            for target in targets:
                if target.collidepoint(jointpos):
                    pygame.mixer.music.load('music/bang.mp3')
                    pygame.mixer.music.play(1,0.0)
                    targets.remove(target)
                    score = score + START_GAME
                    total_target = total_target - 1

        frameNum = drawTarget(frameNum)
    
    else:
        #게임 시작이 아니면 메뉴 화면
        Menu()

def ScoreBoard () :
    bg_trans = pygame.transform.scale(bg_game, (WINDOW_SIZE_WIDTH,WINDOW_SIZE_HEIGHT))
    bg_trans.set_alpha(100)
    windowSurface.blit(bg_trans, [0,0])
    drawText("SCORE : ", windowSurface, 190,220, UIFONT2 , BLACK)
    drawText(str(finalScore), windowSurface, 300,220, UIFONT2 , BLACK)
    backRect = pygame.Rect(190, 265, 150, 40)
    
    if backRect.collidepoint(jointpos):
        drawText("Main Menu", windowSurface, 187, 262, UIFONT2, BLACK)
        global START_GAME
        if bang() == True:
            START_GAME = 0
    else :
        drawText("Main Menu", windowSurface, 190, 265, UIFONT, BLACK)        

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

    game()
    

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
            px = joint[14][0] * (WINDOW_SIZE_WIDTH) + 0
            py = joint[14][1] * (WINDOW_SIZE_HEIGHT) + 0
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

    # cv2.imshow('Game', img)


    pygame.display.update()

    #나가기 추가
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()


    if cv2.waitKey(1) == ord('q'):
        break
