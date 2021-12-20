# Eagle Hunt
Eagle Hunt는 openCV를 이용한 제스쳐 인식 게임입니다.

![그림1](https://user-images.githubusercontent.com/39877181/146775230-7917e86a-74ba-46fa-9873-2c5c4320156d.png)

## 주요 특징

cam이 있는 컴퓨터에서 작동합니다. 손의 위치에 따라 에임이 생성되며 총을 쏘는 모션을 통해 ui에 상호작용 할 수 있습니다.

![그림2](https://user-images.githubusercontent.com/39877181/146775246-65a79a1c-61af-4071-a104-5efdcb74adaf.png)

## 시작하기

### 환경 설정

캠을 연결합니다.

### pip install

``` pip install opencv
or
pip install opencv-python
```

```
pip install mediapipe
```

```
pip install numpy
```

```
pip install pygame
```

## 어플리케이션 시작

필요한 모듈들을 pip를 통해 모두 다운로드 받았다면 다음 코드를 실행하여 Eagle Hunt 를 실행합니다.

```
python single.py
```


## 게임 설명

Eagle Hunt 는 3가지의 난이도가 있습니다. 표적의 크기, 점수는 난의도의 영향을 받습니다.

손동작은 3가지로 나뉩니다.
1. 대기(빨간색 에임) __엄지와 검지, 중지를 편 상대로 나머지를 굽혀 총모양을 만듭니다.__
![그림3](https://user-images.githubusercontent.com/39877181/146775265-468d6b10-dd06-4261-b195-4c5a083212bd.png)
2. 발사(파란색 에임) __대기 상태에서 중지와 검지를 함께 굽혀 총을 발사하는 모양을 만듭니다.__
![그림4](https://user-images.githubusercontent.com/39877181/146775278-799520c9-a61d-4100-8756-b1096b5f8202.png)
3. 안전모드(초록색 에임) __대기와 발사 손모양이 아닌 모든 손모양 입니다.__
![그림 5](https://user-images.githubusercontent.com/39877181/146775283-6ee85c5f-f334-4d27-9bab-c57d13a660b7.png)

### 게임 시작

1. cam에 손을 비추어 에임(핑크색 포인터)이 손을 잘 따라오는지 확인합니다.
2. Start! 버튼에 '발사' 동작을 취해 난이도 선택화면으로 넘어갑니다.
3. 난이도를 선택 후 표적들을 '발사' 합니다.

## 개발환경 및 실행환경

Python 3.9 (Window 10, MAC OS), Powershell 사용
