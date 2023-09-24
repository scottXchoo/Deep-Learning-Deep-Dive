# Fully Convolutional Network (FCN; 완전 합성곱 계층)

## 정의
- FCN은 이미지 분류에서 우수한 성능을 보인 CNN 기반 모델(AlexNet, VGG16, GoogLeNet)을 Segmantic Segmentation Task를 수행할 수 있도록 변형시킨 모델이다.
- 이후 나온 Semantic Segmentation 방법은 대부분 FCN의 아이디어를 기반으로 했다.

## 핵심 아이디어
이미지 분류는 이미지 내의 모든 픽셀에서 Feature를 추출(Extraction)하고, 추출한 Feature들을 분류기(Classifier)에 넣어 입력 이미지(Total)의 Class를 예측하는 구조로 만들어져 있다.

이미지 분할은 이미지 분류에서 좀 더 나아가서 이미지(Total)의 Class를 예측하는 것이 아니라 이미지를 이루는 모든 픽셀들의 Class를 예측하는 문제로 생각해볼 수 있다.

FCN은 기존 이미지 분류에서 쓰인 네트워크를 훈련된 상태에서(pretrain) Feature Extraction 레이어는 그대로 재활용하여 Feature를 추출하고 FC 레이어를 버리고 1X1 Conv 그리고 Up-sampling(Transpose Covolution)로 변경하여(Fine-Tuning) 픽셀 클래스 분류와 입력이미지와 같은 사이즈 회복을 하도록 네트워크가 구성되었다.

<img width="600" alt="image" src="https://github.com/scottXchoo/Deep_Learning_Deep_Dive/assets/107841492/10af52ee-8467-409d-b75a-db5414c1ce5a">

### FCN의 구조
1. Convolution Layer를 통해 Feature 추출
2. 1x1 Convolution Layer를 이용해 feature map의 채널수를 데이터셋 객체의 개수와 동일하게 변경 (Class Presence Heat Map 추출)
3. Up-sampling : 낮은 해상도의 Heat Map을 Upsampling (= Transposed Convolution) 한 뒤, 입력 이미지와 같은 크기의 Map 생성
4. 최종 피처 맵과 라벨 피처맵의 차이를 이용하여 네트워크 학습

## 문제점
<img width="600" alt="image" src="https://github.com/scottXchoo/Deep_Learning_Deep_Dive/assets/107841492/bf0a46b5-7603-4b0c-afbf-ee2b6b47a667">

VGG16에서 입력 이미지의 크기가 224x224 인 경우 5개의 convolution block으르 통과하면 feature map의 크기가 7x7이 된다. 마찬가지로 FCN에서도 기존 입력 이미지 (크기 H x W)가 5개의 convolution block을 통과하면 H/32 x W/32 크기의 feature map을 얻게 된다.

Feature map의 한 픽셀은 입력 이미지의 32x32 pixel을 대표하게 된다. 이처럼 feature map은 낮은 해상도를 가져 **입력 이미지의 위치 정보를 '대략적으로만' 가지고** 있게 된다.

문제는 3번 Up-sampling 과정에서 발생한다. 입력 이미지 위치 정보를 '대략적으로' 가지고 있는 feature map을 Up-sampling하여 얻은 segmentation map은 기존 입력 이미지와 비교했을 때 뭉뚱그려져 있고 디테일하지 못하다.
<img width="600" alt="image" src="https://github.com/scottXchoo/Deep_Learning_Deep_Dive/assets/107841492/0d830e1d-f37a-4bff-aee2-6fccbd3f9a0b">

### 해결방안 : Up-sampling by Transposed convolution
뭉그러짐 문제를 해결하기 위해 먼저 드는 생각은 Down-sampling을 하지 않아 feature map이 작아지지 않도록 하는 것이다.

그러나 Down-sampling을 통해 feature map의 사이즈를 줄이지 않는다면, **연산량이 급격히 늘어나 학습에 필요한 시간 및 비용이 너무 커지게 된다.** 때문에 Down-sampling은 필수적이고 이를 해결하기 위해 효과적인 Up-sampling을 위한 방법이 여러 개 고안된다.

FCN에서는 Transposed convolution을 이용하여 Up-sampling을 진행한다.

## 레퍼런스
[1] [위키독스](https://wikidocs.net/147359)
