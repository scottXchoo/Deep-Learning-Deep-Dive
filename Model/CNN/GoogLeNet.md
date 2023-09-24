# GoogLeNet

<img width="1000" alt="image" src="https://github.com/scottXchoo/Deep_Learning_Deep_Dive/assets/107841492/aaaf78d1-e233-4a75-9600-8703724dfbd4">

## 1x1 컨볼루션
1x1 사이즈의 필터로 컨볼루션해준다. 1x1 컨볼루션은 어떤 의미를 갖는 것일까? GoogLeNet에서 1x1 컨볼루션은 feature map의 개수를 줄이는 목적으로 사용된다. feature map 개수가 줄어들면 그만큼 연산량이 줄어든다.

<img width="1000" alt="image" src="https://github.com/scottXchoo/Deep_Learning_Deep_Dive/assets/107841492/e051cad8-9904-4c0f-8880-c17aaafe829e">

#### 1x1 컨볼루션이 없을 때
예를 들어, 480장의 14x14 사이즈의 feature map(14x14x480)이 있다고 가정해보자. 이것을 **"48개의 5x5x480의 필터커널"** 로 컨볼루션을 해주면 **"48장의 14x14의 feature map(14x14x48)"** 이 생성된다. 이때 필요한 연산횟수는 "(14x14x48) X (5x5x480) = 약 112.9M"이 된다.

#### 1x1 컨볼루션이 있을 때
이번에는 480장의 14x14 사이즈의 feature map(14x14x480)을 먼저 **"16개의 1x1x480의 필터커널"** 로 컨볼루션을 해줘 feature map의 개수를 줄여보자. 결과적으로 16장의 14x14의 feature map(14x14x16)이 생성된다. 이것을 48개의 5x5x16의 필터커널로 컨볼루션을 해주면 **"48장의 14x14의 feature map(14x14x48)"** 이 생성된다. 이때 필요한 연산횟수는 "(14x14x16) X (1x1x480) + (14x14x48) X (5x5x16) = 약 5.3M"이다.

<img width="600" alt="image" src="https://github.com/scottXchoo/Deep_Learning_Deep_Dive/assets/107841492/a635d65b-28f1-458a-be42-12f80d698f47">


## Inception

### v1
<img width="1000" alt="image" src="https://github.com/scottXchoo/Deep_Learning_Deep_Dive/assets/107841492/6ccce363-0ece-43a2-8781-dd041f3b69bf">

GoogLeNet은 총 9개의 인셉션 모듈을 포함하고 있다. 이전 층에서 생성된 feature map을 1x1 컨볼루션, 3x3 컨볼루션, 5x5 컨볼루션, 3x3 최대풀링해준 결과로 얻은 feature map들을 모두 함께 쌓아준다.

AlexNet, VGGNet 등의 이전 CNN 모델들은 한 층에서 동일한 사이즈의 필터커널을 이용해서 컨볼루션을 해줬던 것과 차이가 있다. 따라서 조금 더 다양한 종류의 특성이 도출된다. 여기에 1x1 컨볼루션이 포함되었으니 당연히 연산량은 많이 줄어들었을 것이다.

### Global Average Pooling
<img width="1000" alt="image" src="https://github.com/scottXchoo/Deep_Learning_Deep_Dive/assets/107841492/1fb029c2-7ae1-465d-a041-2b8021420b2d">

AlexNet, VGGNet 등에서는 Fully Connected (FC) 층들이 망의 후반부에 연결되어 있다. GoogLeNet은 FC 방식 대신에 Global Average Pooling이란 방식을 사용한다. Global Average Pooling은 전 층에서 산출된 feature map들을 각각 평균낸 것을 쭉 이어서 1차원 벡터로 만들어주는 것이다. 1차원 벡터를 만들어줘야 최종적으로 이미지 분류를 위한 softmax 층을 연결해줄 수 있기 때문이다.

이렇게 해줌으로써 얻는 장점은 **가중치의 개수를 상당히 많이 없애준다는 것** 이다. 만약 FC 방식을 사용한다면 훈련이 필요한 가중치의 개수는 7x7x1024x1024 = 51.3M이지만 Global Average Pooling을 사용하면 가중치가 단 한 개도 필요하지 않다.


### Auxiliary classifier (보조 분류기)
<img width="1000" alt="image" src="https://github.com/scottXchoo/Deep_Learning_Deep_Dive/assets/107841492/6095af4c-6722-49cb-ba81-6939e962860a">

네트워크 깊이가 깊어질수록 vanishing gradient (기울기 소실) 문제를 피하기 어려워진다. 가중치를 훈련하는 과정에 역전파(back propagation)를 주로 활용하는데, 역전파 과정에서 가중치를 업데이트하는데 사용되는 gradient가 점점 작아져서 0이 되어버리는 것이다. 따라서 네트워크 내의 가중치들이 제대로 훈련되지 않는다.

이 문제를 극복하기 위해서 GoogLeNet에서는 네트워크 중간에 두 개의 보조 분류기(auxiliary classifier)를 달아주었다. 이 보조 분류기들은 훈련시에만 활용되고 사용할 때는 제거해준다.

딥러닝 모델의 성능을 높이는 가장 간단한 방법은 모델(네트워크)의 깊이를 깊고 넓게 만들면 된다. 그러나 이 방법에는 두 가지 큰 문제가 존재한다.

1. 모델의 사이즈가 커지면, 학습해야 할 파라미터들이 많아지고 이는 과적합으로 이어질 수 있다.
2. 컴퓨팅 파워(컴퓨터 연산의 한계 및 자원 낭비)의 문제가 있다.

Inception module이 나오게 된 흐름을 생각해보면 다음과 같다.
- Conv filter size가 작다면 -> 위치적 정보는 잘 볼 수 있지만 local region에 너무 집착한다. (나무만 보고 숲은 안본다)
- Conv filter size가 크다면 -> 추상화 정도는 올라가지만 위치 정보는 떨어진다. (숲을 보는데 나무는 안본다)
=> 어차피 이런 trade-off 관계일거면 순차적으로 쓰거나 하지말고 다 같이 한번에 쓰자!

### v2
<img width="600" alt="image" src="https://github.com/scottXchoo/Deep_Learning_Deep_Dive/assets/107841492/6ccc068c-73bc-4ae2-a0bc-be6b13f88184">
기존 GoogLeNet(Inception V1)에서 연산량을 더 줄여보기 위해 기존 Filter를 나누어 사용했다.

기존의 5x5 컨볼루션 레이어를 2개의 3x3 컨볼루션 레이어로 대체해서 파라미터 수를 5x5=25개에서 3x3x2=18개로 줄였고, 7x7 컨볼루션 레이어 역시 3개의 3x3 컨볼루션 레이어로 대체해서 7x7=49개에서 3x3x3=27개로 약 45%나 줄였다.

또한, Inception v2의 구조를 보면 GoogLeNet에서 사용되던 Auxiliary Classifier 2개 중 하나가 사라졌다. 모델의 초반에 있던 Auxiliary Classifier는 쓸모없다는게 밝혀지면서 없애버렸다.

<img width="1000" alt="image" src="https://github.com/scottXchoo/Deep_Learning_Deep_Dive/assets/107841492/c60b76e1-5551-4940-b032-f9e5343f0177">

### v3
V3 모델은 V2를 만들고 여러 파라미터를 수정해보다가 결과가 더 좋게 나온 것을 합친 모델이다.
- Optimizer 변경
  - RMSProp으로 변경
- Label Smoothing
  - target 값을 one-hot encoding을 사용 X => 오버피팅 방지
  - 값이 0인 레이블에 대해서도 아주 작은 값 ee를 배분
  - 정답은 `1 - (n-1) X e1 - (n-1) X e`로 값을 반영하여 사용
- BN-auxiliary
  - 마지막으로 Fully Connected Layer에 Batch Normalization (BN) 적용

### v4
<img width="500" alt="image" src="https://github.com/scottXchoo/Deep_Learning_Deep_Dive/assets/107841492/c2080f66-fadd-42f7-a3d8-b66315fdd7d3">

기존의 Inception 모델들(~V3)은 뛰어난 성능은 인정 받았지만 모델의 구조가 너무 복잡하다는 평가를 받았다. 이 복잡성 때문에 이미지 분류 대회에서 좋은 성적을 거두던 Inception 계열의 모델들보다 VGGNet이 더 흔하게 사용되었다.

이에 2017년, Inceptio v3보다 단순하고 획일화된 구조와 더 많은 Inception module을 사용한 Inception v4가 등장했다.
<img width="300" alt="image" src="https://github.com/scottXchoo/Deep_Learning_Deep_Dive/assets/107841492/6f2aebfd-9cec-4331-a620-d526e400a13f">


## 레퍼런스
[1] https://bskyvision.com/539

[2] [Inception v1,v2,v3,v4는 무엇이 다른가 (+ CNN의 역사)](https://hyunsooworld.tistory.com/entry/Inception-v1v2v3v4%EB%8A%94-%EB%AC%B4%EC%97%87%EC%9D%B4-%EB%8B%A4%EB%A5%B8%EA%B0%80-CNN%EC%9D%98-%EC%97%AD%EC%82%AC)
