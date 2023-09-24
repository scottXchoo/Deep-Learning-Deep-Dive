# LeNet-5
LeNet은 CNN을 처음으로 개발한 얀 르쿤(Yann Lecun) 연구팀이 1998년에 개발한 CNN 알고리즘의 이름이다. original 논문 제목은 "Gradient-based learning applied to document recognition"이다.

<img width="1000" alt="image" src="https://github.com/scottXchoo/Deep_Learning_Deep_Dive/assets/107841492/9e56a295-5e6e-46f0-8e57-b2f538ccc6e3">

## 네트워크 구조
LeNet-5는 input, 3개의 convolutional layer(C1, C3, C5), 2개의 Subsampling layer(S2, S4), 1층의 full-connected 레이어(F6), 아웃풋 레이어로 구성되어 있다.

1. C1 레이어 : 입력 영상(32x32 사이즈의 이미지)을 6개의 5x5 필터와 컨볼루션 연산을 해준다. 그 결과 **6장의 28x28 feature map**을 얻게 된다.
2. S2 레이어 : 6장의 28x28 feature map에 대해 subsampling을 진행한다. 2x2 필터를 stride 2로 설정해서 진행했기에 결과적으로 28x28 사이즈의 feature map이 **14x14 사이즈의 feature map으로 축소**된다. 여기서 사용하는 subsampling 방법은 평균 풀링(average pooling)이다.
3. C3 레이어 : 6장의 14x14 feature map에 컨볼루션 연산을 수행해 **16장의 10x10 feature map**을 산출해낸다.
4. S4 레이어 : 16장의 10x10 feature map에 대해서 subsampling을 진행해 **16장의 5x5 feature map**으로 축소시킨다.
5. C5 레이어 : 16장의 5x5 feature map을 120개 5x5x16 사이즈의 필터와 컨볼루션 해준다. 결과적으로 **120개 1x1 feature map**을 얻게 된다.
6. F6 레이어 : 84개의 유닛을 가진 feedforward 신경망이다. C5의 결과를 84개의 유닛에 연결시킨다.
7. 아웃풋 레이어 : 10개의 Euclidean radial basis function(RBF) 유닛들로 구성되어있다. 84개 유닛으로부터 인풋을 받는다. 최종적으로 이미지가 속한 클래스를 알려준다.

## 레퍼런스
[1] https://bskyvision.com/418
