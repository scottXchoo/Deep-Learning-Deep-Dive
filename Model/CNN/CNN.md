# CNN (합성곱 신경망; Convolutional Neural Networks)
<img width="500" alt="image" src="https://github.com/scottXchoo/Deep_Learning_Deep_Dive/assets/107841492/f5bd4edd-6c5f-4036-a510-3e5c463acdb1">
<img width="500" alt="image" src="https://github.com/scottXchoo/Deep_Learning_Deep_Dive/assets/107841492/4f21196b-7794-41d8-95c4-a2086580b5be">


- 설명
  - '3차원 데이터'를 입력으로 취하고 뉴런을 그와 비슷한 볼륨으로 정렬한다.
  - 각 뉴런은 이전 계층에서 이웃한 영역에 속한 일부 요소에만 접근하는데, 이 영역을 뉴런의 수용 영역(receptive field)이라고 한다.
  - 데이터를 특징 맵 (feature map)이라고 부르기도 한다.
- 구조
  - convolutional layer (합성곱층)
    - 정의
      - 입력 이미지의 모든 픽셀에 연결되는 것이 아니라, 합성곱층 뉴런의 수용영역(receptive field) 안에 있는 픽셀에만 연결이 된다.
    - 개념
      - filter : 수용영역을 합성곱층에서 필터 혹은 커널이라고 한다. 이 필터가 바로 합성곱층에서의 가중치 파라미터(W)에 해당한다.
      - padding : 합성곱 연산을 수행하기 전, 입력데이터 주변을 특정 값으로 채워 늘리는 것을 말한다. 이를 사용하지 않으면, 데이터의 spartial 크기는 합성곱층을 지날 때마다 작아지므로 가장자리 정보들이 소실되는 문제가 발생한다.
      - stride : 입력 데이터에 필터를 적용할 때, 이동할 간격을 조정하는 것. 즉, 필터가 이동할 간격을 말한다.
  - pooling layer (풀링층)
    - 정의
      - 합성곱층의 출력 데이터를 입력으로 받아서 출력 데이터의 크기를 줄이거나 특정 데이터를 강조하는 용도로 사용된다.
      - 풀링 연산은 쉽게 말해 가로 & 세로 방향의 공간을 줄이는 연산이다.
      - 입력 데이터의 크기가 축소되고 학습하지 않기 때문에 파라미터 수가 줄어들어 **과대적합 (overfitting)이 발생하는 것을 방지해 줍니다.**
    - 종류
      - Max Pooling : 대상 이미지 영역에서 최댓값을 구함
      - Average Pooling : 대상 이미지 영역에서 평균값을 구함
  - fully connected layer (완전 연결 계층)
    - 정의
      - Convolution/Pooling 프로세스의 결과를 취하여 이미지를 정의된 라벨로 분류하는데 사용 (단순한 분류의 예)
      - 2차원의 배열 형태 이미지를 1차원의 평탄화 작업을 통해 이미지를 분류하는데 사용되는 계층이다.
