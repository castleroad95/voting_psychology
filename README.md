# voting_psychology
투표 심리 성향 예측 

### 개요
데이콘에서 진행하는 심리 성향 예측 AI 경진대회의 train과 test set을 사용하였으며, 설문의 답(feature)에 따라 분류하여 투표 심리 성향(target)을 예측한다.


### 처리순서도
데이터 전처리 -> 모델링(modeling) -> 심리 성향 예측(predict) 후 정확도 확인


### 데이터 전처리(preprocessing)
- drop을 사용하여 필요없는 특성(feature)제거 ex)index,familysize,QaE(설문에 대답하는데 걸린시간)
- shape를 사용하여 투표여부(target)을 [index수,1]형태로 변경
- One Hot Encoding

  QaA부터 QtA까지는 설문질문에 대한 답변으로 1부터5까지의 데이터를 갖고 있다. 이는 명목형 자료이기 때문에 One Hot Encoding을 사용하여 각각의 답에 대해 해당값만 True값으로 주고 나머지는 False값을 갖게끔 인코딩하였다.
  One Hot Encoding을 실행한 것과 실행하지 않은것의 predict값의 accuracy는각각 69%와 57%로 약12%의 차이를 보였다.


### 모델링(modeling)
target이 있기 때문에 지도학습의 분류 모델링을 사용하였다.

- SVM(support vector machine)

  데이터를 분리하는 margin(두 데이터 군과 결정 경계와 떨어져 있는 정도)이 큰 최적의 결정 결계를 찾는 알고리즘
  
  rbf커널의 Parameter C(엉마나 많은 데이터샘플이 다른 클래스에 놓이는 것을 허용하는가)를사용하여 soft-margin, hard-margin을 결정한다. 즉, C 값이 크면 과대적합이 될 가능성이 크며(너무 train을 완벽히하여 이상치의 존재 가능성을 낮게 봄) C값이 너무 작으면 과소 적합될 가능성이크다.(좀더 일반적인 결정경게를 찾아냄)
  
  커널기법을 사용하면 주어진 데이터를 고차원 특징 공간으로 사상해준다. 2차원 공간에서 분류 불가능 한 것을 3차원 공간으로 사상하여 분류 가능하게 함.
  
  Parameter gamma는 하나의 데이터 샘플이 영향력을 행사하는 거리(결정 경계의 곡률) 결정한다. gamma값이 크면 데이터 포인터가 영향력을 행사하는 거리가 짧아진다.
  
  grid search로 알맞는 값의 parameter를 찾아야한다.
- KNN(K-nearest-neighbor)
  
  최근접 인접 알고리즘으로 한 데이터에 가까이 있는 K개의 데이터를 토대로 분류한다.
  
  Parameter K(데이터에 가까이 있는 K개의 데이터를 추려내어 판단)를 사용하며,K가 작으면 과대적합(적은 데이터로 모델링)될 가능성이크고 K가 크면 과소적합(많은 데이티로 간단한 모델링)될 가능성이 크다.

- Naive Bayes

  베이즈 정리에 기반한 통계적 분류 기법으로 feature끼리 서로 독립이어야 한다.

- Decision Tree(결정트리)

  특정 기준(질문)에 따라 데이터를 구분하는 모델로 한번의 분기 때마다 변수 영역을 두개로 구분한다.
  
  트리에 가지를 많이 치게 되면 과대적합되며 min_sample_split으로 노드에 있는 데이터수를 세어 분기 여부를 결정하며, max_depth로 가지수를 조절한다.
  
- Logistic Regression

  회귀를 사용하여 데이터가 어떤 범주에 속할 확률을 0에서1 사이의 값으로 예측하고 그 확률에 따라 더 높은 범주에 속하는 것으로 분류해준다.
  
  로지스틱 함수를 구성하는 계수(log odds)와 절편에 대해 Log Loss를 최소화하는 값을 찾아야한다.


### Ensemble 기법
동일한 학습 알고리즘을 사용해 여러 모델을 학습하는 기법

대표적으로 Bagging, Boosting기법이 있다.

Bagging 기법으로 Bootstrap aggregating이 대표적이며 기본 데이터를 샘플링하여, n개의 dataset을 만들어 n개의 모델을 학습시키고 최종결과를 aggregation(집계)하는 방법이다. 병렬적이고 빠르다.(n개의 모델이 독집적으로 동시에 dataset을 학습하기 때문)

Boosting 기법으로 처음 모델은 기본 dataset을 그대로 학습하고, 다음 모델부터는 전체 데이터를 학습하되, 앞선 모델이 맞추지 못한 데이터에 중점을 두고 반복하여 학습한다. 직렬적이며 느리다.(결과를 기반으로 가중치를 결정하고 학습하기 때문) bagging보다 맞추기 어려운 문제를 맞추는데 특화되어 있다.

Stacking 기법은 서로 다른 모델을 조합해 최고의 성능을 내는 모델을 생성

- Ada Boosting

  Adaptive Boosting으로 이진 분류기가 틀련 부분을 adaptive하게 바꾸어가며 잘못 분류 된 데이터에 집중한다.(틀린 부분에 가중치를 부여하며 반복) 최종적으로 합쳐, 경계들이 복잡하게 생성되고 훨씬 더 정확한 예측이 가능하다.
  
  stump로 구성되어 있으며 하나의 stump에서 발생한 error가 다음 stump에 영향을 준다. 여러 stump가 순차적으로 연결되어 최종 결과를 도출한다.
  
  약한 학습기로 구성되어 있으며 stump형태를 갖고 있다. 각각의 stump는 다른 가중치를 갖고 있어 중요한 stump가 있다.

- Gradient Boosting

  가중치를 Gradient Descent(경사하강법)을 사용하며 손실함수를 parameter로 미분해 기울기를 구하고, 손실값이 작아지는 방향으로 parameter을 움직이게 하는 방법이다. Gradient가 현재까지 학습된 분류의 약점을 알려주고, 이후 모델이 약점을 중점으로 보완하는 방식

  하나의 leaf부터 시작하며, leaf는 target값에 대한 초기 추정 값을 나타낸다. tree로 구성되어 있으며, 이전 tree의 error가 다음 tree에 영향을 준다.

- XGBoosting

  약한 분류기를 세트로 묶어 정확도를 예측하는 기법으로 욕심쟁이(Greedy Algorithm)을 사용하여 분류기를 발견하고 분산처리를 사용하여 빠른 속도로 적합한 비중 parameter를 찾는 알고리즘이다.
  
  병렬처리를 사용하여 학습과 분류가 빠르며, 욕심쟁이 알고리즘을 사용하여 과적합이 잘일어나지 않는다.

- LightGBM

  Gradient Boosting의 단점인 느린 학습시간을 보완한 방법으로, 대용량 데이터 처리가 가능하고, 다른 모델보다 더 적은 자원을 사용하며 빠르다.  
  
   기준의 boosting모델들은 tree를 level-wise(갈수록 퍼짐) 방법을 사용했는데, lightbgm은 leaf-wise 트리분할을 사용한다.

- Stacking Ensemble

  각각의 모델을 쌓아서 essemble한 방법 
  
  Level 0: training dataset을 이용하여 sub-model의 예측 결과를 생성한다. 
  
  Level 1: Level 0의 output결과가 level 1의 input으로 들어가며 모델링한다.
  
  stack generalization 과정에서 level 0의 모델들은 다양한 예측 결과를 갖는 알고리즘을 선택하는 것이 level 1의 모델 생성에 좋다.
