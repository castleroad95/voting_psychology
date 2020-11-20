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
  rbf커널의 Parmeter C(엉마나 많은 데이터샘플이 다른 클래스에 놓이는 것을 허용하는가)를사용하여 soft-margin, hard-margin을 결정한다. 즉, C 값이 크면 과대적합이 될 가능성이 크며(너무 train을 완벽히하여 이상치의 존재 가능성을 낮게 봄) C값이 너무 작으면 과소 적합될 가능성이크다.(좀더 일반적인 결정경게를 찾아냄)
  커널기법을 사용하면 주어진 데이터를 고차원 특징 공간으로 사상해준다. 2차원 공간에서 분류 불가능 한 것을 3차원 공간으로 사상하여 분류 가능하게 함.
  Parameter gamma는 하나의 데이터 샘플이 영향력을 행사하는 거리(결정 경계의 곡률) 결정한다. gamma값이 크면 데이터 포인터가 영향력을 행사하는 거리가 짧아진다.
  grid search로 알맞는 값의 parameter를 찾아야한다.
- KNN(K-nearest-neighbor)


- Naive Bayes


- Decision Tree(결정트리)


- Logistic Regression


- Ada Boosting


- Gradient Boosting


- XGBoosting


- LightGBM


- Stacking Ensemble
