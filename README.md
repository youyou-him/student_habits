# Student Habits – Academic Performance Prediction

머신러닝을 활용한 학생 학습 습관 기반 성적 예측 프로젝트

## 프로젝트 개요

이 프로젝트는 학생들의 생활 습관, 공부 시간, 수면 패턴, 정신 건강, 운동 빈도 등 다양한 요인을 바탕으로
시험 점수를 예측하는 머신러닝 회귀 분석 프로젝트입니다.

목표는 단순한 예측 정확도 향상을 넘어,
데이터 전처리, 피처 엔지니어링, 모델 비교, 앙상블 기법(Voting/Stacking) 적용을 통해
모델의 일반화 성능과 해석 가능성을 높이는 것입니다.

---

## 데이터 개요

* 데이터 출처: Kaggle Lifestyle Dataset (학생 습관 데이터를 가공하여 사용)
* 주요 피처

  * 공부 시간(study_hours_per_day)
  * 수면 시간(sleep_hours)
  * 운동 빈도(exercise_frequency)
  * 정신건강(mental_health_rating)
  * SNS 사용시간(social_media_hours)
  * 넷플릭스 시청시간(netflix_hours)
  * 출석률(attendance_percentage)
* 타깃(Target): exam_score (학생의 시험 점수)

---

## 모델링 과정

1. **데이터 전처리**

   * 이상치 및 결측치 처리
   * 범주형 변수 라벨 인코딩(Label Encoding)
   * 수치형 변수 스케일링(Scaling)
   * 파생 피처 생성 (예: 공부 효율, 수면 대비 공부시간 등)

2. **탐색적 데이터 분석(EDA)**

   * 왜도(Skewness), 첨도(Kurtosis) 분석으로 데이터 안정성 검증
   * 피어슨 상관관계 분석으로 주요 영향 요인 확인

3. **모델 학습 및 비교**

   * 회귀 기반 모델 비교

     * DecisionTreeRegressor
     * RandomForestRegressor
     * SVR
     * Ridge
     * Lasso
     * ElasticNet
     * XGBRegressor
     * LGBMRegressor
   * RMSE 기준으로 모델 성능 평가

4. **하이퍼파라미터 튜닝**

   * GridSearchCV 사용
   * 교차검증(cv=5) 기반 일반화 성능 측정

---

## 모델 성능 비교 (RMSE ↓)

| 전처리 / 모델                | DecisionTree | RandomForest | SVR   | Ridge    | Lasso    | ElasticNet | XGB  | LGBM | Voting     | Stacking   |
| :---------------------- | :----------- | :----------- | :---- | :------- | :------- | :--------- | :--- | :--- | :--------- | :--------- |
| 1차 점수 (원본)              | 9.79         | 6.55         | 16.02 | 5.64     | 5.82     | 6.43       | 6.68 | 6.43 | —          | —          |
| 스케일링                    | 9.68         | 6.54         | 8.39  | 5.64     | 6.02     | 7.63       | 6.68 | 6.40 | —          | —          |
| 파생 피처 + 스케일링            | 9.67         | 6.47         | 7.88  | **5.60** | 5.93     | 7.38       | 6.62 | 6.07 | —          | —          |
| 튜닝 후 CV                 | 8.71         | 6.46         | 5.73  | 5.61     | **5.59** | 5.60       | 5.78 | 5.95 | —          | —          |
| 앙상블 (Voting / Stacking) | —            | —            | —     | —        | —        | —          | —    | —    | **5.5384** | **5.5451** |

결과적으로 Voting 앙상블(Ridge + SVR + LGBM)이 RMSE 5.5384로 가장 우수한 성능을 기록했습니다.
Stacking(Ridge 메타모델) 역시 RMSE 5.5451로 유사한 수준의 일반화 성능을 유지했습니다.

---

## 앙상블 모델 요약

**Voting (Weighted Average)**

* Ridge: 선형 안정성
* SVR(linear): 거리 기반 세밀 조정
* LGBM: 비선형 패턴 학습
  → 세 모델의 예측값을 RMSE 역수 가중 평균으로 결합하여 최소 RMSE 달성

**Stacking**

* Base Models: Ridge, Lasso, SVR, LGBM
* Meta Model: Ridge (cv=5, L2 규제 적용)
  → 교차검증 기반 out-of-fold 예측을 통해 과적합을 방지하면서 안정적인 조합 수행

---

## 결과 해석

* SHAP 분석 결과 주요 영향 피처:

  * 시험 점수에 긍정적 영향: 공부시간, 출석률, 수면시간, 운동빈도, 정신건강
  * 부정적 영향: SNS 사용시간, 넷플릭스 시청시간, 쉬는시간
* 데이터의 왜도·첨도 모두 정상 범위 → 안정적인 분포 유지
* Ridge/Lasso 규제를 통해 불필요한 변수의 영향을 자동 억제 → 일반화 성능 향상

---

## 결론

본 프로젝트는 머신러닝 회귀모델과 앙상블 기법을 활용해
학생의 생활습관 요인으로부터 학업 성취도를 예측하였다.
Voting 앙상블이 가장 낮은 RMSE(5.5384)를 기록했으며,
과적합 없이 안정적이고 해석 가능한 모델을 구축하였다.

---

## 기술 스택

* Python 3.10
* Scikit-learn
* XGBoost / LightGBM
* NumPy, Pandas, Matplotlib
* Jupyter Notebook

---

## 파일 구성

```
student_habits/
├── student_habits.ipynb    # 전체 분석 및 모델링 코드
├── data/                   # 원본 및 전처리 데이터
└── README.md               # 프로젝트 설명 파일
```

---

## 향후 개선 방향

* 학습 데이터 확대 및 추가 변수(가정환경, 학습 집중도 등) 반영
* AutoML 기반 모델 탐색 및 비교
* Flask 또는 Streamlit을 활용한 웹 대시보드 시각화
