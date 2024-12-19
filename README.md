# Machine_Learning_Project

# 감성 분석 프로젝트: 트위터 데이터 기반 감성 분류 및 시각화

---

## 프로젝트 선정 이유  

최근에 SNS의 단점에 주목을 했었습니다. 왜냐하면, 인터넷 기술이 발전할 수록 모든 기록은 인터넷상에 남게되며 사람들은 각기 다른 감정을 표출하기 때문입니다. 이 감정들이 누군가를 칭찬하기위해, 위로하기위해 쓰이기도하지만, 한 사람의 잘못에 너무 과몰입을 하여 비난을 하는 그런 집단행동들도 어느샌가 우리 사회에 스며든 것 같습니다. 그래서 SNS상에 사람들의 주어진 감정표현을 바탕으로 분석해보고 싶었습니다. 따라서 Sentiment140 데이터셋을 활용해 긍정(Positive) 또는 부정(Negative)의 감정을 분류하고, 시간에 따른 감성 변화와 주요 변화를 시각화 하기 위해 감성 분석 모델의 구현을 시도해보았습니다.

---

## Code Overview

1. 데이터 로드 및 전처리
2. 감성 변화 시각화 및 저장
3. 감성 변화 피크 탐지
4. 로지스틱 회귀 모델 학습 및 정확도 평가
   
---

## 코드 설명  

### **0. 필요한 라이브러리 import and install

<img width="711" alt="library" src="https://github.com/user-attachments/assets/baeda773-17ed-48d1-bed1-12916a086177" />

프로젝트에서 사용할 라이브러리의 기능과 역할을 설정 및 이후 모든 기반을 제공하는 부분.
pandas : CSV파일 로드 및 데이터프레임 처리를 위한 라이브러리
matplotlib : 그래프를 그리기 위한 라이브러리
re : 특수문자 제거를 위한 라이브러리 (전처리이용)
numpy : 수치 연산 및 배열을 위한 라이브러리
scikit : 머신러닝 알고리즘과 데이터 처리 도구 제공
tensorflow : 딥러닝을 위한 라이브러리

### **1. 데이터 로드 및 전처리**  
<img width="496" alt="text_preprocess" src="https://github.com/user-attachments/assets/eb30202a-dee6-4fd9-8e04-9f99eda6b252" />

- **데이터셋**: Sentiment140 (`training.1600000.processed.noemoticon.csv`)  
- **전처리 과정**:
   - 텍스트를 **소문자화**.
   - 특수 문자 및 불필요한 공백 제거.
   - 감성 레이블을 **긍정(4)** → `POSITIVE`, **부정(0)** → `NEGATIVE`로 변환.
   - 날짜 데이터를 파싱하여 시간 순서대로 정리.
     
<img width="842" alt="load_preprocess" src="https://github.com/user-attachments/assets/baa1286a-d553-4574-a22e-94d3387c96fc" />
   
   - 데이터 로드
    CSV 파일에서 데이터를 읽음.
    date, text, target 열만 선택하여 불필요한 데이터를 제거.

   - 텍스트 데이터 전처리
    text_preprocessing 함수로 텍스트를 소문자로 변환, 특수문자 제거, 공백 정리.

   - 감성 레이블 변환
    target 열의 레이블을 0 → negative, 4 → positive로 변환

   - 결측값 및 날짜 처리
    결측값을 제거하고, date 열을 날짜 형식으로 변환.
    날짜 변환 실패 행을 삭제하여 데이터 품질을 유지.

   - 전처리된 데이터 반환
    최종적으로 정리된 데이터를 반환하며, 이후 분석과 모델 학습에 사용.
     
### **2. 감성 변화 시각화**  
<img width="764" alt="sentiment trends" src="https://github.com/user-attachments/assets/bae8f965-b22a-4d27-8db5-ad2c9747b019" />

- 감성 변화 시각화 및 저장 함수
  **데이터 처리 및 집계**
  - 날짜 정보만을 추출하고, 긍정과 부정별로 행렬 형식으로 변환하며 결측값은 0으로 채움
  - 그래프의 크기를 figsize-(12,6)으로 설정하여 그래프를 생성
  - X-axis Label : Date , Y-axis Label : Number of Tweets
  

### **3. 변화 피크 탐지 (peak_detection)
<img width="766" alt="peak detection" src="https://github.com/user-attachments/assets/cef70f71-ebd5-40da-a54f-f6f8a8e5f7fb" />

- 날짜를 표준 형식으로 변환하고 target별 트윗 수를 날짜별로 집계하고, 변환에 실패한 비정상 데이터는 제거처리.
- 날짜별 변화량의 차이를 계산값을 바탕으로 변화량이 큰 날짜를 탐지(detect).
- 피크인 날짜를 강조하여 표시하고 Peak를 강조하기 위해 피크 날짜에 빨간선(axvline)을 추가하여 표시




### **4. 모델 학습 및 평가 (정확도 평가 함수)**  
accuracy_evaluation 함수는 로지스틱 회귀(logistic regression)을 사용하여 학습 및 테스트 진행 및 정확도와 혼동행렬 출력.
- **TF-IDF(Term Frequency-Inverse Document Frequency)벡터화**
  - 텍스트 데이터를 수치 벡터로 변환하여 머신러닝에 적합하게 변환
  - max_features=5000설정으로 최대 5000개의 중요한 단어를 선택하여 벡터 변환
  - negative,positive를 각각 0,1로 변환하여 머신러닝 학습에 적합성을 높임
 
- **Data Distribution**
  - 학습 데이터와 테스트 데이터를 분리하여 모델의 학습 성능과 일반화 성능을 독립적으로 평가하기 위함
  - specific = 80%(train_set) , 20%(test_set)
    
- **Logistic Regression 모델**:
   - 시그모이드 함수 사용 : \[P(y=1|x) = \frac{1}{1 + e^{-(mx + b)}}\] -> 출력값이 항상 [0,1] 범위 내에 있도록 보장
   - Logistic Regression은 이진 분류 문제를 해결하기 위해 설계된 모델이므로 즉 긍정과 부정으로 분류하는 감성분석에서는
     Logistic Regression이 더 적합함.
   - Logistic REgression은 계산적으로 유리하며, 대규모 데이터셋에서도 빠른 학습이 가능하다는 효율성을 지님
   - 확률 해석 가능성 즉, 출력값이 [0,1] 범위의 확률로 해석이 가능하기 때문에 이 모델이 더 적합

     **왜 Linear Regression이 아닌가?**
        - 선형 회귀는 연속형 출력 값을 예측하는데 사용.
        - 이진 분류에 해당하는 감성 분석에는 선형방정식의 계산은 옳지 않음
        - 시그모이드 함수를 사용하는 Logistic Regression은 비선형관계처리에 더욱 유용
     
 - **혼동행렬(Confusion_matrix)가 무엇인가?**
   - 혼동행렬은 Logistic Regression과 같은 분류 모델의 성능을 평가하기 위한 도구이며, Logistic Regression과는 독립적인 평가 방법
   - 분류 모델이 실제값과 예측값 간의 관계를 2차원 표 형태로 요약한 것. Logistic Regressionr과 같이 이진 분류 문제나 다중 클래스 분류 문제에서 모델의 성능을 평가하는데 사용
  
  
---

   ### **혼동 행렬의 구조**

| 실제\예측       | 예측 부정 (0)  | 예측 긍정 (1)  |
|-----------------|----------------|----------------|
| **실제 부정 (0)** | True Negative (TN) | False Positive (FP) |
| **실제 긍정 (1)** | False Negative (FN) | True Positive (TP) |

- **True Positive (TP)**: 긍정으로 올바르게 예측된 경우.
- **True Negative (TN)**: 부정으로 올바르게 예측된 경우.
- **False Positive (FP)**: 부정을 긍정으로 잘못 예측한 경우.
- **False Negative (FN)**: 긍정을 부정으로 잘못 예측한 경우.

---

   ### **혼동 행렬의 목적**

1. **정확도 평가**:
   - 모델이 얼마나 정확히 데이터를 분류했는지 파악.

2. **성능 분석**:
   - 잘못된 예측(FP, FN)이 많은 특정 클래스나 데이터 유형을 확인하여 모델의 개선 방향을 모색.

3. **평가지표 계산**:
   - 정밀도(Precision), 재현율(Recall), F1-점수(F1-Score)와 같은 주요 성능 지표를 혼동 행렬을 기반으로 계산.

---

### **혼동 행렬의 원리**

1. **비교**:
   - 실제 데이터 레이블(예: `0`, `1`)과 모델 예측 값을 비교.
2. **집계**:
   - 각 조합의 발생 횟수를 집계하여 TN, TP, FP, FN 값을 계산.
3. **행렬 표현**:
   - 집계 결과를 2x2 표(또는 다중 클래스의 경우 NxN 표)로 정리.

---

### **혼동 행렬을 활용한 지표 계산**

혼동 행렬을 사용하여 아래와 같은 주요 지표를 계산할 수 있습니다:

- **정밀도(Precision)**: 예측된 긍정 중 실제로 긍정인 비율  
  \[
  Precision = \frac{TP}{TP + FP}
  \]

- **재현율(Recall)**: 실제 긍정 중 올바르게 예측된 비율  
  \[
  Recall = \frac{TP}{TP + FN}
  \]

- **F1-점수(F1-Score)**: 정밀도와 재현율의 조화 평균  
  \[
  F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
  \]
     
          
### **4. 주요 코드 구조**  
- `text_preprocessing()`: 텍스트 전처리 함수.  
- `load_preprocess()`: 데이터셋 로드 및 전처리.  
- `sentiment_trends()`: 감성 변화 시각화 및 저장.  
- `peak_detection()`: 감성 변화 피크 탐지.  
- `accuaracy_evaluation`: Logistic Regression 모델 정확도 평가.  

---

## 🏁 실행 결과  

### **감성 추이 시각화**  
시간에 따른 긍정 및 부정 트윗 수를 시각화한 결과
  
<img width="1086" alt="스크린샷 2024-12-18 오후 9 45 36" src="https://github.com/user-attachments/assets/ad592371-9378-41ee-855c-2ba8a9867060" />


---

### **감성 변화 피크 탐지**  
감성 변화가 급격한 날짜를 강조하여 표시한 결과:  

<img width="1080" alt="스크린샷 2024-12-18 오후 9 45 42" src="https://github.com/user-attachments/assets/702b7753-3b3b-4d35-9b07-1f05d8692c2a" />


---

### **모델 성능 평가**  

## 모델 성능 평가

Logistic Regression을 사용하여 모델 성능을 평가하였습니다. 평가 지표와 결과는 아래와 같습니다.

---

### 1. 정확도
- **정확도(Accuracy)**: **79.04%**
  - 테스트 데이터 중 약 79.04%를 정확히 분류하였습니다.

---

### 2. 혼동 행렬
혼동 행렬은 모델의 분류 성능을 시각적으로 보여줍니다:

| 실제\예측       | 부정 (0)      | 긍정 (1)      |
|-----------------|---------------|---------------|
| **부정 (0)**    | 123,842       | 35,652        |
| **긍정 (1)**    | 31,417        | 129,089       |

- **True Negatives (TN)**: 123,842 (부정을 올바르게 예측한 수)
- **True Positives (TP)**: 129,089 (긍정을 올바르게 예측한 수)
- **False Positives (FP)**: 35,652 (긍정으로 잘못 예측된 부정)
- **False Negatives (FN)**: 31,417 (부정으로 잘못 예측된 긍정)

---

### 3. 분류 보고서

| 지표             | 부정 (0)      | 긍정 (1)      | 가중 평균(Weighted Avg) |
|------------------|---------------|---------------|--------------------------|
| **정밀도(Precision)** | 80%          | 78%          | 79%                      |
| **재현율(Recall)**    | 78%          | 80%          | 79%                      |
| **F1-점수(F1-Score)** | 79%          | 79%          | 79%                      |

---

### 4. 주요 결과
- **정밀도(Precision)**: 예측된 긍정/부정 중 실제로 올바르게 분류된 비율.
- **재현율(Recall)**: 실제 긍정/부정 데이터 중 정확히 예측된 비율.
- **F1-점수(F1-Score)**: 정밀도와 재현율의 균형을 나타내는 지표.

Logistic Regression 모델은 테스트 데이터에서 균형 잡힌 성능을 보여주었고, 약 79%의 정확도를 달성하였습니다.
