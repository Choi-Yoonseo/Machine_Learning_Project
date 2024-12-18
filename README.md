# Machine_Learning_Project

# 감성 분석 프로젝트: 트위터 데이터 기반 감성 분류 및 시각화

---

## 프로젝트 선정 이유  

최근에 SNS의 단점에 주목을 했었습니다. 왜냐하면, 인터넷 기술이 발전할 수록 모든 기록은 인터넷상에 남게되며 사람들은 각기 다른 감정을 표출하고는 합니다. 이 감정들이 누군가를 칭찬하기위해, 위로하기위해 쓰이기도하지만, 한 유명인의 잘못에 너무 과몰입을 하여 비난을 하는 그런 집단행동들도 어느샌가 우리 사회에 스며든 것 같습니다. 그래서 SNS상에 사람들의 주어진 감정표현을 바탕으로 분석해보고 싶었습니다. 따라서 Sentiment140 데이터셋을 활용해 긍정(Positive) 또는 부정(Negative)의 감정을 분류하고, 시간에 따른 감성 변화와 주요 변화를 시각화 하기 위해 감성 분석 모델의 구현을 시도해보았습니다.

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
    target 열의 레이블을 0 → NEGATIVE, 4 → POSITIVE로 변환하여 가독성 향상.

   - 결측값 및 날짜 처리
    결측값을 제거하고, date 열을 날짜 형식으로 변환.
    날짜 변환 실패 행을 삭제하여 데이터 품질을 유지.

   - 전처리된 데이터 반환
    최종적으로 정리된 데이터를 반환하며, 이후 분석과 모델 학습에 사용.
     
### **2. 감성 변화 시각화**  
<img width="764" alt="sentiment trends" src="https://github.com/user-attachments/assets/bae8f965-b22a-4d27-8db5-ad2c9747b019" />

- 감성 변화 시각화 및 저장 함수
  기능 : 날짜별 긍정(POSITIVE)과 부정(NEGATIVE) 트윗 수를 집계하여 꺾은선 그래프로 시각화.
  
  출력 : 그래프를 화면에 표시하거나, 지정된 경로(save_path)에 이미지 파일로 저장.
  
  활용 : 시간에 따른 감성 변화를 직관적 분석 가능, 중요한 트렌드를 파악 가능.

### **3. peak_detection
<img width="766" alt="peak detection" src="https://github.com/user-attachments/assets/cef70f71-ebd5-40da-a54f-f6f8a8e5f7fb" />

- 날짜를 표준 형식으로 변환하고 target별 트윗 수를 날짜별로 집계.
- 날짜별 변화량의 차이를 계산하여 변화량이 가장 큰 날짜(peak)를 탐지.
- 출력:
   피크 날짜 정보 : 감성 변화가 최대인 날짜 출력.
   강조 그래프 : 꺾은선 그래프에 피크 날짜를 빨간선과 텍스트로 표시.



### **3. 모델 학습 및 평가**  
- **TF-IDF 벡터화**: 텍스트 데이터를 벡터화하여 모델에 입력합니다.  
- **Logistic Regression 모델**:  
   - 트윗을 **긍정** 또는 **부정**으로 분류합니다.  
   - 정확도, 혼동 행렬, 분류 리포트를 통해 성능 평가를 수행합니다.  

### **4. 주요 코드 구조**  
- `preprocess_text()`: 텍스트 전처리 함수.  
- `load_and_preprocess_data()`: 데이터셋 로드 및 전처리.  
- `plot_sentiment_trends()`: 감성 변화 시각화 및 저장.  
- `detect_peaks()`: 감성 변화 피크 탐지.  
- `evaluate_model_accuracy()`: Logistic Regression 모델 정확도 평가.  

---

## 🏁 실행 결과  

### **1. 감성 추이 시각화**  
아래 그래프는 시간에 따른 긍정 및 부정 트윗 수를 시각화한 결과입니다:  
  
<img width="1086" alt="스크린샷 2024-12-18 오후 9 45 36" src="https://github.com/user-attachments/assets/ad592371-9378-41ee-855c-2ba8a9867060" />


---

### **2. 감성 변화 피크 탐지**  
다음 그래프는 감성 변화가 급격한 날짜를 강조하여 표시한 결과입니다:  

<img width="1080" alt="스크린샷 2024-12-18 오후 9 45 42" src="https://github.com/user-attachments/assets/702b7753-3b3b-4d35-9b07-1f05d8692c2a" />


---

### **3. 모델 성능 평가**  

**Logistic Regression 모델**의 평가 결과:  

- **정확도**: `79.04%`  
- **혼동 행렬**:
