
# 🏆 Hand Bone Semantic Segmentation

## 🥇 팀 구성원

### 이상진, 유희석, 정지훈, 천유동, 임용섭,박재우
## 프로젝트 소개
뼈는 우리 몸의 구조와 기능에 중요한 영향을 미치기 때문에, 정확한 뼈 분할은 의료 진단 및 치료 계획을 개발하는 데 필수적입니다.

Bone Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나로, 특히, 딥러닝 기술을 이용한 뼈 Segmentation은 많은 연구가 이루어지고 있으며, 다양한 목적으로 도움을 줄 수 있습니다.

평가지표는 Dice Coefficient을 사용합니다.

<br />

## 📅 프로젝트 일정
프로젝트 전체 일정

- 2024.11.11(월) ~ 2024.11.28(목)

<div align='center'>
    <img src='.\img\gantt.png', alt='간트 차트'>
    <p><em>간트 차트</em></p>
</div>

## 🤝 협업 Tools
### 학습 알림
#### Slack & KakaoTalk
- **학습 시작, 완료, 에러 발생** 시 Slack과 카카오톡을 통해 실시간으로 알림을 전송하여 팀원들이 즉시 확인할 수 있도록 하였습니다.
<div align='center'>
    <img src='.\img\slack.png', alt='slack 학습 알림'>
    <p><em>Slack 알림</em></p>
    <img src='.\img\kakaotalk.png', alt='카카오톡 학습 알림'>
    <p><em>카카오톡 알림</em></p>
</div>

#### Google Sheet
- 서버 사용 현황 확인 및 학습 완료 시 학습결과를 자동으로 작성합니다.
<div align='center'>
    <img src='.\img\spreadsheet_server.png', alt='서버 사용 현황'>
    <p><em>Google Sheet 서버 사용 현황</em></p>
    <img src='.\img\spreadsheet_result.png', alt='학습 결과 저장'>
    <p><em>학습 결과 자동 저장</em></p>
</div>

### 프로젝트 관리
#### ML Flow
<div align='center'>
    <img src='.\img\mlflow.png', alt='서버 사용 현황'>
    <p><em>ML Flow</em></p>
</div>

- ML FLow를 활용해 실험 기록을 추적하였습니다.

#### Notion
- 프로젝트 관리 및 작업 목록, 회의 기록을 공유합니다.
#### Zoom
- 주기적인 회의와 실시간 피드백 제공을 위해 줌을 사용해 소통하였습니다.
#### GitHub
- 코드 버전 관리 및 협업을 위해 GitHub를 사용하였습니다.

<br />


## 🥈 프로젝트 결과
### Public
-  / 24
- Dice : 
### Private
-  / 24
- Dice : 

<br />

## 🥉 데이터셋 구조
```
dataset
     ├─test
     │    └─DCM
     │         ├─ID040
     │         │     *.png
     │         └─ID041
     │               *.png
     ├─train
     │    ├─DCM
     │    │   ├─ID001
     │    │   │     *.png
     │    │   └─ID002
     │    │         *.png
     │    │        
     │    └─outputs_json
     │               ├─ID001
     │               │     *.json
     │               └─ID002
                           *.json
```

<div align='center'>
    <img src='.\img\classes.png', alt='클래스'>
    <p><em>Hand Bone Classes</em></p>
</div>

이 코드는 `부스트캠프 AI Tech`에서 제공하는 데이터셋으로 다음과 같은 구성을 따릅니다. 
- 전체 이미지 개수 : 1088장
- 분류 클래스 : 29개의 클래스
- 전체 데이터 중 학습데이터 800장, 평가데이터 288장으로 사용
- 제출 형식 : Run-Length Encoding(RLE)형식으로 변환하여 CSV 파일로 제출
- 
<br />

## 🥉 프로젝트 구조
```
project/pytorch
     ├─ checkpoints
     ├─ configs
     │    └─ base_train.yaml
     │
     ├─ data
     │    ├─ test_dataset.py
     │    └─ train_dataset.py
     │
     ├─ loss
     ├─ models
     ├─ scheduler
     │    ├─ CustomCAWR
     │    └─ scheduler_selector.py
     │
     ├─ services
     │    ├─ kakao.py
     │    ├─ refresh_kakao_token.py
     │    ├─ sheet_kakao_key_update.py
     │    ├─ sheet_pull_kakao_key.py
     │    ├─ slack.py
     │    └─ spreadsheet.py
     │
     ├─ trainer
     │    ├─ Earlystopping.py
     │    ├─ test.py
     │    ├─ test_rle.py
     │    ├─ trainer.py
     │    └─ visualize.py
     │
     ├─ transform
     │    └─ transform.py
     │
     ├─ utils
     │    ├─ mlflow.py
     │    └─ util.py
     │
     ├─ K-fold_ensemble.py
     ├─ inference.py
     ├─ main.py
     ├─ requirements.txt
     └─ train.py
    
```
### 1) Services
- `kakao.py`: 카카오톡 메세지 전송, uuid 추출 기능을 제공합니다.
- `refresh_kakao_token.py`: 카카오 리프레시 토큰을 이용해 카카오톡 액세스 토큰을 갱신합니다. (crontab 8시간 마다 실행)
- `sheet_kakao_key_update.py`: 갱신된 카카오톡 액세스 토큰을 Google Sheet에 업데이트합니다. (crontab 30초마다 실행)
- `sheet_pull_kakao_key.py`: Google Sheet에서 최신 카카오톡 액세스 토큰을 가져와 로컬 환경에 업데이트 합니다. (crontab 30초마다 실행)
- `slack.py`: 학습현황을 슬랙 메시지로 전송하는 기능을 제공합니다.
- `spreadsheet.py`: 서버 학습 현황 및 학습 데이터를 Google Sheet에 업데이트, 추가하는 기능을 제공합니다.
  
### 2) 


<br />

## ⚙️ 설치

### Dependencies
이 모델은 Tesla v100 32GB의 환경에서 작성 및 테스트 되었습니다.
또 모델 실행에는 다음과 같은 외부 라이브러리가 필요합니다.

```bash
pip install -r requirements.txt
```
- opencv-python-headless==4.10.0.84
- pandas==2.2.3
- scikit-learn==1.5.2
- albumentations==1.4.18
- matplotlib==3.9.2

<br />

## 🚀 빠른 시작
### Train
```python
~/pytorch python train.py
```
### Train Config
#### ~/pytorch/configs/base_train.yaml

## 🏅 Wrap-Up Report   
### 

