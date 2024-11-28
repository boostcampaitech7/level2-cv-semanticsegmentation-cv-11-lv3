
# 🏆 Hand Bone Image Segmentation

## 🥇 팀 구성원

### 박재우, 이상진, 유희석, 정지훈, 천유동, 임용섭


## 프로젝트 소개
뼈는 우리 몸의 구조와 기능에 중요한 영향을 미치기 때문에, 정확한 뼈 분할은 의료 진단 및 치료 계획을 개발하는 데 필수적입니다.

Bone Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나로, 특히, 딥러닝 기술을 이용한 뼈 Segmentation은 많은 연구가 이루어지고 있으며, 다양한 목적으로 도움을 줄 수 있습니다.

질병 진단의 목적으로 뼈의 형태나 위치가 변형되거나 부러지거나 골절 등이 있을 경우, 그 부위에서 발생하는 문제를 정확하게 파악하여 적절한 치료를 시행할 수 있습니다.

수술 계획을 세우는데 도움이 됩니다. 의사들은 뼈 구조를 분석하여 어떤 종류의 수술이 필요한지, 어떤 종류의 재료가 사용될 수 있는지 등을 결정할 수 있습니다.

의료장비 제작에 필요한 정보를 제공합니다. 예를 들어, 인공 관절이나 치아 임플란트를 제작할 때 뼈 구조를 분석하여 적절한 크기와 모양을 결정할 수 있습니다.

의료 교육에서도 활용될 수 있습니다. 의사들은 병태 및 부상에 대한 이해를 높이고 수술 계획을 개발하는 데 필요한 기술을 연습할 수 있습니다.

본 대회는 외부 데이터셋을 사용하는 것이 금지되어 있습니다. 대신 모든 기학습 가중치 사용은 허용됩니다.

평가지표는 `Dice Coefficient`을 사용합니다. `Dice Coefficient`은 Semantic Segmentation에서 주로 사용되는 성능 측정 방법입니다.
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
    <img src='.\img\googlesheet_server.png', alt='서버 사용 현황'>
    <p><em>Google Sheet 서버 사용 현황</em></p>
    <img src='.\img\googlesheet_result.png', alt='학습 결과 저장'>
    <p><em>학습 결과 자동 저장</em></p>
</div>

### 프로젝트 관리
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
 data
     ├─test
     │    └─DCM
     │         ├─ID040
     │         │     image1661319116107.png
     │         │     image1661319145363.png
     │         └─ID041
     │                image1661319356239.png
     │                image1661319390106.png
     │
     ├─train
     │    ├─DCM
     │    │   ├─ID001
     │    │   │     image1661130828152_R.png
     │    │   │     image1661130891365_L.png
     │    │   └─ID002
     │    │          image1661144206667.png
     │    │          image1661144246917.png
     │    │        
     │    └─outputs_json
     │               ├─ID001
     │               │     image1661130828152_R.json
     │               │     image1661130891365_L.json
     │               └─ID002
                             image1661144206667.json
                             image1661144246917.json
 
```
이 코드는 `부스트캠프 AI Tech`에서 제공하는 데이터셋으로 다음과 같은 구성을 따릅니다. 
- 전체 이미지 개수 : 1088장
- 이미지 크기 : (2048, 2048)
- 분류 Classes(29개) : finger 1 ~ finger 19, Trapezium, Trapezoid, Capitate, Hamate, Scaphoid, Lunate, Triquetrum, Pisiform, Radius, Ulna
- 전체 데이터 중 학습데이터 800장, 평가데이터 288장으로 사용
- 제출 형식 : Run-Length Encoding(RLE)형식으로 변환하여 CSV 파일로 제출
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
