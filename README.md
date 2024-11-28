
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

- 2024.10.28(월) ~ 2024.11.7(목)

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
- **4** / 24
- Dice Coeff : **0.9200**
### Private
- **5** / 24
- Dice Coeff : **0.9073**

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
  
### 2) tools
- `cloba2datu.ipynb`: cord 데이터셋을 datumaro 형식으로 변환합니다.
- `datu2ufo.ipynb`: datumaro 형식의 데이터셋을 UFO 형식으로 변환합니다.
- `ufo2datu.ipynb`: UFO 형식의 데이터셋을 datumaro 형식으로 변환합니다.
- `easyocr_pseudo.ipynb`: Easyocr 라이브러리를 활용해서 pseudo-labeling을 진행합니다.
- `img_hash.ipynb`: 이미지 hash값을 이용해 중복 이미지를 제거하고 200장의 이미지를 추출합니다.
- `inference_visualize.ipynb`: inference한 결과를 test이미지에 시각화합니다.
- `server-status.py`: 서버의 CPU, 메모리, GPU 상태를 조회합니다.
- `data_duplication_check.py`: hash값으로 이미지가 겹치는지 확인하고 제거합니다.
- `streamlit_viz.py`: dataset의 annotation을 streamlit으로 시각화합니다.

<br />

## ⚙️ 설치

### Dependencies
이 모델은 Tesla v100 32GB의 환경에서 작성 및 테스트 되었습니다.
또 모델 실행에는 다음과 같은 외부 라이브러리가 필요합니다.

```bash
pip install -r requirements.txt
```
<details>
<summary>requirements 접기/펼치기</summary>

- lanms==1.0.2
- opencv-python==4.10.0.84
- shapely==2.0.5
- albumentations==1.4.12
- torch==2.1.0
- tqdm==4.66.5
- albucore==0.0.13
- annotated-types==0.7.0
- contourpy==1.1.1
- cycler==0.12.1
- eval_type_backport==0.2.0
- filelock==3.15.4
- fonttools==4.53.1
- fsspec==2024.6.1
- imageio==2.35.0
- importlib_resources==6.4.2
- Jinja2==3.1.4
- kiwisolver==1.4.5
- lazy_loader==0.4
- MarkupSafe==2.1.5
- matplotlib==3.7.5
- mpmath==1.3.0
- networkx==3.1
- numpy==1.24.4
- nvidia-cublas-cu12==12.1.3.1
- nvidia-cuda-cupti-cu12==12.1.105
- nvidia-cuda-nvrtc-cu12==12.1.105
- nvidia-cuda-runtime-cu12==12.1.105
- nvidia-cudnn-cu12==8.9.2.26
- nvidia-cufft-cu12==11.0.2.54
- nvidia-curand-cu12==10.3.2.106
- nvidia-cusolver-cu12==11.4.5.107
- nvidia-cusparse-cu12==12.1.0.106
- nvidia-nccl-cu12==2.18.1
- nvidia-nvjitlink-cu12==12.6.20
- nvidia-nvtx-cu12==12.1.105
- packaging==24.1
- pillow==10.4.0
- pydantic==2.8.2
- pydantic_core==2.20.1
- pyparsing==3.1.2
- python-dateutil==2.9.0.post0
- PyWavelets==1.4.1
- PyYAML==6.0.2
- scikit-image==0.21.0
- scipy==1.10.1
- six==1.16.0
- sympy==1.13.2
- tifffile==2023.7.10
- tomli==2.0.1
- triton==2.1.0
- typing_extensions==4.12.2
</details>
<br />

## 🚀 빠른 시작
### Train
```python
python custom_train.py 
```
### Train Parser
기본 설정
- `--data_dir` : Dataset directory
- `--model_dir` : Model directory (기본값 : EAST Model)
- `--device` : `cuda` or `cpu` ( 기본값 : cuda )

학습 설정
- `--num_workers` : 학습할 프로세스 수 (기본값 : 8)
- `--image_size` : 학습할 이미지 크기 (기본값 : 2048)
- `--input_size` : 학습할 입력 이미지 크 (기본값 : 1024)
- `--batch_size` : 배치 크기 결정 ( 기본값 : 8)
- `--learning_rate` : 학습률 설정 ( 기본값 : 0.001)
- `--max_epochs` : 학습할 에폭 수 (기본값 : 150)
- `--save_interval` : 가중치를 저장할 epoch 간격 (기본값 : 5)

## 🏅 Wrap-Up Report   
### [ Wrap-Up Report 👑]
