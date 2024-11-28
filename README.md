
# 🏆 Hand Bone Semantic Segmentation

## 🥇 팀 구성원

### 이상진, 유희석, 정지훈, 천유동, 임용섭,박재우
## 프로젝트 소개
뼈는 우리 몸의 구조와 기능에 중요한 영향을 미치기 때문에, 정확한 뼈 분할은 의료 진단 및 치료 계획을 개발하는 데 필수적입니다.

Bone Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나로, 특히, 딥러닝 기술을 이용한 뼈 Segmentation은 많은 연구가 이루어지고 있으며, 다양한 목적으로 도움을 줄 수 있습니다.

평가지표는 `Dice Coefficient`을 사용합니다.
<br />

## 📅 프로젝트 일정
프로젝트 전체 일정

- 2024.11.11(월) ~ 2024.11.28(목)

<div align='center'>
    <img src='.\img\gantt.png', alt='간트 차트'>
    <p><em>간트 차트</em></p>
</div>

## 📊 데이터셋
이 코드는 `부스트캠프 AI Tech`에서 제공하는 데이터셋으로 다음과 같은 구성을 따릅니다. 
<div align='center'>
    <img src='.\img\classes.png', alt='클래스'>
    <p><em>Hand Bone Classes</em></p>
</div>


- 전체 이미지 개수 : 1088장
- 분류 클래스 : 29개의 클래스
- 전체 데이터 중 학습데이터 800장, 평가데이터 288장으로 사용
- 제출 형식 : Run-Length Encoding(RLE)형식으로 변환하여 CSV 파일로 제출
<br />

## 🏗️ 프로젝트 구조

<details>
<summary><span style="font-size: 20px; font-weight: bold">Project Structure</span></summary>

```plaintext
Project
│   README.md
├───📂 EDA
├───📂 img
├───📂 pytorch/
│   │   📄 .gitignore
│   │   📄 inference.py
│   │   📄 K-fold_ensemble.py
│   │   📄 requirements.txt
│   │   📄 train.py
│   ├───📂 configs
│   │       📄 base_train.yaml
│   ├───📂 data
│   │       📄 test_dataset.py
│   │       📄 train_dataset.py
│   ├───📂 loss
│   │       📄 loss.py
│   │       📄 loss_selector.py
│   ├───📂 models
│   │   ├───📂 CUSTOM
│   │   ├───📂 DUCKNet
│   │   ├───📂 fcn_resnet50
│   │   ├───📂 NestedUNet
│   │   ├───📂 RAPUNet
│   │   ├───📂 SegFormer
│   │   ├───📂 UNet
│   │   └───📂 UNet3plus
│   ├───📂 scheduler
│   │   │   📄 scheduler_selector.py
│   │   └───📂 CustomCAWR
│   │           📄 CustomCosineAnnealingWarmupRestarts.py
│   ├───📂 services
│   │       📄 kakao.py
│   │       📄 refresh_kakao_token.py
│   │       📄 sheet_kakao_key_update.py
│   │       📄 sheet_pull_kakao_key.py
│   │       📄 slack.py
│   │       📄 spreadsheet.py
│   ├───📂 trainer
│   │       📄 Earlystopping.py
│   │       📄 test.py
│   │       📄 test_rle.py
│   │       📄 trainer.py
│   │       📄 visualize.py
│   ├───📂 transform
│   │       📄 transform.py
│   └───📂 utils
│           📄 mlflow.py
│           📄 util.py
└───📂 tools
        📄 hard_voting.py
```

</details>


### 1) Pytorch
PyTorch 기반 학습 및 평가 코드를 포함한 메인 폴더입니다.
- `configs` : YAML 파일로 학습 파라미터(예: 학습률, 배치 크기)를 관리합니다.
- `data` : 데이터셋 로드와 관련된 코드를 포함합니다.
- `loss` : 커스텀 손실 함수 구현과 손실 함수 선택 로직을 제공합니다.
- `model` : UNet, SegFormer, DUCKNet 등 다양한 모델 아키텍처를 구현하며, 커스텀 레이어와 백본 모델 코드(예: RAPUNet)도 포함됩니다.
- `scheduler` : 학습률 스케줄러 구현과 선택 로직을 포함합니다.
- `services` : 카카오톡 메시지 전송, 슬랙 알림, Google 스프레드시트 연동 등 외부 서비스 통신을 처리합니다.
- `trainer` : 학습 및 평가 루프, Early Stopping, 테스트, 시각화 등의 학습 프로세스를 관리합니다.
- `transform` : 데이터 증강 및 변환을 처리합니다.
- `utils` : MLflow를 통한 실험 기록 관리와 기타 유틸리티 기능을 제공합니다.

### 2) tools
앙상블 처리(예: Hard Voting)를 위한 스크립트를 포함합니다.
<br />

## ⚙️ 설치
이 모델은 `Tesla v100 32GB`의 환경에서 작성 및 테스트 되었습니다.

### 전체 설치 (Linux Only)
`리눅스 환경`에서 모든 설정과 설치를 자동으로 실행하려면 아래 명령어를 사용하세요:
```bash
chmod +x setup.sh && ./setup.sh
```
### Dependencies Install
```bash
pip install -r requirements.txt
```

<details>
<summary>requirements 접기/펼치기</summary>

- albumentations==1.4.21
- altair==5.5.0
- fsspec==2023.9.2
- gitpython==3.1.43
- google-api-python-client==2.154.0
- google-auth==2.36.0
- google-auth-oauthlib==1.2.1
- joblib==1.4.2
- matplotlib==3.9.2
- mlflow==2.18.0
- numpy==1.26.0
- omegaconf==2.3.0
- opencv-python==4.10.0.84
- pandas==2.2.3
- PyYAML==6.0
- scikit-learn==1.5.2
- scipy==1.14.1
- seaborn==0.13.2
- streamlit==1.40.1
- torch==2.1.0
- torchvision==0.16.0
- tqdm==4.67.1
- ttach==0.0.3

</details>

<br />

## 🚀 빠른 시작
### Train
```bash
python3 train.py
```
#### Train Parser
- `--config` : 설정 파일 경로 (예: configs/base_train.yaml)

### Test
```bash
python3 inference.py --model {~.pt_file_path}
```

#### Test Parser
- `--model` : 훈련된 모델 가중치 파일 경로
- `--image_root` : 테스트 이미지가 저장된 디렉토리
- `--thr` : 분류 임계값
- `--resize` : 입력 이미지를 리사이즈할 크기

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
#### MLflow & Claudflare
<div align='center'>
    <img src='.\img\mlflow.png', alt='서버 사용 현황'>
    <p><em>ML Flow</em></p>
</div>

- MLfLow를 활용해 실험 기록을 추적하였습니다.
- Claudflare의 퀵 터널링을 통해서 local host에 올려진 MLflow를 웹으로 공유하도록 하였습니다.

#### Notion
- 프로젝트 관리 및 작업 목록, 회의 기록을 공유합니다.
#### Zoom
- 주기적인 회의와 실시간 피드백 제공을 위해 줌을 사용해 소통하였습니다.
#### GitHub
- 코드 버전 관리 및 협업을 위해 GitHub를 사용하였습니다.

<br />


## 🎯 프로젝트 결과
### Public
-  / 24
- Dice Coeff : 
### Private
-  / 24
- Dice Coeff : 

<br />

## 🏅 Wrap-Up Report   
### 

