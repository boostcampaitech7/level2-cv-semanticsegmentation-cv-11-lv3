import torch
import numpy as np
import random
import os
import json
import cv2
from trainer.test_rle import decode_rle_to_mask
import pandas as pd
from tqdm.auto import tqdm

PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

def set_seed(seed):
    '''
    랜덤 시드를 설정하여 코드 실행의 재현성을 보장
    
    Args:
        seed (int) : 사용할 시드 값
        
    Return:
        None
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
def label2rgb(label) -> np.ndarray:
    '''
    클래스별 마스크를 RGB 이미지로 변환
    
    Args:
        label : 클래스별 마스크 배열
    
    Return:
        np.ndarray: RGB 이미지 배열
    '''
    image_size = label.shape[1:] + (3, )
    image = np.zeros(image_size, dtype=np.uint8)
    
    for i, class_label in enumerate(label):
        image[class_label == 1] = PALETTE[i]
    return image

def find_file(root_path, extension:str) -> set:
    '''
    폴더에서 특정 확장자를 가진 파일을 재귀적으로 검색
    
    Args:
        root_path (str): 검색할 루트 디렉토리의 경로.
        extension (str): 검색할 파일 확장자 (ex: '.png', '.json')
    
    Return:
        set: 검색된 파일들의 상대 경로를 포함하는 집합
    '''
    result = {
        os.path.relpath(os.path.join(root,fname), start=root_path)
        for root, _dirs, files in os.walk(root_path)
        for fname in files
        if os.path.splitext(fname)[1].lower() == extension
    }
    return result

def save_best(model, save_dir, file_name='best.pt'):
    '''
    전체 모델을 저장합니다
    
    Args:
        model : 저장할 모델
        save_dir : 저장할 폴더
        file_name : 저장할 파일명
        
    Return:
        None
    '''
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, file_name)
    torch.save(model, output_path)
    print(f"최종 모델이 {output_path}에 저장되었습니다.")
    
def save_ckpt(model, save_dir, epoch):
    '''
    모델의 가중치만 저장합니다.
    
    Args:
        model : 저장할 모델
        save_dir : 저장할 폴더
        epoch : 에폭 번호
        
    Return:
        None
    '''
    os.makedirs('checkpoints', exist_ok=True)
    output_path = os.path.join(save_dir,f"{epoch}_checkpoint.pth")
    torch.save(model.state_dict(), output_path)
    print(f"{output_path}에 {epoch} 체크포인트 파일이 저장되었습니다.")
    
def read_annotations(json_path) -> dict:
    """
    JSON 파일을 읽고 'annotations' 키의 값을 반환

    Args:
        json_path (str): JSON 파일 경로

    Returns:
        dict: 'annotations' 데이터를 반환
    """
    with open(json_path,'r') as f:
        return json.load(f)['annotations']
    
def inference_save(rles, filename_and_class, image_root, classes, save_dir="inference_results", num_samples=10):
    """
    인퍼런스 결과 이미지를 저장

    Args:
        rles (list): RLE 인코딩된 마스크 리스트
        filename_and_class (list): 파일명 리스트 (이미지 이름만 포함)
        image_root (str): 원본 이미지 폴더 경로
        classes (list): 클래스 리스트
        save_dir (str): 저장할 디렉토리 경로 (기본값: "inference_results")
        num_samples (int): 저장할 이미지 샘플 개수 (기본값: 10)
    """
    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)

    # 샘플 개수 설정 (중복 없이 이미지 이름만 사용)
    image_names = list(set([x.split("_", 1)[1] for x in filename_and_class]))
    sample_indices = list(range(min(num_samples, len(image_names))))

    for idx in tqdm(sample_indices, desc="Saving inference results"):
        try:
            # 이미지 이름 추출
            image_name = image_names[idx]
            image_path = os.path.join(image_root, image_name)

            # 이미지 읽기
            image = cv2.imread(image_path)
            if image is None:
                print(f"이미지 불러오기 오류: {image_path}")
                continue

            # 원본 이미지 크기
            orig_h, orig_w = image.shape[:2]

            # 예측 마스크 생성
            preds = []
            for class_idx, class_name in enumerate(classes):
                rle_index = idx * len(classes) + class_idx
                rle = rles[rle_index]
                if rle is not None and rle != "":
                    pred = decode_rle_to_mask(rle, 2048, 2048)
                    # 예측 마스크를 원본 이미지 크기로 리사이즈
                    pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                else:
                    pred = np.zeros((orig_h, orig_w), dtype=np.uint8)
                preds.append(pred)

            # 예측 결과를 RGB 이미지로 변환
            preds = np.stack(preds, axis=0)
            pred_rgb = label2rgb(preds)

            # 원본 이미지와 예측 결과 시각화
            viz = np.hstack((image, pred_rgb))

            # 데이터 타입 변환 및 디버깅 로그
            if viz.dtype != np.uint8:
                print(f"viz 배열의 타입이 잘못되었습니다. 현재 타입: {viz.dtype}, uint8로 변환합니다.")
                viz = viz.astype(np.uint8)
            print(f"저장할 이미지 크기: {viz.shape}, 데이터 타입: {viz.dtype}")

            # 결과 이미지 저장
            image_base_name = os.path.splitext(os.path.basename(image_name))[0]
            save_path = os.path.join(save_dir, f"{image_base_name}_infer.png")
            saved = cv2.imwrite(save_path, viz)
            if not saved:
                print(f"이미지 저장 실패: {save_path}")
            else:
                print(f"이미지 저장 성공: {save_path}")
        except Exception as e:
            print(f"오류 발생: {e}")

    print("인퍼런스 결과 저장이 완료되었습니다.")
    
def inference_to_csv(filename_and_class, rles, output_name="output.csv"):
    """
    인퍼런스 결과를 CSV 파일로 저장

    Args:
        filename_and_class (list): 인퍼런스 결과의 파일명과 클래스 정보 리스트
        rles (list): RLE 인코딩된 마스크 리스트
        output_name (str): 저장할 CSV 파일명
    """
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        'rle': rles,
    })
    df.to_csv(output_name, index=False)


def get_classes():
    classes = [
            'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
            'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
            'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
            'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
            'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
            'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
        ]
    return classes