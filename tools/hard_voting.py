import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from joblib import Parallel, delayed

def decode_rle_to_mask(rle, height, width):
    """RLE를 마스크로 디코딩"""
    try:
        s = rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(height * width, dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(height, width)
    except Exception as e:
        print(f"RLE 디코딩 실패: {e}")
        return np.zeros((height, width), dtype=np.uint8)
    
def encode_mask_to_rle(mask):
    """마스크를 RLE로 인코딩"""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def process_single_image(index, csv_data, threshold, height, width):
    """단일 이미지에 대한 앙상블 처리"""
    model_rles = []
    for data in csv_data:
        rle = data.iloc[index]['rle']
        if pd.isna(rle):
            model_rles.append(np.zeros((height, width)))
        else:
            model_rles.append(decode_rle_to_mask(rle, height, width))

    image = np.sum(model_rles, axis=0)
    image = np.where(image > threshold, 1, 0)

    result_rle = encode_mask_to_rle(image)
    file_info = f"{csv_data[0].iloc[index]['class']}_{csv_data[0].iloc[index]['image_name']}"
    return file_info, result_rle

def csv_ensemble(csv_paths, save_dir, threshold, height=2048, width=2048, n_jobs=-1):
    """CSV 앙상블 메인 함수"""
    csv_data = [pd.read_csv(path) for path in csv_paths]
    csv_column = len(csv_data[0])

    print(f"앙상블할 모델 수: {len(csv_data)}, threshold: {threshold}")

    results = Parallel(n_jobs=n_jobs)(  # 병렬 처리
        delayed(process_single_image)(i, csv_data, threshold, height, width)
        for i in tqdm(range(csv_column))
    )

    filename_and_class, rles = zip(*results)
    classes, filenames = zip(*[x.split("_") for x in filename_and_class])
    image_names = [os.path.basename(f) for f in filenames]

    df = pd.DataFrame({
        "image_name": image_names,
        "class": classes,
        "rle": rles,
    })

    df.to_csv(save_dir, index=False)
    print(f"결과 저장 완료: {save_dir}")
    
csv_paths = ["/data/ephemeral/home/ensem/CUNet.csv",
             "/data/ephemeral/home/ensem/duck.csv",
             "/data/ephemeral/home/ensem/DUCKNet-kfol.csv",
             "/data/ephemeral/home/ensem/fcn.csv",
             "/data/ephemeral/home/ensem/losschange.csv",
             "/data/ephemeral/home/ensem/reskfold.csv",
             "/data/ephemeral/home/ensem/seg.csv"]

threshold = 2
save_path = f"ensemble_threshold_{threshold}.csv"

csv_ensemble(csv_paths, save_path, threshold)