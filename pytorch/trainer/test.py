import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
from trainer.test_rle import encode_mask_to_rle
from utils.util import get_IND2CLASS
import numpy as np
import gc

def test(args, data_loader, thr=0.5):
    model = torch.load(args.model)
    model = model.cuda()
    model.eval()
    rles = []
    filename_and_class = []
    with torch.no_grad():
        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()    
            # outputs = model(images)['out']
            outputs = model(images)
            
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{get_IND2CLASS()[c]}_{image_name}")
    return rles, filename_and_class

def batch_soft_voting(data_loader, model_paths, thr=0.5):
    """
    배치 단위로 소프트 보팅 수행 후 RLE로 변환 (결과를 메모리에 유지).
    
    Args:
        args: Argument 객체.
        data_loader: DataLoader 객체.
        model_paths: 모델 파일 경로 리스트.
        thr: 임계값 (0~1).
    
    Returns:
        rles: RLE 인코딩된 결과 리스트.
        filename_and_class: 파일 이름과 클래스 리스트.
    """
    # 모델 로드
    models = [torch.load(path).cuda().eval() for path in model_paths]

    rles = []
    filename_and_class = []

    with torch.no_grad():
        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()

            # 각 모델의 확률 맵 계산
            outputs_list = []
            for model in models:
                outputs = model(images)
                outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
                outputs = torch.sigmoid(outputs)
                outputs_list.append(outputs)

            # 소프트 보팅: 모델별 평균 확률 계산
            soft_voted_outputs = torch.mean(torch.stack(outputs_list), dim=0)

            # 임계값 적용: (0 또는 1로 변환)
            final_masks = (soft_voted_outputs > thr).detach().cpu().numpy()

            # RLE 인코딩
            for mask, image_name in zip(final_masks, image_names):
                for c, segm in enumerate(mask):
                    rle = encode_mask_to_rle(segm)  # RLE로 변환
                    rles.append(rle)
                    filename_and_class.append(f"{get_IND2CLASS()[c]}_{image_name}")

            # 메모리 정리
            torch.cuda.empty_cache()

    # 모델 메모리 정리
    for model in models:
        del model
    torch.cuda.empty_cache()
    gc.collect()

    return rles, filename_and_class
