import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_coef(y_true, y_pred) -> torch.Tensor:
    '''
    Dice Coef를 계산하여 반환
    
    Args:
        y_true (torch.Tensor): GT 마스크 텐서
        Y_pred (torch.Tensor): 예측 마스크 텐서
        
    Return:
        dice_score : 클래스별 Dice coef 값
    '''
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    inter = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    dice_score = (2. * inter + eps) / (torch.sum(y_true_f,-1) + torch.sum(y_pred_f,-1) + eps)
    return dice_score

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()   
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +   target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def combine_loss(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss