import torch.nn as nn
import torch
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss

    Args:
        alpha (float): 양성 클래스에 대한 가중치 (기본값 0.25)
        gamma (float): 난이도에 따라 손실을 조정하는 매개변수 (기본값 2)
        reduction (str): 손실값 축소 방식 ('mean', 'sum', 'none')

    Forward:
        preds (torch.Tensor): 모델 예측값 (로짓 형태)
        targets (torch.Tensor): 실제 라벨 값 (0 또는 1)
    
    Returns:
        torch.Tensor: Focal Loss 값
    """
    def __init__(self, alpha=.25, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, preds, targets):
        preds = F.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)
        BCE = F.binary_cross_entropy_with_logits(preds, targets, reduction=self.reduction)
        BCE_exp = torch.exp(-BCE)
        loss = self.alpha * (1 - BCE_exp)**self.gamma * BCE
        return loss
    
class DiceLoss(nn.Module):
    """
    Dice Loss

    Args:
        smooth (float): 분모와 분자의 0 나누기 문제를 방지하기 위한 작은 상수 (기본값 1)
    
    Forward:
        preds (torch.Tensor): 모델 예측값
        targets (torch.Tensor): 실제 라벨 값
    
    Returns:
        torch.Tensor: Dice 손실 값
    """
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, preds, targets):
        preds = F.sigmoid(preds)
        preds = preds.contiguous()
        targets = targets.contiguous()
        intersection = (preds * targets).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + self.smooth) / (preds.sum(dim=2).sum(dim=2) + targets.sum(dim=2).sum(dim=2) + self.smooth)))
        return loss.mean()

class FocalTveskyLoss(nn.Module):
    """
    Focal Tversky Loss

    Forward:
        preds (torch.Tensor): 모델 예측값
        targets (torch.Tensor): 실제 라벨 값
        smooth (float): 안정성 값
        alpha (float): False Positive 비중 가중치
        beta (float): False Negative 비중 가중치
        gamma (float): Focal Loss에서의 감쇠 계수

    Returns:
        torch.Tensor: Focal Tversky 손실 값
    """
    def __init__(self, weight=None, size_average=True):
        super(FocalTveskyLoss, self).__init__()

    def forward(self, preds, targets, smooth=1, alpha=0.3, beta=0.7, gamma=1.5):
        preds = F.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)
        TP = (preds * targets).sum()    
        FP = ((1-targets) * preds).sum()
        FN = (targets * (1-preds)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky

class IoULoss(nn.Module):
    """
    IoU Loss

    Args:
        smooth (float): 안정성 값
    
    Forward:
        preds (torch.Tensor): 모델 예측값
        targets (torch.Tensor): 실제 라벨 값
    
    Returns:
        torch.Tensor: IoU 손실 값
    """
    def __init__(self,smooth=1.):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, preds, targets):
        preds = F.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        sum_v = (preds + targets).sum()
        union = sum_v - intersection
        loss = (intersection + self.smooth)/ (union + self.smooth)
        return 1 - loss
    
class CombineLoss(nn.Module):
    """
    여러 손실 함수를 조합하여 가중치 기반 결합 손실 계산.

    Args:
        loss_list (list): 사용할 손실 함수 리스트
        weights (list): 각 손실 함수에 대한 가중치 리스트
    
    Forward:
        preds (torch.Tensor): 모델 예측값
        targets (torch.Tensor): 실제 라벨 값
    
    Returns:
        torch.Tensor: 가중 손실 값
    """
    def __init__(self, loss_list, weights):
        super(CombineLoss, self).__init__()
        self.loss_list = loss_list
        self.weights = weights
        assert len(loss_list)==len(weights), f"loss와 weights 길이가 다릅니다 맞춰주세요\nloss len : {len(loss_list)}, weights len : {len(weights)}"
        
    def forward(self, preds, targets):
        total_loss = 0.0
        for lossfn, weight in zip(self.loss_list, self.weights):
            total_loss += lossfn(preds, targets) * weight
        return total_loss

class CustomBCEWithLogitsLoss(nn.Module):
    """
    Base Code에서 사용된 BCEWithLogitsLoss
    """
    def __init__(self, **kwargs):
        super(CustomBCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, predictions, targets):
        return self.loss(predictions, targets)

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

class LovaszSoftmax(nn.Module):
    def __init__(self, reduction='mean'):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction

    def prob_flatten(self, input, target):
        assert input.dim() in [4, 5]
        num_class = input.size(1)
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        target_flatten = target.view(-1)
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            if num_classes == 1:
                input_c = inputs[:, 0]
            else:
                input_c = inputs[:, c]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)

        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

    def forward(self, inputs, targets):
        inputs, targets = self.prob_flatten(inputs, targets)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses