import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
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
        BCE_exp = BCE.exp(-BCE)
        loss = self.alpha * (1 - BCE_exp)**self.gamma * BCE
        return loss
    
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, preds, targets):
        preds = preds.contiguous()
        targets = targets.contiguous()
        intersection = (preds * targets).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + self.smooth) / (preds.sum(dim=2).sum(dim=2) +   targets.sum(dim=2).sum(dim=2) + self.smooth)))
        return loss.mean()

class IoULoss(nn.Module):
    def __init__(self, smooth):
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
