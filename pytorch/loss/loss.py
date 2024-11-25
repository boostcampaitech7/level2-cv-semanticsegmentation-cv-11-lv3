import torch.nn as nn
import torch
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
        BCE_exp = torch.exp(-BCE)
        loss = self.alpha * (1 - BCE_exp)**self.gamma * BCE
        return loss
    
class DiceLoss(nn.Module):
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
    def __init__(self, weight=None, size_average=True):
        super(FocalTveskyLoss, self).__init__()

    def forward(self, preds, targets, smooth=1, alpha=0.4, beta=0.6, gamma=1):
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