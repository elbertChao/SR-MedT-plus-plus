import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.nn.functional import cross_entropy

EPSILON = 1e-32

class LogNLLLoss(_WeightedLoss):
    """
    LogNLLLoss (wrapper for CrossEntropy)
    Input: y_input (logits, shape [B, C, H, W])
    Input: y_target (indices, shape [B, H, W])
    """
    __constants__ = ['weight', 'reduction', 'ignore_index']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction=None,
                 ignore_index=-100):
        super(LogNLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, y_input, y_target):
        return cross_entropy(y_input, y_target, weight=self.weight,
                             ignore_index=self.ignore_index)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=0, apply_softmax=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.apply_softmax = apply_softmax

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1) if self.apply_softmax else logits
        fg_probs = probs[:, 1].reshape(-1)
        fg_targets = (targets == 1).float().reshape(-1)
        intersection = (fg_probs * fg_targets).sum()
        denominator = fg_probs.sum() + fg_targets.sum()
        dice_coeff = (2. * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice_coeff


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.56, reduction='mean', weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def forward(self, logits, targets):
        if self.weight is not None:
            if self.weight.device != logits.device:
                self.weight = self.weight.to(logits.device)

        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.weight)
        p_t = torch.exp(-ce_loss)
        focal_loss = torch.pow(1 - p_t, self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedDiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5,
                 start_gamma=1.5, max_gamma=2.56, class_weights=None,
                 dynamic_gamma=False, apply_softmax=True):
        super(CombinedDiceFocalLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dynamic_gamma = dynamic_gamma
        self.start_gamma = start_gamma
        self.max_gamma = max_gamma

        self.dice_loss = DiceLoss(apply_softmax=apply_softmax)

        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights).float()
        else:
            self.class_weights = None

        self.focal_loss = FocalLoss(gamma=self.start_gamma, weight=self.class_weights)

        print(f"Initialized CombinedDiceFocalLoss (Dice: {dice_weight}, Focal: {focal_weight})")
        if self.dynamic_gamma:
            print(f"Dynamic Gamma Enabled: Annealing from {start_gamma} to {max_gamma}")
        else:
            print(f"Static Gamma: {start_gamma}")
            
        if self.class_weights is not None:
            print(f"Class weights for Focal Loss used: {self.class_weights}")

    def update_gamma(self, current_epoch, total_epochs):
        if self.dynamic_gamma:
            # Cap at 60% of epochs so gamma is stable before early stopping typically fires
            progress = min(current_epoch / max(1, int(total_epochs * 0.6) - 1), 1.0)
            new_gamma = self.start_gamma + (self.max_gamma - self.start_gamma) * progress
            self.focal_loss.gamma = new_gamma
            return new_gamma
        return self.focal_loss.gamma

    def forward(self, logits, targets):
        d_loss = self.dice_loss(logits, targets)
        f_loss = self.focal_loss(logits, targets)
        return self.dice_weight * d_loss + self.focal_weight * f_loss

def classwise_iou(output, gt):
    gt_one_hot = torch.zeros_like(output).scatter_(1, gt[:, None, :], 1)
    output_binary_indices = torch.argmax(output, dim=1)
    output_one_hot = torch.zeros_like(output).scatter_(1, output_binary_indices[:, None, :], 1)

    dims = (0, *range(2, len(output.shape)))  # sum over B, H, W
    intersection = output_one_hot * gt_one_hot
    union = output_one_hot + gt_one_hot - intersection
    
    classwise_iou = (intersection.sum(dim=dims).float() + EPSILON) / (union.sum(dim=dims) + EPSILON)

    return classwise_iou


def classwise_f1(output, gt):
    epsilon = 1e-20
    n_classes = output.shape[1]

    output = torch.argmax(output, dim=1)
    true_positives = torch.tensor([((output == i) * (gt == i)).sum() for i in range(n_classes)]).float()
    selected = torch.tensor([(output == i).sum() for i in range(n_classes)]).float()
    relevant = torch.tensor([(gt == i).sum() for i in range(n_classes)]).float()

    precision = (true_positives + epsilon) / (selected + epsilon)
    recall = (true_positives + epsilon) / (relevant + epsilon)
    classwise_f1 = 2 * (precision * recall) / (precision + recall)

    return classwise_f1


def make_weighted_metric(classwise_metric):
    def weighted_metric(output, gt, weights=None):
        dims = (0, *range(2, len(output.shape)))

        if weights == None:
            weights = torch.ones(output.shape[1]) / output.shape[1]
        else:
            if len(weights) != output.shape[1]:
                raise ValueError("The number of weights must match with the number of classes")
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights)
            weights /= torch.sum(weights)

        classwise_scores = classwise_metric(output, gt).cpu()

        return classwise_scores 

    return weighted_metric


jaccard_index = make_weighted_metric(classwise_iou)
f1_score = make_weighted_metric(classwise_f1)


if __name__ == '__main__':
    output, gt = torch.zeros(3, 2, 5, 5), torch.zeros(3, 5, 5).long()
    print(classwise_iou(output, gt))