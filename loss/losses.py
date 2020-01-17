import torch
from settings import *


class SingleSourceDirectLoss(torch.nn.Module):
    def __init__(self, main_device):
        super(SingleSourceDirectLoss, self).__init__()
        self.main_device = main_device
        self.L1Loss = torch.nn.L1Loss().to(main_device)

    def forward(self, x):
        _, _, gt_masks, pred_masks = x
        loss = self.L1Loss(pred_masks[:, 0], gt_masks[:, ISOLATED_SOURCE_ID])
        return loss


class IndividualLosses(torch.nn.Module):
    def __init__(self, main_device):
        super(IndividualLosses, self).__init__()
        self.main_device = main_device
        self.L1Loss = torch.nn.L1Loss().to(main_device)
        self.loss_terms=torch.empty([len(SOURCES_SUBSET)])

    def forward(self, x):
        _, _, gt_masks, pred_masks = x
        for idx in range(len(SOURCES_SUBSET)):
            self.loss_terms[idx] = self.L1Loss(pred_masks[:, idx], gt_masks[:, idx])
        return self.loss_terms

class EnergyBasedLoss(torch.nn.Module):
    def __init__(self, main_device):
        super(EnergyBasedLoss, self).__init__()
        self.main_device = main_device
        self.L1Loss = torch.nn.L1Loss().to(main_device)
        self.loss_terms=torch.empty([len(SOURCES_SUBSET)])
        self.weights = torch.empty([len(SOURCES_SUBSET)])

    def forward(self, x):
        gt_mags, _, gt_masks, pred_masks = x
        l1 = self.L1Loss(pred_masks[:, 0], gt_masks[:, 0])
        w1 = 1/pow(gt_mags[:, 0].pow(2).sum(), 1)
        l2 = self.L1Loss(pred_masks[:, 1], gt_masks[:, 1])
        w2 = 1 / pow(gt_mags[:, 1].pow(2).sum(), 1)
        l11 = w1*l1 + w2*l2
        if K == 4:
            l3 = self.L1Loss(pred_masks[:, 2], gt_masks[:, 2])
            w3 = 1 / pow(gt_mags[:, 2].pow(2).sum(), 1)
            l4 = self.L1Loss(pred_masks[:, 3], gt_masks[:, 3])
            w4 = 1 / pow(gt_mags[:, 3].pow(2).sum(), 1)
            l22 = w3*l3 + w4*l4
            return [l1,l2,l3,l4,l11+l22]
        return [l1,l2,l11]
