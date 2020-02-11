import torch
from settings import *


class SingleSourceDirectLoss(torch.nn.Module):
    def __init__(self, main_device):
        super(SingleSourceDirectLoss, self).__init__()
        self.main_device = main_device
        self.L1Loss = torch.nn.L1Loss().to(main_device)

    def forward(self, x):
        gt_mags_sq, pred_mags_sq, _, _, _, _ = x
        loss = self.L1Loss(pred_mags_sq[:, 0], gt_mags_sq[:, ISOLATED_SOURCE_ID])
        return loss


class IndividualLosses(torch.nn.Module):
    def __init__(self, main_device):
        super(IndividualLosses, self).__init__()
        self.main_device = main_device
        self.L1Loss = torch.nn.L1Loss().to(main_device)

    def forward(self, x):
        gt_mags_sq, pred_mags_sq, _, _, _, _ = x
        l1 = self.L1Loss(pred_mags_sq[:, 0], gt_mags_sq[:, 0])
        l2 = self.L1Loss(pred_mags_sq[:, 1], gt_mags_sq[:, 1])
        if K == 4:
            l3 = self.L1Loss(pred_mags_sq[:, 2], gt_mags_sq[:, 2])
            l4 = self.L1Loss(pred_mags_sq[:, 3], gt_mags_sq[:, 3])
            return [l1, l2, l3, l4]
        return [l1, l2]

class UnitWeightedLoss(torch.nn.Module):
    def __init__(self, main_device):
        super(UnitWeightedLoss, self).__init__()
        self.main_device = main_device
        self.L1Loss = torch.nn.L1Loss().to(main_device)

    def forward(self, x):
        gt_mags_sq, pred_mags_sq, _, _, _, _ = x
        l1 = self.L1Loss(pred_mags_sq[:, 0], gt_mags_sq[:, 0])
        l2 = self.L1Loss(pred_mags_sq[:, 1], gt_mags_sq[:, 1])
        if K == 4:
            l3 = self.L1Loss(pred_mags_sq[:, 2], gt_mags_sq[:, 2])
            l4 = self.L1Loss(pred_mags_sq[:, 3], gt_mags_sq[:, 3])
            return [l1, l2, l3, l4, l1+l2+l3+l4]
        return [l1, l2, l1+l2]


class EnergyBasedLossPowerP(torch.nn.Module):
    def __init__(self, main_device, power=1):
        super(EnergyBasedLossPowerP, self).__init__()
        self.main_device = main_device
        self.L1Loss = torch.nn.L1Loss().to(main_device)
        self.power = power

    def forward(self, x):
        gt_mags_sq, pred_mags_sq, gt_mags, _, _, _ = x
        if K == 2:
            l1 = self.L1Loss(pred_mags_sq[:, 0], gt_mags_sq[:, 0])
            w1 = (ACC_ENERGY / VOC_ENERGY)**self.power
            l2 = self.L1Loss(pred_mags_sq[:, 1], gt_mags_sq[:, 1])
            w2 = 1
            l11 = w1 * l1 + w2 * l2
            return [l1, l2, l11]
        else:
            l1 = self.L1Loss(pred_mags_sq[:, 0], gt_mags_sq[:, 0])
            w1 = (BAS_ENERGY / VOC_ENERGY)**self.power
            l2 = self.L1Loss(pred_mags_sq[:, 1], gt_mags_sq[:, 1])
            w2 = (BAS_ENERGY / DRU_ENERGY)**self.power
            l11 = w1 * l1 + w2 * l2
            l3 = self.L1Loss(pred_mags_sq[:, 2], gt_mags_sq[:, 2])
            w3 = 1
            l4 = self.L1Loss(pred_mags_sq[:, 3], gt_mags_sq[:, 3])
            w4 = (BAS_ENERGY / OTH_ENERGY)**self.power
            l22 = w3 * l3 + w4 * l4
            return [l1, l2, l3, l4, l11 + l22]

class EnergyBasedLossInstantwise(torch.nn.Module):
    def __init__(self, main_device, power=1):
        super(EnergyBasedLossInstantwise, self).__init__()
        self.main_device = main_device
        self.L1Loss = torch.nn.L1Loss().to(main_device)
        self.power = power

    def forward(self, x):
        gt_mags_sq, pred_mags_sq, gt_mags, _, _, _ = x
        if K == 2:
            l1 = self.L1Loss(pred_mags_sq[:, 0], gt_mags_sq[:, 0])
            w1 = pow(gt_mags[:, 1].pow(2).sum()/gt_mags[:, 0].pow(2).sum(), self.power)
            l2 = self.L1Loss(pred_mags_sq[:, 1], gt_mags_sq[:, 1])
            w2 = 1
            l11 = w1 * l1 + w2 * l2
            return [l1, l2, l11]
        else:
            l1 = self.L1Loss(pred_mags_sq[:, 0], gt_mags_sq[:, 0])
            w1 = pow(gt_mags[:, 2].pow(2).sum()/gt_mags[:, 0].pow(2).sum(), self.power)
            l2 = self.L1Loss(pred_mags_sq[:, 1], gt_mags_sq[:, 1])
            w2 = pow(gt_mags[:, 2].pow(2).sum()/gt_mags[:, 1].pow(2).sum(), self.power)
            l11 = w1 * l1 + w2 * l2
            l3 = self.L1Loss(pred_mags_sq[:, 2], gt_mags_sq[:, 2])
            w3 = 1
            l4 = self.L1Loss(pred_mags_sq[:, 3], gt_mags_sq[:, 3])
            w4 = pow(gt_mags[:, 2].pow(2).sum()/gt_mags[:, 3].pow(2).sum(), self.power)
            l22 = w3 * l3 + w4 * l4
            return [l1, l2, l3, l4, l11 + l22]
