import torch
import torch.nn.functional as F
from settings import *


def gradient_loss(gen_frames, gt_frames, alpha=1):
    def gradient(x):
        # idea from tf.image.image_gradients(image)
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        # x: (b,c,h,w), float32 or float64
        # dx, dy: (b,c,h,w)

        h_x = x.size()[-2]
        w_x = x.size()[-1]
        # gradient step=1
        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
        dx, dy = right - left, bottom - top
        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy

    # gradient
    gen_dx, gen_dy = gradient(gen_frames)
    gt_dx, gt_dy = gradient(gt_frames)
    #
    grad_diff_x = torch.abs(gt_dx - gen_dx)
    grad_diff_y = torch.abs(gt_dy - gen_dy)

    # condense into one tensor and avg
    return torch.mean(grad_diff_x ** alpha + grad_diff_y ** alpha)


class CUNetLoss(torch.nn.Module):
    def __init__(self, main_device):
        super(CUNetLoss, self).__init__()
        self.main_device = main_device
        self.L1Loss = torch.nn.L1Loss().to(main_device)

    def forward(self, x):
        gt_mags_sq, pred_mags_sq, _, _, _, _ = x
        loss = self.L1Loss(pred_mags_sq, gt_mags_sq)
        return loss


class GradientLoss(torch.nn.Module):
    def __init__(self, main_device, power=1):
        super(GradientLoss, self).__init__()
        self.main_device = main_device
        self.L1Loss = torch.nn.L1Loss().to(main_device)
        self.power = power

    def forward(self, x):
        gt_mags_sq, pred_mags_sq, gt_mags, _, _, _ = x
        lg = gradient_loss(pred_mags_sq, gt_mags_sq)
        if K == 2:
            l1 = self.L1Loss(pred_mags_sq[:, 0], gt_mags_sq[:, 0])
            w1 = (ACC_ENERGY / VOC_ENERGY) ** self.power
            l2 = self.L1Loss(pred_mags_sq[:, 1], gt_mags_sq[:, 1])
            w2 = 1
            l11 = w1 * l1 + w2 * l2 + lg
            return [l1, l2, lg, l11]
        else:
            l1 = self.L1Loss(pred_mags_sq[:, 0], gt_mags_sq[:, 0])
            w1 = (BAS_ENERGY / VOC_ENERGY) ** self.power
            l2 = self.L1Loss(pred_mags_sq[:, 1], gt_mags_sq[:, 1])
            w2 = (BAS_ENERGY / DRU_ENERGY) ** self.power
            l11 = w1 * l1 + w2 * l2
            l3 = self.L1Loss(pred_mags_sq[:, 2], gt_mags_sq[:, 2])
            w3 = 1
            l4 = self.L1Loss(pred_mags_sq[:, 3], gt_mags_sq[:, 3])
            w4 = (BAS_ENERGY / OTH_ENERGY) ** self.power
            l22 = w3 * l3 + w4 * l4
            return [l1, l2, l3, l4, lg, l11 + l22 + lg]


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


class SpecChannelUnetLoss(torch.nn.Module):
    def __init__(self, main_device):
        super(SpecChannelUnetLoss, self).__init__()
        self.main_device = main_device
        self.L1Loss = torch.nn.L1Loss().to(main_device)

    def forward(self, x):
        gt_mags_sq, pred_mags_sq, _, _, _, _ = x
        l1 = self.L1Loss(pred_mags_sq[:, 0], gt_mags_sq[:, 0])
        l2 = self.L1Loss(pred_mags_sq[:, 1], gt_mags_sq[:, 1])
        if K == 4:
            l3 = self.L1Loss(pred_mags_sq[:, 2], gt_mags_sq[:, 2])
            l4 = self.L1Loss(pred_mags_sq[:, 3], gt_mags_sq[:, 3])
            return [l1, l2, l3, l4, w_1*l1+w_2*l2+w_3*l3+w_4*l4]
        return [l1, l2, w_1*l1+w_2*l2]


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
            w1 = (ACC_ENERGY / VOC_ENERGY) ** self.power
            l2 = self.L1Loss(pred_mags_sq[:, 1], gt_mags_sq[:, 1])
            w2 = 1
            l11 = w1 * l1 + w2 * l2
            return [l1, l2, l11]
        else:
            l1 = self.L1Loss(pred_mags_sq[:, 0], gt_mags_sq[:, 0])
            w1 = (BAS_ENERGY / VOC_ENERGY) ** self.power
            l2 = self.L1Loss(pred_mags_sq[:, 1], gt_mags_sq[:, 1])
            w2 = (BAS_ENERGY / DRU_ENERGY) ** self.power
            l11 = w1 * l1 + w2 * l2
            l3 = self.L1Loss(pred_mags_sq[:, 2], gt_mags_sq[:, 2])
            w3 = 1
            l4 = self.L1Loss(pred_mags_sq[:, 3], gt_mags_sq[:, 3])
            w4 = (BAS_ENERGY / OTH_ENERGY) ** self.power
            l22 = w3 * l3 + w4 * l4
            return [l1, l2, l3, l4, l11 + l22]


class EnergyBasedLossPowerPMask(torch.nn.Module):
    def __init__(self, main_device, power=1):
        super(EnergyBasedLossPowerPMask, self).__init__()
        self.main_device = main_device
        self.L1Loss = torch.nn.L1Loss().to(main_device)
        self.power = power

    def forward(self, x):
        _, _, _, _, gt_masks, pred_masks = x
        if K == 2:
            l1 = self.L1Loss(pred_masks[:, 0], gt_masks[:, 0])
            w1 = (ACC_ENERGY / VOC_ENERGY)**self.power
            l2 = self.L1Loss(pred_masks[:, 1], gt_masks[:, 1])
            w2 = 1
            l11 = w1 * l1 + w2 * l2
            return [l1, l2, l11]
        else:
            l1 = self.L1Loss(pred_masks[:, 0], gt_masks[:, 0])
            w1 = (BAS_ENERGY / VOC_ENERGY)**self.power
            l2 = self.L1Loss(pred_masks[:, 1], gt_masks[:, 1])
            w2 = (BAS_ENERGY / DRU_ENERGY)**self.power
            l11 = w1 * l1 + w2 * l2
            l3 = self.L1Loss(pred_masks[:, 2], gt_masks[:, 2])
            w3 = 1
            l4 = self.L1Loss(pred_masks[:, 3], gt_masks[:, 3])
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
