import torch
from utils import warpgrid, linearize_log_freq_scale
from settings import *


class SingleSourceDirectLoss(torch.nn.Module):
    def __init__(self, main_device, grid_unwarp):
        super(SingleSourceDirectLoss, self).__init__()
        self.main_device = main_device
        self.L1Loss = torch.nn.L1Loss().to(main_device)
        self.grid_unwarp = grid_unwarp

    def forward(self, x):
        gt_mags, mix_mag, gt_masks, pred_masks = x
        # loss = self.L1Loss(pred_masks[:, 0], gt_masks[:, ISOLATED_SOURCE_ID])

        if gt_mags.shape[0] == BATCH_SIZE:
            grid_unwarp = self.grid_unwarp
        else:  # for the last batch, where the number of samples are generally lesser than the batch_size
            grid_unwarp = torch.from_numpy(
                warpgrid(gt_mags.shape[0], NFFT // 2 + 1, STFT_WIDTH, warp=False)).to('cuda')

        pred_masks_linear = linearize_log_freq_scale(pred_masks, grid_unwarp)
        pred_mags = mix_mag * pred_masks_linear
        loss = self.L1Loss(pred_mags[:, 0], gt_mags[:, ISOLATED_SOURCE_ID])
        return loss


class IndividualLosses(torch.nn.Module):
    def __init__(self, main_device, grid_unwarp):
        super(IndividualLosses, self).__init__()
        self.main_device = main_device
        self.L1Loss = torch.nn.L1Loss().to(main_device)
        self.grid_unwarp = grid_unwarp

    def forward(self, x):
        gt_mags, mix_mag, gt_masks, pred_masks = x
        if gt_mags.shape[0] == BATCH_SIZE:
            grid_unwarp = self.grid_unwarp
        else:  # for the last batch, where the number of samples are generally lesser than the batch_size
            grid_unwarp = torch.from_numpy(
                warpgrid(gt_mags.shape[0], NFFT // 2 + 1, STFT_WIDTH, warp=False)).to('cuda')

        pred_masks_linear = linearize_log_freq_scale(pred_masks, grid_unwarp)
        pred_mags = mix_mag * pred_masks_linear
        # l1 = self.L1Loss(pred_masks[:, 0], gt_masks[:, 0])
        l1 = self.L1Loss(pred_mags[:, 0], gt_mags[:, 0])
        # l2 = self.L1Loss(pred_masks[:, 1], gt_masks[:, 1])
        l2 = self.L1Loss(pred_mags[:, 1], gt_mags[:, 1])
        if K == 4:
            # l3 = self.L1Loss(pred_masks[:, 2], gt_masks[:, 2])
            l3 = self.L1Loss(pred_mags[:, 2], gt_mags[:, 2])
            # l4 = self.L1Loss(pred_masks[:, 3], gt_masks[:, 3])
            l4 = self.L1Loss(pred_mags[:, 3], gt_mags[:, 3])
            return [l1, l2, l3, l4]
        return [l1, l2]


class EnergyBasedLoss(torch.nn.Module):
    def __init__(self, main_device, grid_unwarp):
        super(EnergyBasedLoss, self).__init__()
        self.main_device = main_device
        self.L1Loss = torch.nn.L1Loss().to(main_device)
        self.loss_terms = torch.empty([len(SOURCES_SUBSET)])
        self.weights = torch.empty([len(SOURCES_SUBSET)])
        self.grid_unwarp = grid_unwarp

    def forward(self, x):
        gt_mags, mix_mag, gt_masks, pred_masks = x
        if gt_mags.shape[0] == BATCH_SIZE:
            grid_unwarp = self.grid_unwarp
        else:  # for the last batch, where the number of samples are generally lesser than the batch_size
            grid_unwarp = torch.from_numpy(
                warpgrid(gt_mags.shape[0], NFFT // 2 + 1, STFT_WIDTH, warp=False)).to('cuda')

        pred_masks_linear = linearize_log_freq_scale(pred_masks, grid_unwarp)
        pred_mags = mix_mag * pred_masks_linear
        # l1 = self.L1Loss(pred_masks[:, 0], gt_masks[:, 0])
        l1 = self.L1Loss(pred_mags[:, 0], gt_mags[:, 0])
        w1 = (255 / 175) ** 2  # pow(gt_mags[:, 0].pow(2).sum(), 1)
        # l2 = self.L1Loss(pred_masks[:, 1], gt_masks[:, 1])
        l2 = self.L1Loss(pred_mags[:, 1], gt_mags[:, 1])
        w2 = (255 / 220) ** 2  # pow(gt_mags[:, 1].pow(2).sum(), 1)
        l11 = w1 * l1 + w2 * l2
        if K == 4:
            # l3 = self.L1Loss(pred_masks[:, 2], gt_masks[:, 2])
            l3 = self.L1Loss(pred_mags[:, 2], gt_mags[:, 2])
            w3 = 1  # pow(gt_mags[:, 2].pow(2).sum(), 1)
            # l4 = self.L1Loss(pred_masks[:, 3], gt_masks[:, 3])
            l4 = self.L1Loss(pred_mags[:, 3], gt_mags[:, 3])
            w4 = (255 / 218) ** 2  # pow(gt_mags[:, 3].pow(2).sum(), 1)
            l22 = w3 * l3 + w4 * l4
            return [l1, l2, l3, l4, l11 + l22]
        return [l1, l2, l11]
