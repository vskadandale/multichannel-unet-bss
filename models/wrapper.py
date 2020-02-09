import torch
from utils.utils import warpgrid
import torch.nn.functional as F
from settings import *


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.L = len(SOURCES_SUBSET)
        self.model = model
        self.grid_warp = torch.from_numpy(
            warpgrid(BATCH_SIZE, 256, STFT_WIDTH, warp=True)).to('cuda')

    def forward(self, x):
        if x.shape[0] == BATCH_SIZE:
            mags = F.grid_sample(x, self.grid_warp)
        else:  # for the last batch, where the number of samples are generally lesser than the batch_size
            custom_grid_warp = torch.from_numpy(
                warpgrid(x.shape[0], 256, STFT_WIDTH, warp=True)).to('cuda')
            mags = F.grid_sample(x, custom_grid_warp)

        gt_masks = torch.div(mags[:, :-1], mags[:, -1].unsqueeze(1).expand(x.shape[0], self.L, *mags.shape[2:]))
        gt_masks.clamp_(0., 10.)

        log_mags = torch.log(mags[:, -1].unsqueeze(1)).detach()
        gt_mags = x[:, :-1]
        mix_mag = x[:, -1].unsqueeze(1)
        pred_masks = self.model(log_mags)
        pred_masks = torch.relu(pred_masks)
        mag_mix_sq = mags[:, -1].unsqueeze(1)
        pred_mags_sq = pred_masks * mag_mix_sq
        gt_mags_sq = gt_masks * mag_mix_sq

        network_output = [gt_mags_sq, pred_mags_sq, gt_mags, mix_mag, gt_masks,
                          pred_masks]  # BxKx256x256, BxKx256x256, BxKx512x256, Bx1x512x256, BxKx256x256, BxKx256x256
        return network_output
