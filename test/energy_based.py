import sys
sys.path.append('..')
import shutil

from dataset.dataloaders import UnetInput
from flerken import pytorchfw
from flerken.models import UNet
from flerken.framework.pytorchframework import set_training, config, ctx_iter, \
    classitems,checkpoint_on_key,assert_workdir
from flerken.framework import train, val
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import *
from models.wrapper import Wrapper
from tqdm import tqdm
from loss.losses import *
from collections import  OrderedDict
from settings import *


class EnergyBased(pytorchfw):
    def __init__(self, model, rootdir, workname, main_device=0, trackgrad=False):
        super(EnergyBased, self).__init__(model, rootdir, workname, main_device, trackgrad)
        self.audio_dumps_path=os.path.join(DUMPS_FOLDER, 'audio')
        self.visual_dumps_path = os.path.join(DUMPS_FOLDER, 'visuals')
        self.audio_dumps_folder = os.path.join(self.audio_dumps_path, TEST_UNET_CONFIG, 'test')
        self.visual_dumps_folder = os.path.join(self.visual_dumps_path, TEST_UNET_CONFIG, 'test')
        self.grid_unwarp = torch.from_numpy(
            warpgrid(BATCH_SIZE, NFFT // 2 + 1, STFT_WIDTH, warp=False)).to('cuda')

        self.l1_ = classitems.TensorScalarItem()
        self.l2_ = classitems.TensorScalarItem()
        if K == 4:
            self.l3_ = classitems.TensorScalarItem()
            self.l4_ = classitems.TensorScalarItem()
        self.val_iterations = 0

    def print_args(self):
        setup_logger('log_info', self.workdir+'/info_file.txt',
                     FORMAT="[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s]")
        logger = logging.getLogger('log_info')
        self.print_info(logger)
        logger.info(f'\r\t Spectrogram data dir: {ROOT_DIR}\r'
                    'TRAINING PARAMETERS: \r\t'
                    f'Run name: {self.workname}\r\t'
                    f'Batch size {BATCH_SIZE} \r\t'
                    f'Optimizer {OPTIMIZER} \r\t'
                    f'Initializer {INITIALIZER} \r\t'
                    f'Epochs {EPOCHS} \r\t'
                    f'LR General: {LR} \r\t'
                    f'SGD Momentum {MOMENTUM} \r\t'
                    f'Weight Decay {WEIGHT_DECAY} \r\t'
                    f'Pre-trained model:  {PRETRAINED} \r'
                    'MODEL PARAMETERS \r\t'
                    f'Nº instruments (K) {K} \r\t'
                    f'U-Net activation: {ACTIVATION} \r\t'
                    f'U-Net Input channels {INPUT_CHANNELS}\r\t'
                    f'U-Net Batch normalization {USE_BN} \r\t')

    def set_optim(self, *args, **kwargs):
        if OPTIMIZER == 'adam':
            return torch.optim.Adam(*args, **kwargs)
        elif OPTIMIZER == 'SGD':
            return torch.optim.SGD(*args, **kwargs)
        else:
            raise Exception('Non considered optimizer. Implement it')

    def hyperparameters(self):
        self.dataparallel = False
        self.initializer = INITIALIZER
        self.EPOCHS = EPOCHS
        self.optimizer = self.set_optim(self.model.parameters(), momentum=MOMENTUM, lr=LR)
        self.LR = LR
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=7, threshold=3e-4)

    def set_config(self):
        self.batch_size = BATCH_SIZE
        self.criterion = EnergyBasedLoss(self.main_device)

    @config
    @set_training
    def train(self):

        self.print_args()
        validation_data = UnetInput('test')
        self.val_loader = torch.utils.data.DataLoader(validation_data,
                                                      batch_size=BATCH_SIZE,
                                                      shuffle=True,
                                                      num_workers=10)
        for self.epoch in range(self.start_epoch,self.EPOCHS):
            with val(self):
                self.run_epoch()
            break

    def validate_epoch(self):
        with tqdm(self.val_loader, desc='Validation: [{0}/{1}]'.format(self.epoch, self.EPOCHS)) as pbar, ctx_iter(
                self):
            for inputs, visualization in pbar:
                self.val_iterations += 1
                self.loss_.data.update_timed()
                inputs = self._allocate_tensor(inputs)
                output = self.model(*inputs) if isinstance(inputs, list) else self.model(inputs)
                self.loss_terms = self.criterion(output)
                if K == 2:
                    [self.l1, self.l2, self.loss] = self.loss_terms
                elif K == 4:
                    [self.l1, self.l2, self.l3, self.l4, self.loss] = self.loss_terms
                self.tensorboard_writer(self.loss, output, None, self.absolute_iter, visualization)
                pbar.set_postfix(loss=self.loss.item())
        self.loss = self.loss_.data.update_epoch(self.state)
        self.tensorboard_writer(self.loss, output, None, self.absolute_iter, visualization)

    def tensorboard_writer(self, loss, output, gt, absolute_iter, visualization):
        if self.iterating:
            self.l1_(self.l1, self.state)
            self.l2_(self.l2, self.state)
            if K == 4:
                self.l3_(self.l3, self.state)
                self.l4_(self.l4, self.state)

            text = visualization[1]
            self.writer.add_text('Filepath', text[-1], self.val_iterations)
            phase = visualization[0].detach().cpu().clone().numpy()
            gt_mags_sq, pred_mags_sq, gt_mags, mix_mag, gt_masks, pred_masks = output
            if len(text) == BATCH_SIZE:
                grid_unwarp = self.grid_unwarp
            else:  # for the last batch, where the number of samples are generally lesser than the batch_size
                grid_unwarp = torch.from_numpy(
                    warpgrid(len(text), NFFT // 2 + 1, STFT_WIDTH, warp=False)).to('cuda')
            pred_masks_linear = linearize_log_freq_scale(pred_masks, grid_unwarp)
            gt_masks_linear = linearize_log_freq_scale(gt_masks, grid_unwarp)
            oracle_spec = (mix_mag * gt_masks_linear)
            pred_spec = (mix_mag * pred_masks_linear)

            for i, sample in enumerate(text):
                sample_id = os.path.basename(sample)[:-4]
                folder_name = os.path.basename(os.path.dirname(sample))
                pred_audio_out_folder = os.path.join(self.audio_dumps_folder, folder_name, sample_id)
                create_folder(pred_audio_out_folder)
                visuals_out_folder = os.path.join(self.visual_dumps_folder, folder_name, sample_id)
                create_folder(visuals_out_folder)

                for j, source in enumerate(SOURCES_SUBSET):
                    gt_audio = torch.from_numpy(
                        istft_reconstruction(gt_mags.detach().cpu().numpy()[i][j], phase[i][0], HOP_LENGTH))
                    pred_audio = torch.from_numpy(
                        istft_reconstruction(pred_spec.detach().cpu().numpy()[i][j], phase[i][0], HOP_LENGTH))
                    librosa.output.write_wav(os.path.join(pred_audio_out_folder, 'GT_' + source + '.wav'),
                                             gt_audio.cpu().detach().numpy(), TARGET_SAMPLING_RATE)
                    librosa.output.write_wav(os.path.join(pred_audio_out_folder, 'PR_' + source + '.wav'),
                                             pred_audio.cpu().detach().numpy(), TARGET_SAMPLING_RATE)

                    ### PLOTTING MAG SPECTROGRAMS ###
                    save_spectrogram(gt_mags[i][j].unsqueeze(0).detach().cpu(),
                                     os.path.join(visuals_out_folder, source), '_MAG_GT.png')
                    save_spectrogram(oracle_spec[i][j].unsqueeze(0).detach().cpu(),
                                     os.path.join(visuals_out_folder, source), '_MAG_ORACLE.png')
                    save_spectrogram(pred_spec[i][j].unsqueeze(0).detach().cpu(),
                                     os.path.join(visuals_out_folder, source), '_MAG_ESTIMATE.png')

            ### PLOTTING MAG SPECTROGRAMS ###
            plot_spectrogram(self.writer, gt_mags.detach().cpu().view(-1, 1, 512, 256)[:8],
                             self.state + '_GT_MAG', self.val_iterations)
            plot_spectrogram(self.writer, (pred_masks_linear * mix_mag).detach().cpu().view(-1, 1, 512, 256)[:8],
                             self.state + '_PRED_MAG', self.val_iterations)

        else:
            self.l1 = self.l1_.data.update_epoch(self.state)
            self.l2 = self.l2_.data.update_epoch(self.state)
            if K == 4:
                self.l3 = self.l3_.data.update_epoch(self.state)
                self.l4 = self.l4_.data.update_epoch(self.state)

            if K == 2:
                self.writer.add_scalars(self.state + ' losses_epoch', {'Voice Est Loss': self.l1.item()}, self.epoch)
                self.writer.add_scalars(self.state + ' losses_epoch', {'Acc Est Loss': self.l2.item()}, self.epoch)
            elif K == 4:
                self.writer.add_scalars(self.state + ' losses_epoch', {'Voice Est Loss': self.l1.item()}, self.epoch)
                self.writer.add_scalars(self.state + ' losses_epoch', {'Drums Est Loss': self.l2.item()}, self.epoch)
                self.writer.add_scalars(self.state + ' losses_epoch', {'Bass Est Loss': self.l3.item()}, self.epoch)
                self.writer.add_scalars(self.state + ' losses_epoch', {'Other Est Loss': self.l4.item()}, self.epoch)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # SET MODEL
    u_net = UNet([32, 64, 128, 256, 512, 1024, 2048], K, None, dropout=DROPOUT, verbose=False, useBN=True)
    if not os.path.exists(ROOT_DIR):
        raise Exception('Directory does not exist')

    state_dict = torch.load(TEST_UNET_WEIGHTS_PATH, map_location=lambda storage, loc: storage)
    if 'checkpoint' in TEST_UNET_WEIGHTS_PATH:
        state_dict = state_dict['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k.replace('model.', '')
        new_state_dict[name] = v
    u_net.load_state_dict(new_state_dict, strict=True)
    model = Wrapper(u_net)

    work = EnergyBased(model, ROOT_DIR, PRETRAINED, trackgrad=TRACKGRAD)
    work.model_version = 'ENERGY_BASED_TESTING'
    work.train()


if __name__ == '__main__':
    main()

# Usage python3 energy_based.py --train/test
