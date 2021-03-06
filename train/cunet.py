import sys

sys.path.append('..')

from dataset.dataloaders import CUnetInput
from flerken import pytorchfw
from models.cunet import CUNet
from flerken.framework.pytorchframework import set_training, config, ctx_iter
from flerken.framework import train, val
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.utils import *
from utils.EarlyStopping import EarlyStopping
from models.wrapper import CUNetWrapper
from tqdm import tqdm
from loss.losses import *
from settings import *


class CUNetTrain(pytorchfw):
    def __init__(self, model, rootdir, workname, main_device=0, trackgrad=False):
        super(CUNetTrain, self).__init__(model, rootdir, workname, main_device, trackgrad)
        self.audio_dumps_path = os.path.join(DUMPS_FOLDER, 'audio')
        self.visual_dumps_path = os.path.join(DUMPS_FOLDER, 'visuals')
        self.main_device=main_device
        self.grid_unwarp = torch.from_numpy(
            warpgrid(BATCH_SIZE, NFFT // 2 + 1, STFT_WIDTH, warp=False)).to(self.main_device)
        self.EarlyStopChecker = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
        self.val_iterations = 0

    def print_args(self):
        setup_logger('log_info', self.workdir + '/info_file.txt',
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
        self.criterion = CUNetLoss(self.main_device)

    @config
    @set_training
    def train(self):

        self.print_args()
        self.audio_dumps_folder = os.path.join(self.audio_dumps_path, self.workname, 'train')
        create_folder(self.audio_dumps_folder)
        self.visual_dumps_folder = os.path.join(self.visual_dumps_path, self.workname, 'train')
        create_folder(self.visual_dumps_folder)

        self.optimizer = self.set_optim(self.model.parameters(), momentum=MOMENTUM, lr=LR)
        training_data = CUnetInput('train')
        self.train_loader = torch.utils.data.DataLoader(training_data,
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=True,
                                                        num_workers=10)

        validation_data = CUnetInput('val')
        self.val_loader = torch.utils.data.DataLoader(validation_data,
                                                      batch_size=BATCH_SIZE,
                                                      shuffle=True,
                                                      num_workers=10)
        for self.epoch in range(self.start_epoch, self.EPOCHS):
            with train(self):
                self.run_epoch(self.train_iter_logger)
            self.scheduler.step(self.loss)
            with val(self):
                self.run_epoch()
            self.__update_db__()
            stop = self.EarlyStopChecker.check_improvement(self.loss_.data.tuple['val'].epoch_array.val,
                                                           self.epoch)
            if stop:
                print('Early Stopping Epoch : [{0}], '
                      'Best Checkpoint Epoch : [{1}]'.format(self.epoch,
                                                             self.EarlyStopChecker.best_epoch))
                break

    def train_epoch(self, logger):
        j = 0
        self.train_iterations = len(iter(self.train_loader))
        with tqdm(self.train_loader, desc='Epoch: [{0}/{1}]'.format(self.epoch, self.EPOCHS)) as pbar, ctx_iter(self):
            for inputs, visualization in pbar:
                try:
                    self.absolute_iter += 1
                    inputs = self._allocate_tensor(inputs)
                    output = self.model(inputs)
                    self.loss = self.criterion(output)
                    self.optimizer.zero_grad()
                    self.loss.backward()
                    self.gradients()
                    self.optimizer.step()
                    pbar.set_postfix(loss=self.loss)
                    self.loss_.data.print_logger(self.epoch, j, self.train_iterations, logger)
                    self.tensorboard_writer(self.loss, output, None, self.absolute_iter, visualization)
                    j += 1
                except Exception as e:
                    try:
                        self.save_checkpoint(filename=os.path.join(self.workdir, 'checkpoint_backup.pth'))
                    except:
                        self.err_logger.error('Failed to deal with exception. Could not save backup at {0} \n'
                                              .format(os.path.join(self.workdir, 'checkpoint_backup.pth')))
                    self.err_logger.error(str(e))
                    raise e
        for tsi in self.tensor_scalar_items:
            setattr(self, tsi, getattr(self, tsi + '_').data.update_epoch(self.state))
        self.tensorboard_writer(self.loss, output, None, self.absolute_iter, visualization)
        self.__update_db__()
        self.save_checkpoint()

    def validate_epoch(self):
        with tqdm(self.val_loader, desc='Validation: [{0}/{1}]'.format(self.epoch, self.EPOCHS)) as pbar, ctx_iter(
                self):
            for inputs, visualization in pbar:
                self.val_iterations += 1
                self.loss_.data.update_timed()
                inputs = self._allocate_tensor(inputs)
                output = self.model(inputs)
                self.loss = self.criterion(output)
                self.tensorboard_writer(self.loss, output, None, self.absolute_iter, visualization)
                pbar.set_postfix(loss=self.loss)
        for tsi in self.tensor_scalar_items:
            setattr(self, tsi, getattr(self, tsi + '_').data.update_epoch(self.state))
        self.tensorboard_writer(self.loss, output, None, self.absolute_iter, visualization)

    def tensorboard_writer(self, loss, output, gt, absolute_iter, visualization):
        if self.state == 'train':
            iter_val = absolute_iter
        elif self.state == 'val':
            iter_val = self.val_iterations

        if self.iterating:
            if iter_val % PARAMETER_SAVE_FREQUENCY == 0:
                text = visualization[1]
                self.writer.add_text('Filepath', text[-1], iter_val)
                phase = visualization[0].detach().cpu().clone().numpy()
                gt_mags_sq, pred_mags_sq, gt_mags, mix_mag, gt_masks, pred_masks = output
                if len(text) == BATCH_SIZE:
                    grid_unwarp = self.grid_unwarp
                else:  # for the last batch, where the number of samples are generally lesser than the batch_size
                    grid_unwarp = torch.from_numpy(
                        warpgrid(len(text), NFFT // 2 + 1, STFT_WIDTH, warp=False)).to(self.main_device)
                pred_masks_linear = linearize_log_freq_scale(pred_masks, grid_unwarp)
                gt_masks_linear = linearize_log_freq_scale(gt_masks, grid_unwarp)
                oracle_spec = (mix_mag * gt_masks_linear)
                pred_spec = (mix_mag * pred_masks_linear)
                j = 0
                for i, sample in enumerate(text):
                    sample_id = os.path.basename(sample)[:-4]
                    folder_name = os.path.basename(os.path.dirname(sample))
                    pred_audio_out_folder = os.path.join(self.audio_dumps_folder, folder_name, sample_id)
                    create_folder(pred_audio_out_folder)
                    visuals_out_folder = os.path.join(self.visual_dumps_folder, folder_name, sample_id)
                    create_folder(visuals_out_folder)

                    gt_audio = torch.from_numpy(
                        istft_reconstruction(gt_mags.detach().cpu().numpy()[i][j], phase[i][0], HOP_LENGTH))
                    pred_audio = torch.from_numpy(
                        istft_reconstruction(pred_spec.detach().cpu().numpy()[i][j], phase[i][0], HOP_LENGTH))
                    librosa.output.write_wav(os.path.join(pred_audio_out_folder, 'GT_TARGET.wav'),
                                             gt_audio.cpu().detach().numpy(), TARGET_SAMPLING_RATE)
                    librosa.output.write_wav(os.path.join(pred_audio_out_folder, 'PR_TARGET.wav'),
                                             pred_audio.cpu().detach().numpy(), TARGET_SAMPLING_RATE)

                    ### PLOTTING MAG SPECTROGRAMS ###
                    save_spectrogram(gt_mags[i][j].unsqueeze(0).detach().cpu(),
                                     os.path.join(visuals_out_folder, 'TARGET'), '_MAG_GT.png')
                    save_spectrogram(oracle_spec[i][j].unsqueeze(0).detach().cpu(),
                                     os.path.join(visuals_out_folder, 'TARGET'), '_MAG_ORACLE.png')
                    save_spectrogram(pred_spec[i][j].unsqueeze(0).detach().cpu(),
                                     os.path.join(visuals_out_folder, 'TARGET'), '_MAG_ESTIMATE.png')

                ### PLOTTING MAG SPECTROGRAMS ###
                plot_spectrogram(self.writer, gt_mags.detach().cpu().view(-1, 1, 512, 256)[:8],
                                 self.state + '_GT_MAG', iter_val)
                plot_spectrogram(self.writer, (pred_masks_linear * mix_mag).detach().cpu().view(-1, 1, 512, 256)[:8],
                                 self.state + '_PRED_MAG', iter_val)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

    # SET MODEL
    u_net = CUNet([32, 64, 128, 256, 512, 1024, 2048], 1, None, dropout=CUNET_DROPOUT)
    model = CUNetWrapper(u_net, main_device=MAIN_DEVICE)

    if not os.path.exists(ROOT_DIR):
        raise Exception('Directory does not exist')
    work = CUNetTrain(model, ROOT_DIR, PRETRAINED, main_device=MAIN_DEVICE, trackgrad=TRACKGRAD)
    work.model_version = 'Conditioned-U-NET'
    work.train()


if __name__ == '__main__':
    main()

# Usage python3 baseline.py --train/test
