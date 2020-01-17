import torch
import torch.nn.functional as F
import logging
import numpy as np
import librosa
import torchvision
import six
from settings import *


def setup_logger(logger_name, log_file, level=logging.INFO, FORMAT='%(message)s', mode='w'):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(FORMAT)
    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(file_handler)
    l.addHandler(stream_handler)


def create_folder(path):
    if not os.path.exists(path):
        os.umask(0)                # To mask the permission restrictions on new files/directories being create
        os.makedirs(path, 0o755)   # setting permissions for the folder


def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units

    This computes the scaling ``10 * log10(S / ref)`` in a numerically
    stable way.

    Parameters
    ----------
    S : np.ndarray
        input power

    ref : scalar or callable
        If scalar, the amplitude `abs(S)` is scaled relative to `ref`:
        `10 * log10(S / ref)`.
        Zeros in the output correspond to positions where `S == ref`.

        If callable, the reference value is computed as `ref(S)`.

    amin : float > 0 [scalar]
        minimum threshold for `abs(S)` and `ref`

    top_db : float >= 0 [scalar]
        threshold the output at `top_db` below the peak:
        ``max(10 * log10(S)) - top_db``

    Returns
    -------
    S_db   : np.ndarray
        ``S_db ~= 10 * log10(S) - 10 * log10(ref)``

    See Also
    --------
    perceptual_weighting
    db_to_power
    amplitude_to_db
    db_to_amplitude

    Notes
    -----
    This function caches at level 30.


    Examples
    --------
    Get a power spectrogram from a waveform ``y``

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = np.abs(librosa.stft(y))
    >>> librosa.power_to_db(S**2)
    array([[-33.293, -27.32 , ..., -33.293, -33.293],
           [-33.293, -25.723, ..., -33.293, -33.293],
           ...,
           [-33.293, -33.293, ..., -33.293, -33.293],
           [-33.293, -33.293, ..., -33.293, -33.293]], dtype=float32)

    Compute dB relative to peak power

    >>> librosa.power_to_db(S**2, ref=np.max)
    array([[-80.   , -74.027, ..., -80.   , -80.   ],
           [-80.   , -72.431, ..., -80.   , -80.   ],
           ...,
           [-80.   , -80.   , ..., -80.   , -80.   ],
           [-80.   , -80.   , ..., -80.   , -80.   ]], dtype=float32)


    Or compare to median power

    >>> librosa.power_to_db(S**2, ref=np.median)
    array([[-0.189,  5.784, ..., -0.189, -0.189],
           [-0.189,  7.381, ..., -0.189, -0.189],
           ...,
           [-0.189, -0.189, ..., -0.189, -0.189],
           [-0.189, -0.189, ..., -0.189, -0.189]], dtype=float32)


    And plot the results

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(2, 1, 1)
    >>> librosa.display.specshow(S**2, sr=sr, y_axis='log')
    >>> plt.colorbar()
    >>> plt.title('Power spectrogram')
    >>> plt.subplot(2, 1, 2)
    >>> librosa.display.specshow(librosa.power_to_db(S**2, ref=np.max),
    ...                          sr=sr, y_axis='log', x_axis='time')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Log-Power spectrogram')
    >>> plt.tight_layout()

    """

    if amin <= 0:
        raise Exception('amin must be strictly positive')

    amin = torch.tensor(amin)

    if six.callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(S)
    else:
        ref_value = ref.clone().detach()
    log_spec = 10 * torch.max(S, amin.clone().detach().expand(S.size())).log10()
    log_spec = log_spec - 10 * torch.max(amin, ref_value).log10()

    if top_db is not None:
        if top_db < 0:
            raise Exception('top_db must be non-negative')
        log_spec = torch.max(log_spec, (log_spec.max() - top_db).expand(log_spec.size()))

    return log_spec


def amplitude_to_db(S, ref=1.0, amin=1e-5, top_db=80.0):
    '''Convert an amplitude spectrogram to dB-scaled spectrogram.

    This is equivalent to ``power_to_db(S**2)``, but is provided for convenience.

    Parameters
    ----------
    S : np.ndarray
        input amplitude

    ref : scalar or callable
        If scalar, the amplitude `abs(S)` is scaled relative to `ref`:
        `20 * log10(S / ref)`.
        Zeros in the output correspond to positions where `S == ref`.

        If callable, the reference value is computed as `ref(S)`.

    amin : float > 0 [scalar]
        minimum threshold for `S` and `ref`

    top_db : float >= 0 [scalar]
        threshold the output at `top_db` below the peak:
        ``max(20 * log10(S)) - top_db``


    Returns
    -------
    S_db : np.ndarray
        ``S`` measured in dB

    See Also
    --------
    power_to_db, db_to_amplitude

    Notes
    -----
    This function caches at level 30.
    '''



    if six.callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(S)
    else:
        ref_value = torch.tensor(abs(ref))

    power = S.pow(2)

    return power_to_db(power, ref=ref_value**2, amin=amin**2,top_db=top_db)

def rescale(x,max_range,min_range):
    max_val = torch.max(x)
    min_val = torch.min(x)
    return (max_range-min_range)/(max_val-min_val)*(x-max_val)+max_range


def warpgrid(bs, h, w, warp=True):
    ### What is the role? Bends uniform ramp. Ground truth masks to be computed ONLY after warping!
    # ab = np.linspace(-1, 1, 256)
    # plt.plot(ab)
    # plt.show()
    # grid_warp = torch.from_numpy(warpgrid(1, 256,256, warp=True))
    # plt.plot(grid_warp[0,:,0,1].cpu().detach().numpy())
    # plt.show()

    # meshgrid
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xv, yv = np.meshgrid(x, y)
    grid = np.zeros((bs, h, w, 2))
    grid_x = xv
    if warp:
        grid_y = (np.power(21, (yv+1)/2) - 11) / 10
    else:
        grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
    grid[:, :, :, 0] = grid_x
    grid[:, :, :, 1] = grid_y
    grid = grid.astype(np.float32)
    return grid


def istft_reconstruction(mag, phase, hop_length=256):
    spec = mag.astype(np.complex) * np.exp(1j*phase)
    wav = librosa.istft(spec, hop_length=hop_length)
    return np.clip(wav, -1., 1.)


def linearize_log_freq_scale(nonlinear_vec,grid_unwarp):
    linear_vec = F.grid_sample(nonlinear_vec, grid_unwarp)
    return linear_vec


def plot_spectrogram(writer, spectrogram, identifier, iter_val):
    spectrogram_db = amplitude_to_db(spectrogram, ref=torch.max)
    spectrogram_db= rescale(spectrogram_db, min_range=0, max_range=1)
    x = torchvision.utils.make_grid(spectrogram_db[:8].detach().cpu(), nrow=4)
    writer.add_images(identifier, x.unsqueeze(0), iter_val)


def save_spectrogram(spectrogram,path, identifier):
    spectrogram_db = amplitude_to_db(spectrogram, ref=torch.max)
    spectrogram_db= rescale(spectrogram_db, min_range=0, max_range=1)
    img = torchvision.transforms.ToPILImage()(spectrogram_db)
    img.save(path + identifier)
