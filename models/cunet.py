import sys

sys.path.append('..')

import torch
import torch.nn as nn
from numbers import Number
from warnings import warn
from settings import *


__all__ = ['CUNet']


def isnumber(x):
    return isinstance(x, Number)


def crop(img, i, j, h, w):
    """Crop the given Image.
    Args:
        img Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    return img[:, :, i:i + h, j:j + w]


def center_crop(img, output_size):
    """This function is prepared to crop tensors provided by dataloader.
    Cited tensors has shape [1,N_maps,H,W]
    """
    _, _, h, w = img.size()
    th, tw = output_size[0], output_size[1]
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)


class DenseBlock(nn.Module):
    def __init__(self, dropout=0.5, bn_momentum=0.1, **kwargs):
        super(DenseBlock, self).__init__()
        """Defines the dense block which generates complete set of gammas and betas for C-UNet for all the layers
        Args:

        Forward:
            Returns:
                gammas: gammas are to be added to specific layers of U-Net as a scaling factor
                betas: betas are to be added to specific layers as a multiplying factor before gammas are added
        """
        self.L1 = nn.Linear(4, 16)
        self.ReLu1 = nn.ReLU(0)
        self.L2 = nn.Linear(16, 128)
        self.ReLu2 = nn.ReLU(0)
        self.DO2 = nn.Dropout(p=dropout)
        self.BN2 = nn.BatchNorm1d(128, momentum=bn_momentum)
        self.L3 = nn.Linear(128, 1024)
        self.ReLu3 = nn.ReLU(0)
        self.DO3 = nn.Dropout(p=dropout)
        self.BN3 = nn.BatchNorm1d(1024, momentum=bn_momentum)
        self.L4 = nn.Linear(1024, 1008)  # 4064 = 32+64+128+256+512+1024+2048
        self.ReLu4 = nn.ReLU(0)

    def forward(self, c):
        c = self.L1(c)
        c = self.ReLu1(c)
        c = self.L2(c)
        c = self.ReLu2(c)
        c = self.DO2(c)
        c = self.BN2(c)
        c = self.L3(c)
        c = self.ReLu3(c)
        c = self.DO3(c)
        c = self.BN3(c)
        c = self.L4(c)
        c = self.ReLu4(c)
        return c


class ConvolutionalBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_conv=3, kernel_MP=2, stride_conv=1, stride_MP=2, padding=1,
                 bias=True, dropout=False, bn_momentum=0.1, **kwargs):
        super(ConvolutionalBlock, self).__init__()
        """Defines a (down)convolutional  block
        Args:
            dim_in: int dimension of feature maps of block input.
            dim_out: int dimension of feature maps of block output.
            kernel_conv: int or tuple kernel size for convolutions
            kernel_MP: int or tuple kernel size for Max Pooling
            stride_conv: int or tuple stride for convolutions
            stride_MP: int or tuple stride for Max Pooling
            padding: padding for convolutions
            bias: bool Set bias or not

        Forward:
            Returns:
                to_cat: output previous to Max Pooling for skip connections
                to_down: Max Pooling output to be used as input for next block
        """
        assert isinstance(dropout, Number)
        self.dropout = dropout
        self.Conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_conv, stride=stride_conv, padding=padding,
                               bias=bias)
        self.BN1 = nn.BatchNorm2d(dim_out, momentum=bn_momentum)
        self.ReLu1 = nn.LeakyReLU(0.1)
        self.Conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=kernel_conv, stride=stride_conv, padding=padding,
                               bias=bias)
        self.BN2 = nn.BatchNorm2d(dim_out, momentum=bn_momentum)
        self.ReLu2 = nn.LeakyReLU(0.1)
        self.MaxPooling = nn.MaxPool2d(kernel_size=kernel_MP, stride=stride_MP, padding=0, dilation=1,
                                       return_indices=False, ceil_mode=False)

    def forward(self, *args):
        x, gamma, beta = args
        x = self.Conv1(x)
        x = self.BN1(x)
        x = self.ReLu1(x)
        x = self.Conv2(x)
        x = self.BN2(x)
        # {(1,16,1,1) , (1,1,H,W)} >>> (1,16,H,W)
        gamma_tiled = gamma.unsqueeze(2).unsqueeze(2).repeat([1, 1, *x.shape[-2:]])
        beta_tiled = beta.unsqueeze(2).unsqueeze(2).repeat([1, 1, *x.shape[-2:]])
        x = gamma_tiled + (beta_tiled * x)
        # x = self.scale(c).unsqueeze(2).unsqueeze(2) * x + self.bias(c).unsqueeze(2).unsqueeze(2)
        to_cat = self.ReLu2(x)
        to_down = self.MaxPooling(to_cat)
        return to_cat, to_down


class AtrousBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_conv=3, kernel_UP=2, stride_conv=1, stride_UP=2, padding=1, bias=True,
                 finalblock=False, printing=False, bn_momentum=0.1, dropout=False, **kwargs):
        """Defines a upconvolutional  block
        Args:
            dim_in: int dimension of feature maps of block input.
            dim_out: int dimension of feature maps of block output.
            kernel_conv: int or tuple kernel size for convolutions
            kernel_MP: int or tuple kernel size for Max Pooling
            stride_conv: int or tuple stride for convolutions
            stride_MP: int or tuple stride for Max Pooling
            padding: padding for convolutions
            bias: bool Set bias or not
            finalblock: bool Set true if it's the last upconv block not to do upconvolution.
        Forward:
            Input:
                x: previous block input.
                to_cat: skip connection input.
            Returns:
                x: block output
        """
        super(AtrousBlock, self).__init__()
        self.finalblock = finalblock
        self.printing = printing
        self.dropout = dropout
        assert isinstance(dropout, Number)
        self.Conv1 = nn.Conv2d(2 * dim_in, dim_in, kernel_size=kernel_conv, stride=stride_conv, padding=padding,
                               bias=bias)
        self.BN1 = nn.BatchNorm2d(dim_in, momentum=bn_momentum)
        self.ReLu1 = nn.LeakyReLU(0.1)
        self.Conv2 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_conv, stride=stride_conv, padding=padding,
                               bias=bias)
        self.BN2 = nn.BatchNorm2d(dim_in, momentum=bn_momentum)
        self.ReLu2 = nn.LeakyReLU(0.1)
        if self.dropout:
            self.DO1 = nn.Dropout2d(self.dropout)
            self.DO2 = nn.Dropout2d(self.dropout)
        if not finalblock:
            self.AtrousConv = nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel_UP, stride=stride_UP, padding=0,
                                                 dilation=1)

    def forward(self, x, to_cat=None):
        if self.printing:
            print('Incoming variable from previous Upconv Block: {}'.format(x.size()))

        to_cat = center_crop(to_cat, x.size()[2:4])
        x = torch.cat((x, to_cat), dim=1)
        x = self.Conv1(x)
        x = self.BN1(x)
        if self.dropout:
            x = self.DO1(x)
        x = self.ReLu1(x)
        x = self.Conv2(x)
        x = self.BN2(x)
        x = self.ReLu2(x)
        if self.dropout:
            x = self.DO2(x)
        if not self.finalblock:
            x = self.AtrousConv(x)
        return x


class TransitionBlock(nn.Module):
    """Specific class for lowest block. Change values carefully.
        Args:
            dim_in: int dimension of feature maps of block input.
            dim_out: int dimension of feature maps of block output.
            kernel_conv: int or tuple kernel size for convolutions
            kernel_MP: int or tuple kernel size for Max Pooling
            stride_conv: int or tuple stride for convolutions
            stride_MP: int or tuple stride for Max Pooling
            padding: padding for convolutions
            bias: bool Set bias or not
        Forward:
            Input:
                x: previous block input.
            Returns:
                x: block output
    """

    def __init__(self, dim_in, dim_out, kernel_conv=3, kernel_UP=2, stride_conv=1, stride_UP=2, padding=1,
                 bias=True, bn_momentum=0.1, **kwargs):
        super(TransitionBlock, self).__init__()
        self.Conv1 = nn.Conv2d(int(dim_in / 2), dim_in, kernel_size=kernel_conv, stride=stride_conv,
                               padding=padding, bias=bias)
        self.BN1 = nn.BatchNorm2d(dim_in, momentum=bn_momentum)
        self.ReLu1 = nn.LeakyReLU(0.1)
        self.Conv2 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_conv, stride=stride_conv, padding=padding,
                               bias=bias)
        self.BN2 = nn.BatchNorm2d(dim_in, momentum=bn_momentum)
        self.ReLu2 = nn.LeakyReLU(0.1)
        self.AtrousConv = nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel_UP, stride=stride_UP, padding=0,
                                             dilation=1)

    def forward(self, *args):
        x, gamma, beta = args
        x = self.Conv1(x)
        x = self.BN1(x)
        x = self.ReLu1(x)
        x = self.Conv2(x)
        x = self.BN2(x)
        # {(1,16,1,1) , (1,1,H,W)} >>> (1,16,H,W)
        gamma_tiled = gamma.unsqueeze(2).unsqueeze(2).repeat([1, 1, *x.shape[-2:]])
        beta_tiled = beta.unsqueeze(2).unsqueeze(2).repeat([1, 1, *x.shape[-2:]])
        x = gamma_tiled + (beta_tiled * x)
        # x = self.scale(c).unsqueeze(2).unsqueeze(2) * x + self.bias(c).unsqueeze(2).unsqueeze(2)
        x = self.ReLu2(x)
        to_up = self.AtrousConv(x)

        return to_up


class CUNet(nn.Module):
    """It's recommended to be very careful  while managing vectors, since they are inverted to
    set top blocks as block 0. Notice there are N upconv blocks and N-1 downconv blocks as bottom block
    is considered as upconvblock.

    C-U-Net based on this paper https://arxiv.org/pdf/1904.05979.pdf
    """
    """
    Example:
    model = CUNet([64,128,256,512,1024,2048,4096],K,input_channels=1)

        K(int) : Amount of outgoing channels 
        input_channels (int): Amount of input channels
        dimension_vector (tuple/list): Its length defines amount of block. Elements define amount of filters per block



    """

    # TODO Use bilinear interpolation in addition to upconvolutions

    def __init__(self, dimensions_vector, K, verbose=False, input_channels=1,
                 activation=None, **kwargs):
        super(CUNet, self).__init__()
        self.K = K
        self.printing = verbose

        self.input_channels = input_channels
        self.dim = dimensions_vector
        self.init_assertion(**kwargs)

        self.vec = range(len(self.dim))
        self.gamma_generator = DenseBlock(**kwargs)
        self.beta_generator = DenseBlock(**kwargs)
        self.encoder = self.add_encoder(input_channels, **kwargs)
        self.decoder = self.add_decoder(**kwargs)

        self.activation = activation
        self.final_conv = nn.Conv2d(self.dim[0], self.K, kernel_size=1, stride=1, padding=0)
        if self.activation is not None:
            self.final_act = self.activation

    def init_assertion(self, **kwargs):
        assert isinstance(self.dim, (tuple, list))
        for x in self.dim:
            assert x % 2 == 0
        if list(map(lambda x: x / self.dim[0], self.dim)) != list(map(lambda x: 2 ** x, range(0, len(self.dim)))):
            raise ValueError('Dimension vector must double their channels sequentially ej. [16,32,64,128,...]')
        assert isinstance(self.input_channels, int)
        assert self.input_channels > 0
        assert isinstance(self.K, int)
        assert self.K > 0
        if kwargs.get('dropout') is not None:
            dropout = kwargs['dropout']
            assert isinstance(dropout, Number)
            assert dropout >= 0
            assert dropout <= 1
        if kwargs.get('bn_momentum') is not None:
            bn_momentum = kwargs['bn_momentum']
            assert isinstance(bn_momentum, Number)
            assert bn_momentum >= 0
            assert bn_momentum <= 1

    def add_encoder(self, input_channels, **kwargs):
        encoder = []
        for i in range(len(self.dim) - 1):  # There are len(self.dim)-1 downconv blocks
            if self.printing:
                print('Building Downconvolutional Block {} ...OK'.format(i))
            if i == 0:
                """SET 1 IF GRAYSCALE OR 3 IF RGB========================================"""
                encoder.append(ConvolutionalBlock(input_channels, self.dim[i], **kwargs))
            else:
                encoder.append(ConvolutionalBlock(self.dim[i - 1], self.dim[i], **kwargs))
        encoder = nn.Sequential(*encoder)
        return encoder

    def add_decoder(self, **kwargs):
        decoder = []
        for i in self.vec[::-1]:  # [::-1] inverts the order to set top layer as layer 0 and to order
            # layers from the bottom to above according to  flow of information.
            if self.printing:
                print('Building Upconvolutional Block {}...OK'.format(i))
            if i == max(self.vec):  # Special condition for lowest block
                decoder.append(
                    TransitionBlock(self.dim[i], self.dim[i - 1], **kwargs))
            elif i == 0:  # Special case for last (top) upconv block
                decoder.append(
                    AtrousBlock(self.dim[i], self.dim[i - 1], finalblock=True, **kwargs))
            else:
                decoder.append(AtrousBlock(self.dim[i], self.dim[i - 1], **kwargs))
        decoder = nn.Sequential(*decoder)
        return decoder

    def forward(self, *args):
        x, c = args
        gammas = self.gamma_generator(c)
        betas = self.beta_generator(c)
        init_index = 0
        if self.printing:
            print('UNet input size {0}'.format(x.size()))
        to_cat_vector = []
        for i in range(len(self.dim) - 1):
            if self.printing:
                print('Forward Prop through DownConv block {}'.format(i))
            end_index = init_index + self.dim[i]
            gamma = gammas[:, :, init_index:end_index]
            beta = betas[:, :, init_index:end_index]
            to_cat, x = self.encoder[i](x, gamma, beta)
            to_cat_vector.append(to_cat)
            init_index = end_index
        final_gamma = gammas[:, :, init_index:]
        final_beta = betas[:, :, init_index:]
        for i in self.vec:
            if self.printing:
                print('Concatenating and Building  UpConv Block {}'.format(i))
            if i == 0:
                x = self.decoder[i](x, final_gamma, final_beta)
            else:
                x = self.decoder[i](x, to_cat_vector[-i])
        x = self.final_conv(x)
        if self.activation is not None:
            x = self.final_act(x)
        if self.printing:
            print('UNet Output size {}'.format(x.size()))

        return x
