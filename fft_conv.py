# Cool dependencies
from typing import Tuple, Union, Iterable
import warnings
from functools import partial
from math import floor, ceil

# Import dependencies
from torch.fft import fftn, ifftn, ifftshift, fftshift, rfftn, irfftn
from torch import mul, abs, nn, Tensor
from torch.nn import functional as f
import gc

## Defining FFT Convolution Definition
def fftconv(
    signal: Tensor,
    kernel: Tensor,
    bias: Tensor = None,
    padding: Union[int, Iterable[int], str] = 0,
    padding_mode: str = "constant",
    stride: Union[int, Iterable[int]] = 1,
    dilation: Union[int, Iterable[int]] = 1,
    groups: int = 1,
) -> Tensor:
    n = signal.ndim - 2
    if type(stride) == int:
        stride = (stride,) * n
    if type(stride[0]) != int:
        # Make sure tuple only contains integers
        kernel_size = tuple(int(k) for k in kernel_size)
    if type(dilation) == int:
        dilation = (dilation,) * n
    if type(dilation[0]) != int:
        # Make sure tuple only contains integers
        dilation = tuple(int(d) for d in dilation)

    if isinstance(padding, str):
        if padding == "same":
            if (1 not in stride) or (1 not in dilation):
                raise ValueError("Stride and dilation must be 1 for padding='same'.")
            padding_ = [(k - 1) / 2 for k in kernel.shape[2:]]
        else:
            raise ValueError(f"Padding mode {padding} not supported.")
    else:
        if type(padding) == int:
            padding = (padding,) * n
        if type(padding[0]) != int:
            # Make sure tuple only contains integers
            padding = tuple(int(p) for p in padding)
    
    #internal dilation offsets
    offset = torch.zeros(1, 1, *dilation, device=signal.device, dtype=signal.dtype)
    offset[(slice(None), slice(None), *((0,) * n))] = 1.0

    # correct the kernel by cutting off unwanted dilation trailing zeros
    cutoff = tuple(slice(None, -d + 1 if d != 1 else None) for d in dilation)

    # pad the kernel internally according to the dilation parameters
    kernel = torch.kron(kernel, offset)[(slice(None), slice(None)) + cutoff]

    # Pad the input signal & kernel tensors (round to support even sized convolutions)
    signal_padding = [r(p) for p in padding_[::-1] for r in (floor, ceil)]
    signal = f.pad(signal, signal_padding, mode=padding_mode)

    # Because PyTorch computes a *one-sided* FFT, we need the final dimension to
    # have *even* length.  Just pad with one more zero if the final dimension is odd.
    signal_size = signal.size()  # original signal size without padding to even
    if signal.size(-1) % 2 != 0:
        signal = f.pad(signal, [0, 1])

    kernel_padding = [
        pad
        for i in reversed(range(2, signal.ndim))
        for pad in [0, signal.size(i) - kernel.size(i)]
    ]
    padded_kernel = f.pad(kernel, kernel_padding)

    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    signal_fr = rfftn(signal.float(), dim=tuple(range(2, signal.ndim)))
    kernel_fr = rfftn(padded_kernel.float(), dim=tuple(range(2, signal.ndim)))

    kernel_fr.imag *= -1
    output_fr = torch.mul(signal_fr, kernel_fr)
    output = irfftn(output_fr, dim=tuple(range(2, signal.ndim)))

    # Remove extra padded values
    crop_slices = [slice(None), slice(None)] + [
        slice(0, (signal_size[i] - kernel.size(i) + 1), stride[i - 2])
        for i in range(2, signal.ndim)
    ]
    output = output[crop_slices].contiguous()

    # Optionally, add a bias term before returning.
    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        output += bias.view(bias_shape)

    return output

# Defining FFT Convolution PyTorch Class
class _fftconv(nn.Module):
    
    '''Base class for FFT-based convolutional layers.'''
    
    def __init__(self, in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Iterable[int]],
        padding: Union[int, Iterable[int]] = 0,
        padding_mode: str = "constant",
        stride: Union[int, Iterable[int]] = 1,
        dilation: Union[int, Iterable[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        ndim: int = 1,):
        
        """_summary_
        """
        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.padding_mode = padding_mode
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        
        ### Error handling is exactly the same as torch.nn.functional.convNd.
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        valid_padding_strings = {'same', 'valid'}

        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    f"Invalid padding string {padding!r}, should be one of {valid_padding_strings}")
            if padding == 'same' and stride != 1:
                raise ValueError("padding='same' is not supported for strided convolutions")
            
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
            
        ### End PyTorch Error Handling
        
        if type(kernel_size) == int:
            kernel_size = (kernel_size,) * ndim
        if type(kernel_size[0]) != int:
            # Make sure tuple only contains integers
            kernel_size = tuple(int(k) for k in kernel_size)

        weight = torch.randn(out_channels, in_channels // groups, *kernel_size)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        
    def forward(self, signal):
        return fftconv(signal, self.weight, bias=self.bias, padding=self.padding, padding_mode=self.padding_mode, stride=self.stride, dilation=self.dilation, groups=self.groups,)

FFT_Conv1d = partial(_fftconv, ndim=1)
FFT_Conv2d = partial(_fftconv, ndim=2)
FFT_Conv3d = partial(_fftconv, ndim=3)