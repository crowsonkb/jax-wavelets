"""The 2D discrete wavelet transform for JAX."""

from einops import rearrange
import jax
import jax.numpy as jnp
import pywt


def get_filter_bank(wavelet, dtype=jnp.float32):
    """Get the filter bank for a given pywavelets wavelet name. See
    https://wavelets.pybytes.com for a list of available wavelets.

    Args:
        wavelet: Name of the wavelet.
        dtype: dtype to cast the filter bank to.

    Returns:
        A JAX array containing the filter bank.
    """
    filt = jnp.array(pywt.Wavelet(wavelet).filter_bank, dtype)
    # Special case for some bior family wavelets
    if jnp.all(filt[:, 0] == 0):
        filt = filt[:, 1:]
    return filt


def make_2d_kernel(lo, hi):
    """Make a 2D convolution kernel from 1D lowpass and highpass filters."""
    lo = jnp.flip(lo)
    hi = jnp.flip(hi)
    ll = jnp.outer(lo, lo)
    lh = jnp.outer(hi, lo)
    hl = jnp.outer(lo, hi)
    hh = jnp.outer(hi, hi)
    kernel = jnp.stack([ll, lh, hl, hh])[:, None]
    return kernel


def make_kernels(filter, channels):
    """Precompute the convolution kernels for the DWT and IDWT for a given number of
    channels.

    Args:
        filter: A JAX array containing the filter bank.
        channels: The number of channels in the input image.

    Returns:
        A tuple of JAX arrays containing the convolution kernels for the DWT and IDWT.
    """
    kernel = jnp.zeros(
        (channels * 4, channels, filter.shape[1], filter.shape[1]), filter.dtype
    )
    index_i = jnp.repeat(jnp.arange(4), channels)
    index_j = jnp.tile(jnp.arange(channels), 4)
    k_dec = make_2d_kernel(filter[0], filter[1])
    k_rec = make_2d_kernel(filter[2], filter[3])
    kernel_dec = kernel.at[index_i * channels + index_j, index_j].set(k_dec[index_i, 0])
    kernel_rec = kernel.at[index_i * channels + index_j, index_j].set(k_rec[index_i, 0])
    kernel_rec = jnp.swapaxes(kernel_rec, 0, 1)
    return kernel_dec, kernel_rec


def wavelet_dec_once(x, kernel):
    """Do one level of the DWT."""
    channels = kernel.shape[1]
    low, high = x[..., :channels], x[..., channels:]

    n = kernel.shape[-1] - 1
    lo, hi = n // 2, n // 2 + n % 2
    low = jnp.pad(low, ((0, 0), (lo, hi), (lo, hi), (0, 0)), "wrap")
    low = jax.lax.conv_general_dilated(
        lhs=low,
        rhs=kernel,
        window_strides=(2, 2),
        padding=((0, 0), (0, 0)),
        dimension_numbers=("NHWC", "OIHW", "NHWC"),
    )
    high = rearrange(
        high, "n (h h2) (w w2) (c c2) -> n h w (c h2 w2 c2)", h2=2, w2=2, c2=channels
    )
    x = jnp.concatenate([low, high], axis=-1)
    return x


def wavelet_rec_once(x, kernel):
    """Do one level of the IDWT."""
    channels = kernel.shape[0]
    low, high = x[..., : channels * 4], x[..., channels * 4 :]

    n = kernel.shape[-1]
    lo, hi = n // 2 + n % 2, n // 2
    low = jnp.pad(low, ((0, 0), (lo, hi), (lo, hi), (0, 0)), "wrap")
    low = jax.lax.conv_general_dilated(
        lhs=low,
        rhs=kernel,
        window_strides=(1, 1),
        padding=((0, 0), (0, 0)),
        lhs_dilation=(2, 2),
        dimension_numbers=("NHWC", "OIHW", "NHWC"),
    )
    low = low[:, lo:-hi, lo:-hi, :]
    high = rearrange(
        high, "n h w (c h2 w2 c2) -> n (h h2) (w w2) (c c2)", h2=2, w2=2, c2=channels
    )
    x = jnp.concatenate([low, high], axis=-1)
    return x


def wavelet_dec(x, kernel, levels):
    """Do the DWT for a given number of levels.

    Args:
        x: Input image (NHWC layout).
        kernel: Decomposition kernel.
        levels: Number of levels.

    Returns:
        The DWT coefficients, with shape
        (N, H // 2 ** levels, W // 2 ** levels, C * 4 ** levels).
    """
    for i in range(levels):
        x = wavelet_dec_once(x, kernel)
    return x


def wavelet_rec(x, kernel, levels):
    """Do the IDWT for a given number of levels.

    Args:
        x: Input array of IDWT coefficients.
        kernel: Reconstruction kernel.
        levels: Number of levels.

    Returns:
        The IDWT coefficients, with shape
        (N, H * 2 ** levels, W * 2 ** levels, C // 4 ** levels).
    """
    for i in reversed(range(levels)):
        x = wavelet_rec_once(x, kernel)
    return x


def unpack(x, levels):
    """Unpack the DWT coefficients into a pywavelets wavedec2() coefficients list.

    Args:
        x: Input array of DWT coefficients.
        levels: Number of levels.

    Returns:
        A pywavelets wavedec2() coefficients list.
    """
    channels = x.shape[-1] // 4**levels
    y = [x[..., :channels]]
    for i in range(levels):
        y_cur = x[..., channels * 4**i : channels * 4 ** (i + 1)]
        for j in range(i):
            y_cur = rearrange(
                y_cur,
                "n h w (c h2 w2 c2) -> n (h h2) (w w2) (c c2)",
                h2=2,
                w2=2,
                c2=channels,
            )
        y.append(tuple(jnp.split(y_cur, 3, axis=-1)))
    return y


def pack(x):
    """Pack the pywavelets wavedec2() coefficients list into a DWT coefficients array.

    Args:
        x: Input pywavelets wavedec2() coefficients list.

    Returns:
        A DWT coefficients array.
    """
    y = x[0]
    for i in range(len(x) - 1):
        y_cur = jnp.concatenate(x[i + 1], axis=-1)
        for j in range(i):
            y_cur = rearrange(
                y_cur,
                "n (h h2) (w w2) (c c2) -> n h w (c h2 w2 c2)",
                h2=2,
                w2=2,
                c2=x[0].shape[-1],
            )
        y = jnp.concatenate([y, y_cur], axis=-1)
    return y
