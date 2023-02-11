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
    return jnp.array(pywt.Wavelet(wavelet).filter_bank, dtype)


def make_kernel(lo, hi):
    """Make a 2D convolution kernel from 1D lowpass and highpass filters."""
    lo = jnp.flip(lo)
    hi = jnp.flip(hi)
    ll = jnp.outer(lo, lo)
    lh = jnp.outer(hi, lo)
    hl = jnp.outer(lo, hi)
    hh = jnp.outer(hi, hi)
    kernel = jnp.stack([ll, lh, hl, hh], 0)
    kernel = jnp.expand_dims(kernel, 1)
    return kernel


def wavelet_dec_once(x, filt, channels):
    """Do one level of the DWT."""
    low, high = x[..., :channels], x[..., channels:]
    k = make_kernel(filt[0], filt[1])

    if jax.default_backend() == "tpu":
        kernel = jnp.zeros((channels * 4, channels, k.shape[2], k.shape[3]), k.dtype)
        for o in range(channels):
            for i in range(4):
                kernel = kernel.at[i + o * 4, o].set(k[i, 0])
        groups = 1
    else:
        kernel = jnp.tile(k, [channels, 1, 1, 1])
        groups = channels

    n = kernel.shape[-1] - 1
    lo, hi = n // 2, n // 2 + n % 2
    low = jnp.pad(low, ((0, 0), (lo, hi), (lo, hi), (0, 0)), "wrap")
    low = jax.lax.conv_general_dilated(
        lhs=low,
        rhs=kernel,
        window_strides=(2, 2),
        padding=((0, 0), (0, 0)),
        dimension_numbers=("NHWC", "OIHW", "NHWC"),
        feature_group_count=groups,
    )
    low = rearrange(low, "n h w (c1 c2) -> n h w (c2 c1)", c2=4)
    high = rearrange(
        high, "n (h h2) (w w2) (c c2) -> n h w (c h2 w2 c2)", h2=2, w2=2, c2=channels
    )
    x = jnp.concatenate([low, high], axis=-1)
    return x


def wavelet_rec_once(x, filt, channels):
    """Do one level of the IDWT."""
    low, high = x[..., : channels * 4], x[..., channels * 4 :]
    k = make_kernel(filt[2], filt[3])

    if jax.default_backend() == "tpu":
        kernel = jnp.zeros((channels * 4, channels, k.shape[2], k.shape[3]), k.dtype)
        for o in range(channels):
            for i in range(4):
                kernel = kernel.at[i + o * 4, o].set(k[i, 0])
        groups = 1
    else:
        kernel = jnp.tile(k, [1, channels, 1, 1])
        groups = channels

    n = kernel.shape[-1]
    lo, hi = n // 2 + n % 2, n // 2
    lo_pre, hi_pre = lo // 2 + lo % 2, lo // 2
    lo_post, hi_post = lo_pre * 2, hi_pre * 2
    low = rearrange(low, "n h w (c1 c2) -> n h w (c2 c1)", c1=4)
    low = jnp.pad(low, ((0, 0), (lo_pre, hi_pre), (lo_pre, hi_pre), (0, 0)), "wrap")
    low = jax.lax.conv_general_dilated(
        lhs=low,
        rhs=kernel,
        window_strides=(1, 1),
        padding=((lo, hi), (lo, hi)),
        lhs_dilation=(2, 2),
        dimension_numbers=("NHWC", "IOHW", "NHWC"),
        feature_group_count=groups,
    )
    low = low[:, lo_post:-hi_post, lo_post:-hi_post, :]
    high = rearrange(
        high, "n h w (c h2 w2 c2) -> n (h h2) (w w2) (c c2)", h2=2, w2=2, c2=channels
    )
    x = jnp.concatenate([low, high], axis=-1)
    return x


def wavelet_dec(x, filt, levels):
    """Do the DWT for a given number of levels.

    Args:
        x: Input image (NHWC layout).
        filt: Filter bank.
        levels: Number of levels.

    Returns:
        The DWT coefficients, with shape
        (N, H // 2 ** levels, W // 2 ** levels, C * 4 ** levels).
    """
    channels = x.shape[-1]
    for i in range(levels):
        x = wavelet_dec_once(x, filt, channels)
    return x


def wavelet_rec(x, filt, levels):
    """Do the IDWT for a given number of levels.

    Args:
        x: Input array of IDWT coefficients.
        filt: Filter bank.
        levels: Number of levels.

    Returns:
        The IDWT coefficients, with shape
        (N, H * 2 ** levels, W * 2 ** levels, C // 4 ** levels).
    """
    channels = x.shape[-1] // 4**levels
    for i in reversed(range(levels)):
        x = wavelet_rec_once(x, filt, channels)
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
