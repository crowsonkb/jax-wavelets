"""A benchmark and test cases for jax-wavelets."""

import argparse
from functools import partial
import time

import jax
import jax.numpy as jnp
import pywt

from . import *


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--batch-size", "-bs", type=int, default=1, help="the batch size"
    )
    parser.add_argument(
        "--channels", "-c", type=int, default=3, help="the number of channels"
    )
    parser.add_argument(
        "--dtype", type=jnp.dtype, default=jnp.dtype("float32"), help="the dtype"
    )
    parser.add_argument(
        "--levels", type=int, default=3, help="the number of decomposition levels"
    )
    parser.add_argument("-n", type=int, default=100, help="the number of iterations")
    parser.add_argument(
        "--size", type=int, nargs=2, default=(512, 512), help="the image size"
    )
    parser.add_argument("--wavelet", type=str, default="bior4.4", help="the wavelet")
    args = parser.parse_args()

    print(f"Using device: {jnp.zeros(()).device().device_kind}")

    print(f"Using dtype: {args.dtype}")
    print(f"Number of decomposition levels: {args.levels}")
    print(f"Number of iterations: {args.n}")
    print(f"Using wavelet: {args.wavelet}")

    filt = get_filter_bank(args.wavelet, args.dtype)
    kernel_dec, kernel_rec = make_kernels(filt, args.channels)
    kdh, kdw = kernel_dec.shape[2], kernel_dec.shape[3]
    krh, krw = kernel_rec.shape[2], kernel_rec.shape[3]
    print(f"Kernel sizes: {kdh}x{kdw}, {krh}x{krw}")

    x = jax.random.normal(
        jax.random.PRNGKey(0),
        (args.batch_size, *args.size, args.channels),
        dtype=args.dtype,
    )
    print(f"Input shape: {x.shape}")

    # Benchmark DWT forward pass
    jit_down = jax.jit(partial(wavelet_dec, kernel=kernel_dec, levels=args.levels))
    y = jit_down(x)
    start = time.time()
    for i in range(args.n):
        y = jit_down(x)
    y.block_until_ready()
    time_taken = (time.time() - start) / args.n
    print(f"Time for  DWT  forward: {time_taken:g} s/it ({1 / time_taken:g} it/s)")

    # Benchmark IDWT forward pass
    jit_up = jax.jit(partial(wavelet_rec, kernel=kernel_rec, levels=args.levels))
    z = jit_up(y)
    start = time.time()
    for i in range(args.n):
        z = jit_up(y)
    z.block_until_ready()
    time_taken = (time.time() - start) / args.n
    print(f"Time for IDWT  forward: {time_taken:g} s/it ({1 / time_taken:g} it/s)")

    # Benchmark DWT backward pass
    vjp_down = jax.jit(jax.vjp(jit_down, x)[1])
    _ = vjp_down(y)
    start = time.time()
    for i in range(args.n):
        _ = vjp_down(y)[0]
    _.block_until_ready()
    time_taken = (time.time() - start) / args.n
    print(f"Time for  DWT backward: {time_taken:g} s/it ({1 / time_taken:g} it/s)")

    # Benchmark IDWT backward pass
    vjp_up = jax.jit(jax.vjp(jit_up, y)[1])
    _ = vjp_up(z)
    start = time.time()
    for i in range(args.n):
        _ = vjp_up(z)[0]
    _.block_until_ready()
    time_taken = (time.time() - start) / args.n
    print(f"Time for IDWT backward: {time_taken:g} s/it ({1 / time_taken:g} it/s)")

    # Compute reconstruction error
    mse = jnp.mean(jnp.square(jnp.float32(x - z)))
    rms = jnp.sqrt(mse)
    psnr = -10 * jnp.log10(mse)
    print(f"RMS reconstruction error: {rms.item():g} (PSNR: {psnr.item():g} dB)")

    # Compare with PyWavelets
    y_pywt = pywt.wavedec2(
        x, args.wavelet, level=args.levels, mode="periodization", axes=(1, 2)
    )
    y_unpack = unpack(y, args.levels)
    sq_norms = jax.tree_map(lambda x, y: jnp.sum((x - y) ** 2), y_pywt, y_unpack)
    mse = jax.tree_util.tree_reduce(jnp.add, sq_norms) / x.size
    rms = jnp.sqrt(mse)
    psnr = -10 * jnp.log10(mse)
    print(f"RMS diff from pywavelets: {rms.item():g} (PSNR: {psnr.item():g} dB)")

    # Test pack()
    y_pack = pack(y_unpack)
    assert jnp.all(y == y_pack)


if __name__ == "__main__":
    main()
