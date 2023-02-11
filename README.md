# jax-wavelets

The 2D [discrete wavelet transform](https://en.wikipedia.org/wiki/Discrete_wavelet_transform) for [JAX](https://jax.readthedocs.io/en/latest/).

## Motivation

The motivation for `jax-wavelets` is to replace the patching and unpatching transforms in [Vision Transformer](https://arxiv.org/abs/2010.11929) with transforms whose basis vectors are smooth and overlap, without increasing the number of floating point values input to and output from the model. This idea is from "[simple diffusion: End-to-end diffusion for high resolution images](https://arxiv.org/abs/2301.11093)" (Hoogeboom et al. 2023). The unpatching transform, for Vision Transformers which generate images, often leaves patch edge artifacts in the output image which it is difficult for the model to learn to remove entirely, and replacing it with the IDWT results in output images which are smoother and more visually appealing.

## Usage

Since it is intended for use with transformers, the `jax-wavelets` DWT takes an array shaped with shape `(N, H, W, C)` and returns an array with shape `(N, H // 2 ** levels, W // 2 ** levels, C * 4 ** levels)`. It uses the [PyWavelets](https://pywavelets.readthedocs.io/en/latest/) "[periodization](https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html#padding-using-pywavelets-signal-extension-modes-pad)" signal extension mode and no other modes are supported, since they would not result in a rectangular array of coefficients with the same total size as the input array. The `(N, H, W, C)` array can then be reshaped to `(N, H * W, C)` and passed to a learned input projection. The IDWT is intended to be used in an analogous fashion.

### Example

```python
import jax.numpy as jnp
import jax_wavelets as jw

# See https://wavelets.pybytes.com for wavelet names
filt = jw.get_filter_bank("bior4.4")
kernel_dec, kernel_rec = jw.make_kernels(filt, 3)

x = jnp.ones((1, 32, 32, 3))
y = jw.wavelet_dec(x, kernel_dec, 3)
z = jw.wavelet_rec(y, kernel_rec, 3)
print(jnp.sqrt(jnp.mean(jnp.square(x - z))))
```

TODO: add example of using with a transformer

### Benchmark/unit tests

```python
python3 -m jax_wavelets
```

The benchmark has many options which can be seen by using `--help`.
