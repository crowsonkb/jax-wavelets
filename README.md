# jax-wavelets

The 2D [discrete wavelet transform](https://en.wikipedia.org/wiki/Discrete_wavelet_transform) for [JAX](https://jax.readthedocs.io/en/latest/).

## Motivation

The motivation for `jax-wavelets` is to replace the patching and unpatching transforms in [Vision Transformer](https://arxiv.org/abs/2010.11929) with transforms whose basis vectors are smooth and overlap, without increasing the number of floating point values input to and output from the model. This idea is from "[simple diffusion: End-to-end diffusion for high resolution images](https://arxiv.org/abs/2301.11093)" (Hoogeboom et al. 2023). The unpatching transform, for Vision Transformers which generate images, often leaves patch edge artifacts in the output image which it is difficult for the model to learn to remove entirely, and replacing it with the IDWT results in output images which are smoother and more visually appealing.

## Usage

`jax-wavelets` supports [PyWavelets](https://pywavelets.readthedocs.io/en/latest/)' discrete wavelet types and refers to them by the same names. See https://wavelets.pybytes.com for wavelet names. The two wavelets used by [JPEG 2000](https://en.wikipedia.org/wiki/JPEG_2000) are [CDF](https://en.wikipedia.org/wiki/Cohen–Daubechies–Feauveau_wavelet) 5/3 for lossless compression and 9/7 for lossy compression. They are known to `jax-wavelets` as "bior2.2" and "bior4.4".

Since it is intended for use with transformers, the `jax-wavelets` DWT takes an array shaped with shape `(N, H, W, C)` and returns an array with shape `(N, H // 2 ** levels, W // 2 ** levels, C * 4 ** levels)`. The resulting array of coefficients can then be reshaped to `(N, H * W // 4 ** levels, C * 4 ** levels)` and passed to a learned input projection. The IDWT is intended to be used in an analogous fashion, and must be used with the same signal extension mode as the DWT.

`jax-wavelets` supports the "wrap" (PyWavelets "periodization", MATLAB "per") and "reflect" (PyWavelets "reflect", MATLAB "symw") signal extension modes. Both produce the same number of output coefficients as input coefficients. "wrap" is supported for all discrete wavelet types, while "reflect" is only supported for symmetric odd length wavelets, that is, "bior2.2", "bior2.4", "bior2.6", "bior2.8", "bior4.4", and "bior6.8".


### Example

```python
import jax
import jax.numpy as jnp
import jax_wavelets as jw

# See https://wavelets.pybytes.com for wavelet names
filt = jw.get_filter_bank("bior4.4")
kernel_dec, kernel_rec = jw.make_kernels(filt, 3)

x = jax.random.normal(jax.random.PRNGKey(0), (1, 32, 32, 3))
y = jw.wavelet_dec(x, kernel_dec, levels=3, mode="reflect")
z = jw.wavelet_rec(y, kernel_rec, levels=3, mode="reflect")
print(jnp.sqrt(jnp.mean(jnp.square(x - z))))
```

TODO: add example of using with a transformer

### Benchmark/unit tests

```python
python3 -m jax_wavelets
```

The benchmark has many options which can be seen by using `--help`.
