# UMAP
This repo provides a JAX implementation of UMAP and a color space (CIECAM16-UCS)
that points can be embedded into along with spacial dimensions. It also features
an updated data structure that should be more cache friendly.

The code works on test data, but never seems to finish tracing with reasonable
real-world dimensions (eg `TestIntegration.test_digits_avl_aknn` as of JAX
`0.4.33`)
