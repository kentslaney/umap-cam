# UMAP
This repo provides a JAX implementation of UMAP and a color space (CIECAM16-UCS)
that points can be embedded into along with spacial dimensions. It also features
an updated data structure that should be more cache friendly.

The tests work when run on data points 0-166 or 1-168, but hang with 0-167
```bash
python test.py TestHangingIsolation
```
(output warning: 10k-20k lines scrollback)

```bash
ps x | grep "[t]est.py" | sed "s/^ *\([0-9]\+\) .*$/\1/" | xargs kill -9
```
(the hanging test needs ^Z to freeze the job and this to kill it)

hopefully not pertinent for long, but felt obligated to document my frustration
