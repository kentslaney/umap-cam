from nnd import test_step, npy_cache
from umap import initialize, Optimizer
import jax, pathlib

path = pathlib.Path.home() / "logdump" / "tensorboard"
data, heap = npy_cache("test_step", neighbors=14)
with jax.profiler.trace(str(path)):
    # test_step()[1].indices.block_until_ready()
    rng, embed, adj = initialize(jax.random.key(0), data, heap, 2)
    rng, lo, hi = Optimizer().optimize(rng, embed, adj)
    lo.block_until_ready()
