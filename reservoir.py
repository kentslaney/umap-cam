import jax, os
import jax.numpy as jnp
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parents[0]))
import nnd, rpt
from importlib import reload
nnd, rpt = reload(nnd), reload(rpt)

from nnd import NNDHeap, NNDHeapGPU, RPCandidates

def test_setup(data=None, k=16, rng=None, max_candidates=16, n_trees=1):
    rng = jax.random.key(0) if rng is None else rng
    if data is None:
        rng, subkey = jax.random.split(rng)
        data = jax.random.normal(subkey, (512, 8))
    heap = NNDHeapGPU(data.shape[0], k)
    heap, rng = heap.randomize(data, rng)
    if n_trees != 0:
        rng, trees = RPCandidates.forest(rng, data, n_trees, max_candidates)
        heap, _ = heap.update(trees, data)
    return data, heap, rng

data, heap, _ = test_setup()

def test_cached(data=None, k=16, rng=None, max_candidates=16, n_trees=1, path=None, uniq="reservoir"):
    import pathlib
    path = pathlib.Path.cwd() if path is None else pathlib.Path(path)
    path = path.parents[0] if path.is_file() else path
    assert path.is_dir()
    full = path / f"{uniq}.npz"
    params = (data, k, rng, max_candidates, n_trees)
    if full.is_file():
        data, heap, prev = jnp.load(full, allow_pickle=True).values()
        if any(params != prev):
            os.remove(full)
        else:
            heap = NNDHeapGPU.tree_unflatten((), heap)
    if not full.is_file():
        data, heap, _ = test_setup(*params)
        jnp.savez(full, data, heap, params)
    return data, heap

def test(data=None, k=16, rng=None, max_candidates=16, n_trees=1):
    data, heap = test_cached(data, k, rng, max_candidates, n_trees)
    rng = jax.random.key(0)
    rng, subkey = jax.random.split(rng)
    heap = heap.at['flags'].set(jax.random.bernoulli(subkey, shape=(512, 16)))
    return (heap, data) + heap.build(max_candidates, rng)

# heap, data, update, step, rng = test()
# tail, head = step.links()
