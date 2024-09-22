import jax, os
import jax.numpy as jnp
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parents[0]))
import nnd, rpt, group, nnd_gpu, avl
from importlib import reload
group = reload(group)
nnd, rpt = reload(nnd), reload(rpt)
avl, nnd_gpu = reload(avl), reload(nnd_gpu)

from nnd import RPCandidates, NNDHeap
from nnd_gpu import NNDHeap as NNDHeapGPU

def test_setup(data=None, k=16, rng=None, max_candidates=16, n_trees=1):
    rng = jax.random.key(0) if rng is None else rng
    if data is None:
        rng, subkey = jax.random.split(rng)
        data = jax.random.normal(subkey, (512, 8))
    heap = NNDHeap(data.shape[0], k)
    heap, rng = heap.randomize(data, rng)
    if n_trees != 0:
        rng, trees = RPCandidates.forest(rng, data, n_trees, max_candidates)
        heap, _ = heap.update(trees, data)
    return data, heap, rng

# data, heap, _ = test_setup()

def test_cached(
        data=None, k=16, rng=None, max_candidates=16, n_trees=1,
        path=None, uniq="reservoir"):
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
            data = jnp.asarray(data)
    if not full.is_file():
        data, heap, _ = test_setup(*params)
        jnp.savez(full, data, heap, params)
    form = NNDHeapGPU(*heap.shape[1:])
    heap = NNDHeapGPU.tree_unflatten(
            (), tuple(heap) + form.tree_flatten()[0][len(heap):])
    return data, heap

def test(data=None, k=16, rng=None, max_candidates=16, n_trees=1):
    data, heap = test_cached(data, k, rng, max_candidates, n_trees)
    rng = jax.random.key(0)
    rng, subkey = jax.random.split(rng)
    heap = heap.at['flags'].set(jax.random.bernoulli(subkey, shape=(512, 16)))
    return (heap, data) + heap.build(max_candidates, rng)

heap, data, update, step, rng = test()
# data, heap = test_cached()
heap = heap.remap().batched()
links = step.links()
bounds = step.bounds(data, heap, True)
print(links.rebuild(step, bounds, heap, data))
# print(heap)
