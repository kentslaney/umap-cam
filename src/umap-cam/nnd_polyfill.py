import numpy as np
import jax.numpy as jnp
from group import grouping
from pynndescent import NNDescent

class NNDResult(grouping(
        "NNDResult", ("points", "size"), ("distances", "indices", "flags"))):
    pass

def aknn(
        k, rng, data, delta=0.0001, iters=10, max_candidates=32, n_trees=None,
        verbose=False):
    data = np.asarray(data)
    knn = NNDescent(
            data, "euclidean", n_neighbors=k + 1, delta=delta, n_iters=iters,
            max_candidates=max_candidates, n_trees=n_trees,
            verbose=verbose).neighbor_graph[::-1]
    knn = tuple(jnp.asarray(i[:, 1:]) for i in knn)
    return rng, NNDResult.tree_unflatten(
            (), (*knn, jnp.zeros_like(knn[0], dtype=jnp.bool)))
