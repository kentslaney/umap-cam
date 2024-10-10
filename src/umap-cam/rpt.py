import math, warnings
from functools import partial
import jax.numpy as jnp
import jax

def sample(rng, start, end):
    rng, first, second = jax.random.split(rng, 3)
    first = jax.random.randint(first, (), start, end)
    second = jax.random.randint(second, (), start, end - 1)
    second += second >= first
    return rng, first, second

def hyperplane(left, right):
    delta = left - right
    boundary = jnp.sum(delta * (left + right) / 2.)
    return delta, boundary

def rightmostSetBit(n):
    if isinstance(n, jnp.ndarray):
        return jnp.int32(jnp.ceil(jnp.log2((n & -n) + 1))) - 1
    return math.ceil(math.log2((n & -n) + 1)) - 1

def rightmostUnsetBit(n):
    if isinstance(n, jnp.ndarray):
        return jax.lax.cond(
                n & (n + 1) == 0,
                lambda: jnp.int32(jnp.log2(n + 1)),
                lambda: rightmostSetBit(~n))
    if n & (n + 1) == 0:
        return int(math.log2(n + 1))
    return rightmostSetBit(~n)

def binaryBounds(n, idx, size):
    if isinstance(n, jnp.ndarray):
        f = lambda n, m: idx[(n - 2 ** (m + 1) + 1) >> (m + 1)]
        nominal = lambda: (f(n, rightmostSetBit(n + 1)), jax.lax.cond(
                (n + 1) & (n + 2) == 0, lambda: size,
                lambda: f(n, rightmostUnsetBit(n))))
        return jax.lax.cond(n == 0, lambda: (0, size), lambda: jax.lax.cond(
                n & (n + 1) == 0, lambda: (0, idx[(n - 1) // 2]), nominal))
    if n == 0:
        return 0, size
    if n & (n + 1) == 0: # rightmostSetBit(n + 1) == math.log2(n + 1)
        return 0, idx[(n - 1) // 2]
    left = rightmostSetBit(n + 1) + 1
    left = (n - 2 ** left + 1) >> left
    if (n + 1) & (n + 2) == 0: # rightmostUnsetBit(n + 1) == math.log2(n + 2)
        return idx[left], size
    right = rightmostUnsetBit(n + 1) + 1
    right = (n - 2 ** right + 1) >> right
    return idx[left], idx[right]

def test_bounds():
    for i in range(16):
        print(
                f"{i:02} {i:04b} {rightmostSetBit(i): 2} {rightmostUnsetBit(i)}"
                + (" {: 4}" * 2).format(*binaryBounds(i, range(100, 116), 16)))

@partial(jax.jit, static_argnames=(
        "goal_leaf_size", "bound", "loops", "warn", "verbose"))
def rp_tree(
        rng, data, goal_leaf_size=30, bound=0.75, loops=None, warn=True,
        verbose=False):
    if loops is None:
        loops = math.ceil(math.log(goal_leaf_size / data.shape[0], 0.5))
    splits = jnp.zeros((2 ** loops - 1,), dtype=jnp.int32)
    planes = jnp.zeros((2 ** loops - 1, data.shape[1] + 1))
    order = jnp.arange(data.shape[0])

    @partial(jax.vmap, in_axes=(None, None, 0, 0))
    @partial(jax.jit, static_argnames=('largest_possible',))
    def inner(largest_possible, splits, rng, segment):
        left, right = binaryBounds(segment, splits, data.shape[0])
        size = right - left
        def partition(rng, step, planes):
            start = jnp.minimum(data.shape[0] - largest_possible, left)
            idx = start + jnp.arange(largest_possible)
            window = (idx >= left) & (idx < right)
            sliced = jax.lax.dynamic_slice_in_dim(
                    order, start, largest_possible)

            def cond(args):
                count = args[0]
                return (count < (1 - bound) * size) | (count > bound * size)
            def degenerate(count, mask, rng, normal, boundary):
                rng, subkey = jax.random.split(rng)
                mask = jax.random.bernoulli(subkey, shape=(
                    largest_possible,))
                return (
                        jnp.sum(mask & window), mask, rng,
                        jnp.asarray((jnp.inf,) * data.shape[1]),
                        jnp.asarray(jnp.inf))
            def loop(args):
                prev, mask, rng, normal, boundary = args
                rng, first, second = sample(rng, left, right)
                # for arbitrary distance functions, check which is closer
                normal, boundary = hyperplane(
                        data[order[first]], data[order[second]])
                mask = jnp.sum(data[sliced] * normal[None], 1) > boundary
                count = jnp.sum(mask & window)
                return jax.lax.cond(
                        (count == 0) | (count == size) | (
                                count == prev) | (size - count == prev),
                        degenerate, lambda *a: a,
                        count, mask, rng, normal, boundary)

            empty = (jnp.zeros((data.shape[1],)), jnp.zeros(()))
            count, mask, rng, normal, boundary = jax.lax.while_loop(
                    cond, loop, (0, window, rng, *empty))

            sort = (mask & (idx >= left)) | (idx >= right)
            _splits = left + count
            planes = planes.at[0].set(boundary).at[1:].set(normal)
            delta = jnp.where(
                    sort, jnp.cumsum(~sort[::-1])[::-1], -jnp.cumsum(sort))
            step = jax.lax.dynamic_update_slice_in_dim(step, delta, start, 0)
            return step, _splits, planes
        return jax.lax.cond(
                size > goal_leaf_size, partition,
                lambda rng, step, planes: (step , -1, planes), rng,
                jnp.zeros(data.shape[0], jnp.int32),
                jnp.zeros(data.shape[1] + 1))

    for depth in range(loops):
        largest_possible = math.ceil(data.shape[0] * bound ** depth)
        # skips over already-determined splits
        start, end = (2 ** depth - 1, 2 ** (depth + 1) - 1)
        layer = jnp.arange(start, end)
        rng = jax.random.split(rng, layer.size + 1)
        rng, subkeys = rng[0], rng[1:]
        step, _splits, _planes = inner(largest_possible, splits, subkeys, layer)
        splits = splits.at[start:end].set(_splits)
        planes = planes.at[start:end].set(_planes)
        order = order[jnp.sum(step, 0) + jnp.arange(data.shape[0])]
        if verbose:
            jax.debug.print(f"finished rpt loop {depth} of {loops}")

    def warner(randomized, total):
        if total == 0:
            return
        at = jnp.concatenate(jnp.where(randomized))
        if total >= 16:
            warnings.warn(f"randomized {total} splits")
            return
        warnings.warn(
                f"randomized {total} split"
                f"{'s: indices' if total > 1 else ': index'} "
                f"{', '.join(map(str, at[:-1]))}{',' if len(at) > 2 else ''}"
                f"{' and ' if len(at) > 1 else ''}{"".join(at[-1:])}")
    randomized = jnp.all(jnp.isinf(planes), axis=1)
    total = jnp.sum(randomized)
    jax.lax.cond(warn & (total > 0), lambda: jax.debug.callback(
        warner, randomized, total), lambda: None)
    return rng, splits, order, planes

@partial(jax.jit, static_argnames=("max_leaf_size", "bound"))
def flatten(rng, splits, order, max_leaf_size=30, bound=0.75):
    splits = jnp.where(splits < 0, 0, splits)
    splits = jnp.where(splits > order.shape[0], order.shape[0], splits)
    splits = jnp.concatenate((splits, jnp.asarray((0, order.shape[0]))))
    splits = jnp.sort(splits)
    deltas = splits[1:] - splits[:-1]
    bins = jnp.int32(jnp.ceil(deltas / max_leaf_size))
    spread = lambda x: jnp.repeat(x, deltas, total_repeat_length=order.shape[0])
    idx = spread(jnp.cumsum(bins)) - (jnp.arange(order.shape[0]) % spread(bins))
    idx = jnp.vstack((idx - 1, order))
    rng, subkey = jax.random.split(rng)
    idx, order = jax.lax.sort_key_val(*jax.random.permutation(subkey, idx, 1))
    max_leaves = math.ceil(order.shape[0] / (max_leaf_size * (1 - bound)))
    offset = jnp.arange(order.shape[0]) % max_leaf_size
    candidates = jnp.full((max_leaves, max_leaf_size), -1)
    return rng, candidates.at[idx, offset].set(order), jnp.sum(bins)

@partial(jax.jit, static_argnames=(
        "n_trees", "max_leaf_size", "bound", "loops", "verbose"))
def forest(
        rng, data, n_trees=None, max_leaf_size=30, bound=0.75, loops=None,
        verbose=False):
    if n_trees is None:
        n_trees = 5 + int(round((data.shape[0]) ** 0.25))
        n_trees = min(32, n_trees)
    _, splits, order, planes = jax.eval_shape(partial(
            rp_tree, goal_leaf_size=max_leaf_size,
            bound=bound, loops=loops, warn=False), rng, data)
    _, leaves, size = jax.eval_shape(partial(
            flatten, max_leaf_size=max_leaf_size,
            bound=bound), rng, splits, order)
    def loop(rng):
        rng, splits, order, planes = rp_tree(
                rng, data, max_leaf_size, bound, loops, verbose, verbose)
        rng, leaves, size = flatten(rng, splits, order, max_leaf_size, bound)
        return (leaves, size)
    rng, *subkeys = jax.random.split(rng, n_trees + 1)
    trees, totals = jax.vmap(loop)(jnp.stack(subkeys))
    totals = jnp.cumsum(jnp.concatenate((jnp.zeros((1,), jnp.int32), totals)))
    flat = jnp.full((n_trees * order.shape[0], max_leaf_size), -1)
    for i in range(n_trees):
        flat = jax.lax.dynamic_update_slice_in_dim(flat, trees[i], totals[i], 0)
    return rng, totals[-1], flat
