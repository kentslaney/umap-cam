import math
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

# print(f"{i:02} {i:04b} {rightmostSetBit(i): 2} {rightmostUnsetBit(i)}" + (
#         " {: 4}" * 2).format(*binaryBounds(i, range(100, 116), 16)))

@partial(jax.jit, static_argnames=("max_leaf_size", "bound", "loops"))
def partition(data, rng, max_leaf_size=30, bound=0.75, loops=None):
    if loops is None:
        loops = math.ceil(math.log(max_leaf_size / data.shape[0], 0.5))
    splits = jnp.zeros((2 ** loops - 1,), dtype=jnp.int32)
    planes = jnp.zeros((2 ** loops - 1, data.shape[1] + 1))
    order = jnp.arange(data.shape[0])
    for depth in range(loops):
        largest_possible = math.ceil(data.shape[0] * bound ** depth)
        def inner(segment, args):
            rng, splits, step, planes, order = args
            left, right = binaryBounds(segment, splits, data.shape[0])
            size = right - left
            def non_leaf(rng, splits, step, planes, order):
                start = jnp.minimum(data.shape[0] - largest_possible, left)
                idx = start + jnp.arange(largest_possible)
                window = (idx >= left) & (idx < right)
                sliced = jax.lax.dynamic_slice_in_dim(
                        order, start, largest_possible)
                empty = (jnp.zeros((data.shape[1],)), jnp.zeros(()))

                def cond(args):
                    count = args[0]
                    return (count < (1 - bound) * size) | (count > bound * size)
                def degenerate(count, mask, rng, normal, boundary):
                    rng, subkey = jax.random.split(rng)
                    mask = jax.random.bernoulli(subkey, shape=(
                        largest_possible,))
                    return jnp.sum(mask & window), mask, rng, *empty
                def loop(args):
                    prev, mask, rng, normal, boundary = args
                    rng, first, second = sample(rng, left, right)
                    normal, boundary = hyperplane(
                            data[order[first]], data[order[second]])
                    mask = jnp.sum(data[sliced] * normal[None], 1) > boundary
                    count = jnp.sum(mask & window)
                    return jax.lax.cond(
                            (count == prev) | (size - count == prev),
                            degenerate, lambda *a: a,
                            count, mask, rng, normal, boundary)
                count, mask, rng, normal, boundary = jax.lax.while_loop(
                        cond, loop, (0, window, rng, *empty))

                sort = (mask & (idx >= left)) | (idx >= right)
                splits = splits.at[segment].set(left + count)
                planes = planes.at[segment, 0].set(boundary)
                planes = planes.at[segment, 1:].set(normal)
                delta = jnp.where(
                        sort, jnp.cumsum(~sort[::-1])[::-1], -jnp.cumsum(sort))
                section = jax.lax.dynamic_slice_in_dim(
                        step, start, largest_possible)
                step = jax.lax.dynamic_update_slice_in_dim(
                        step, section + delta, start, 0)
                return rng, splits, step, planes, order
            return jax.lax.cond(
                    size > max_leaf_size, non_leaf,
                    lambda rng, splits, step, planes, order: (
                            rng, splits.at[segment].set(-1), step, planes,
                            order),
                    *args)
        rng, splits, step, planes, order = jax.lax.fori_loop(
                2 ** depth - 1, 2 ** (depth + 1) - 1, inner,
                (rng, splits, jnp.arange(data.shape[0]), planes, order))
        order = order[step]
    return rng, splits, order, planes

if __name__ == "__main__":
    rng = jax.random.key(0)
    rng, subkey = jax.random.split(rng)
    data = jax.random.normal(subkey, (100, 1))
    print(partition(data, rng))
