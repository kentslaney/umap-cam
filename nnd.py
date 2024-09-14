from collections import namedtuple
from functools import partial
import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

import rpt
from group import Group, grouping, groupaux

class Heap(Group):
    @property
    def order(self):
        return self[0]

    def swapped(self, i0, i1):
        i0 = i0 if isinstance(i0, tuple) else (i0,)
        i1 = i1 if isinstance(i1, tuple) else (i1,)
        return self.at[:, *i0].set(self[:, *i1]).at[:, *i1].set(self[:, *i0])

    def sifted(self, i, bound=None):
        bound = self.order.shape[0] if bound is None else bound
        sel = (lambda x, y: x, lambda x, y: y)
        def cond(args):
            heap, i, broken = args
            return ~broken & (i * 2 + 1 < bound)

        def inner(args):
            heap, i, broken = args
            left, right, swap = i * 2 + 1, i * 2 + 2, i
            lpivot = heap.order[swap] < heap.order[left]
            swap = jax.lax.cond(lpivot, *sel, left,  swap)
            rpivot = (right < bound) & (heap.order[swap] < heap.order[right])
            swap = jax.lax.cond(rpivot, *sel, right, swap)

            broken = swap == i
            heap, i = jax.lax.cond(
                    broken, lambda heap, i, swap: (heap, i),
                    lambda heap, i, swap: (heap.swapped(swap, i), swap),
                    *(heap, i, swap))
            return heap, i, broken
        return jax.lax.while_loop(cond, inner, (self, i, False))[0]

    # will be replaced by AVL tree
    def ascending(self):
        def outer(i, heap):
            def inner(j, heap):
                j = self.shape[2] - 1 - j
                return heap.swapped((i, 0), (i, j)).at[:, i].sifted(0, j)
            return jax.lax.fori_loop(0, self.shape[2], inner, heap)
        return jax.lax.fori_loop(0, self.shape[1], outer, self)

    # low memory but O(size); PyNNDescent uses python sets in high memory mode
    # most devices probably vectorize it since it's a contiguous chunk of int32s
    # a binary search might be worth profiling but GPU jobs are usually IO bound
    # rapids uses bloom filters (even though they're removing elements?)
    def check(self, ins, idx):
        return jnp.all(ins[idx][None] != self[idx])

    # TODO: creating an updated index then doing a gather might be better than
    #   ... CaS followed by read dependencies? may have the same problem though
    #       need to think through the op order and what branch prediction can do
    #       even XLA native sort might be better because of sorting networks
    #       regardless, in practice this can be improved for consecutive inserts
    #       rapids keeps a fully ordered list using insertion sort and memmove
    #   ... but XLA can't guarantee contiguous memory blocks move efficiently
    #       asymptotically optimal is probably just a sorted tree data structure
    #       sort (distance, index) to prevent extra traversal for deduplication
    #       child pointers via an extra level of indirection alongside the heap
    #   ... is better for space efficiency and tree balance requirements
    #       it would be nice to be a strict superset of functionality
    def push(self, *value, checked=()):
        assert len(value) <= len(self), \
                f"can't push {len(value)} values to a group of {len(self)}"
        value = tuple(
                jnp.asarray(i) if isinstance(i, (int, float))
                else i for i in value)
        class Inserting(self[:len(value), 0].__class__, Heap):
            pass
        ins = Inserting.tree_unflatten(self.aux_data, value)
        res = (ins.order < self.order[0]) & jnp.all(jnp.asarray(tuple(
                self.check(ins, i) for i in checked)))

        def init(heap):
            heap = heap.at[:len(value), 0].set(value)
            _, i, heap = jax.lax.while_loop(
                    lambda a: a[0], inner, (True, 0, heap))
            return heap.at[:len(value), i].set(value)

        def inner(args):
            continues, i, heap = args
            left, right = 2 * i + 1, 2 * i + 2
            loob = left >= heap.order.shape[0]
            roob = right >= heap.order.shape[0]
            flipped = lambda: heap.order[left] >= heap.order[right]
            pivot = jnp.where(roob | flipped(), left, right)
            swapping = ~loob & (ins.order < heap.order[pivot])
            return (swapping,) + jax.lax.cond(
                    swapping,
                    lambda: (pivot, heap.at[:, i].set(heap[:, pivot])),
                    lambda: (i, heap))
        return res, jax.lax.cond(res, init, lambda a: a, self)

    def pusher(self, *a, **kw):
        return self.push(*a, **kw)[1]

euclidean = jax.jit(lambda x, y: jnp.sqrt(jnp.sum((x - y) ** 2)))

@jax.tree_util.register_pytree_node_class
class NNDHeap(Heap, grouping(
        "NNDHeap", ("points", "size"), ("distances", "indices", "flags"),
        (jnp.float32(jnp.inf), jnp.int32(-1), jnp.bool(False)))):
    @partial(jax.jit, static_argnames=('limit',))
    def build(self, limit, rng):
        def init(i, args):
            def loop(j, args):
                conts = self.indices[i, j] < 0
                return jax.lax.cond(conts, lambda *a: a[2:], inner, j, i, *args)
            return jax.lax.fori_loop(0, self.shape[2], loop, args)
        def inner(j, i, heap, rng):
            rng, subkey = jax.random.split(rng)
            d = jax.random.uniform(subkey)
            # check flag
            idx, isn = self.indices[i, j], self.flags[i, j]
            update = jax.lax.cond(isn, lambda: heap[1], lambda: heap[0])
            update = update.at[:, i].pusher(d, idx, checked=("indices",))
            update = update.at[:, idx].pusher(d, i, checked=("indices",))
            update = jax.lax.cond(
                    isn, lambda: (heap[0], update), lambda: (update, heap[1]))
            return heap.tree_unflatten(heap.aux_data, update), rng
        clone = self.__new__(
                self.__class__, self.spec.points, limit, **self.aux_dict)
        heap, rng = jax.lax.fori_loop(0, self.shape[1], init, (self.grouped(
                clone, clone, names=("old", "new")), rng))
        def end(i, cur):
            def loop(j, cur):
                mask = cur.indices[i] == heap.new.indices[i, j]
                # reset flag
                return cur.at["flags"].set(jnp.where(mask, False, cur.flags[i]))
            return jax.lax.fori_loop(0, limit, loop, cur)
        cur = jax.lax.fori_loop(0, self.shape[1], end, self)
        return cur, NNDCandidates(*heap[:, "indices"]), rng

    @partial(jax.jit, static_argnames=('dist',))
    def randomize(self, data, rng, dist=euclidean):
        def inner(j, args):
            i, heap, rng = args
            rng, subkey = jax.random.split(rng)
            idx = jax.random.randint(subkey, (), 0, self.shape[1] - 1)
            idx += idx >= i
            heap = heap.at[:, i].pusher(
                    dist(data[idx], data[i]), idx, jnp.bool(True),
                    checked=("indices",))
            return i, heap, rng
        def init(i, args):
            heap, rng = args
            cond = heap.indices[i, 0] < 0
            return jax.lax.cond(cond, lambda i, heap, rng: jax.lax.fori_loop(
                    jnp.sum(heap.indices[i] >= 0), heap.shape[1], inner,
                    (i, heap, rng)), lambda *a: a, i, *args)[1:]
        return jax.lax.fori_loop(0, self.shape[1], init, (self, rng))

    def apply(self, p, q, d):
        heap, total = self, 0
        for i, j in ((p, q), (q, p)):
            # set flag
            added, updated = heap.indirect[:, i].push(
                    d, j, 1, checked=("indices",))
            heap = heap.at[:, i].set(updated)
            total += added
        return heap, total

    @staticmethod
    def accumulator(p, q, d, heap, total):
        heap, added = heap.apply(p, q, d)
        return heap, total + added

    @partial(jax.jit, static_argnames=('dist',))
    def update(self, candidates, data, dist=euclidean):
        return candidates.updates(
                self.accumulator, self.distances[:, 0], data, self, 0,
                dist=dist)

@jax.tree_util.register_pytree_node_class
class NNDHeapGPU(NNDHeap):
    def build(self, limit, rng):
        # want to enable asyncronous writes but need deterministic order
        # original repo's implementation interleaves threads' memory access
        # instead, start by assigning random values to avoid prng order issues
        # first pass over indices sorted by flag: jax.scan carrying count so far
        # second pass pallas which can have out of order write b/c cache misses
        # reservoir sample referenced row, writing iff the prev count is smaller
        # requires atomic read/write; jax.experimental.pallas.atomic_max

        assert limit > 0, "limit should be strictly positive"
        assert (limit & (limit - 1)) == 0, "limit should be a power of 2"
        assert (self.shape[2] & (self.shape[2] - 1)) == 0, \
                "n_neighbors should be a power of 2" # TODO: relax

        counts = jnp.sum(self.flags, axis=1)
        counts = jnp.asarray((self.shape[2] - counts, counts))
        if limit < self.shape[2]:
            rng, subkey = jax.random.split(rng)
            order = jnp.tile(jnp.arange(self.shape[2]), (self.shape[1], 1))
            order = jax.random.permute(subkey, order, 1, True)
            split = jnp.take(self.flags, order, 1, unique_indices=True)
            _, order = jax.lax.sort_key_val(split, order)
            ordered = jnp.take(self.indices, order, 1, unique_indices=True)
        else:
            _, ordered = jax.lax.sort_key_val(self.flags, self.indices)
        ordered = jnp.asarray((ordered[:, :limit], ordered[:, :~limit:-1]))

        # could be built iteratively and stored
        def sample_n(counts, node):
            bound, node = node[0], node[1:]
            valid = jnp.arange(node.size) < bound
            return counts.at[node].add(valid), counts[node]
        nodes = jnp.concatenate((counts[..., None], ordered), -1)
        _, n = jax.vmap(lambda counts, nodes: jax.lax.scan(
                sample_n, counts, nodes))(counts, nodes)

        rng, subkey = jax.random.split(rng)
        d = jax.random.uniform(subkey, ordered.shape)
        d = jnp.int32(jnp.maximum(d, n < limit) * n)

        mask = jnp.arange(limit)[None] < counts[:, :, None]
        oob = jnp.stack((self.shape[1], limit))[None, :, None, None]
        d = jnp.where(mask[:, None], jnp.stack((ordered, d), 1), oob)

        def reservoir(idx, ref):
            via = jnp.tile(jnp.arange(idx.shape[1])[:, None], (1, idx.shape[2]))
            return ref.at[*idx.reshape(2, -1)].max(via.flatten(), mode="drop")
        ref = jax.vmap(reservoir)(d, jnp.full((2, self.shape[1], limit), -1))

        def backfill(count, forward, ref):
            return jnp.where(
                    (jnp.arange(self.shape[2]) < count) & (ref < 0),
                    forward, ref)
        ref = jax.vmap(jax.vmap(backfill))(counts, ordered, ref)
        update = self.indices[:, :limit] != ref[1, :, ::-1]
        unset = ((0, 0), (0, max(0, limit - self.shape[2])))
        update = jnp.pad(update, unset, constant_values=True)
        update = self.at["flags"].set(self.flags & update)
        return update, NNDCandidates(*ref), rng

    def randomize(self, data, rng, dist=euclidean):
        def outer(rng, i, _, existing, flags):
            idx = jax.random.choice(
                    rng, self.shape[1] - 1, (self.shape[2],), False)
            idx = jnp.where(existing == -1, idx + (idx >= i), existing)
            d = jax.vmap(dist, (0, None))(data[idx], data[i])
            _, flags = jax.lax.sort_key_val(d, existing == -1)
            d, idx = jax.lax.sort_key_val(d, idx)
            return d[::-1], idx[::-1], flags
        rng = jax.random.split(rng, self.shape[1] + 1)
        res = jax.vmap(outer)(rng[1:], jnp.arange(self.shape[1]), *self)
        return self.tree_unflatten((), res), rng[0]

class Candidates:
    def checked(self, apply, thresholds, data, dist):
        def inner(k, idx1, j, idx0, i, out):
            p, q = idx0[i, j], idx1[i, k]
            d = dist(data[p], data[q])
            cond = (d < thresholds[p]) | (d < thresholds[q])
            return jax.lax.cond(
                    (p != q) & cond, lambda: apply(p, q, d, *out), lambda: out)

        def outer(idx, start, f):
            idx = self[idx]
            def loop(j, *args):
                i, out = (j, *args)[-2:]
                def inner(k, args):
                    return jax.lax.cond(
                            idx[i, k] >= 0, lambda *a: (*a[2:-1], f(*a)),
                            lambda *a: a[2:], k, idx, *args)
                return jax.lax.fori_loop(
                        start(j), self.shape[2], inner, (j, *args))[-1]
            return loop
        return outer, inner

class NNDCandidates(Candidates, grouping(
        "NNDCandidates", ("points", "size"), ("old", "new"))):
    @partial(jax.jit, static_argnames=('apply', 'dist'))
    def updates(self, apply, thresholds, data, *out, dist=euclidean):
        outer, inner = self.checked(apply, thresholds, data, dist)
        f = outer("new", lambda i: 0, lambda *a: h(*a[:-1], g(*a)))
        g = outer("new", lambda i: i + 1, inner)
        h = outer("old", lambda i: 0, inner)
        # TODO: want vmap, but has an arbitrary number of reverse neighbors
        # https://github.com/rapidsai/raft/blob/branch-24.10/cpp/include/raft/neighbors/detail/nn_descent.cuh#L517
        # huh?
        # convert each linked list to an AVL tree then merge
        return jax.lax.fori_loop(0, self.shape[1], f, out)

    # could be built iteratively and stored
    # jax.scan over rows to create a linked list of reverse neighbors
    def links(self):
        def linker(head, args):
            node, i = args
            oob = jnp.where(node == -1, self.spec.points, node)
            res = jnp.where((node == -1)[:, None], -1, head[node])
            col = jnp.arange(self.spec.size)
            coords = jnp.stack((jnp.broadcast_to(i, self.spec.size), col), -1)
            return head.at[oob].set(coords, mode="drop"), res
        init = jnp.full((self.spec.points + 1, 2), -1)
        f = lambda x: jax.lax.scan(
                linker, init, (x, jnp.arange(self.spec.points)))
        return jax.vmap(f)(jnp.stack(self))

@jax.tree_util.register_pytree_node_class
class RPCandidates(groupaux("total"), Candidates, grouping(
        "RPCandidates", ("points", "size"))):
    @partial(jax.jit, static_argnames=('apply', 'dist'))
    def updates(self, apply, thresholds, data, *out, dist=euclidean):
        outer, inner = self.checked(apply, thresholds, data, dist)
        f = outer(0, lambda i: 0, lambda *a: g(*a))
        g = outer(0, lambda i: i + 1, inner)
        return jax.lax.fori_loop(0, self.total, f, out)

    @classmethod
    def forest(cls, *a, **kw):
        rng, total, trees = rpt.forest(*a, **kw)
        return rng, cls(trees, total=total)

    def __repr__(self):
        sliced = self[0, :self.total] if self.total != self.shape[1] else self
        return str(sliced._value)

@partial(jax.jit, static_argnames=("k", "max_candidates", "n_trees"))
def aknn(k, rng, data, delta=0.0001, iters=10, max_candidates=32, n_trees=None):
    max_candidates = min(64, k) if max_candidates is None else max_candidates
    heap = NNDHeap(data.shape[0], k)
    heap, rng = heap.randomize(data, rng)
    if n_trees != 0:
        rng, trees = RPCandidates.forest(rng, data, n_trees, max_candidates)
        heap, _ = heap.update(trees, data)
    def cond(args):
        i, broken, _, _ = args
        return ~broken & (i < iters)
    def loop(args):
        i, _, heap, rng = args
        heap, step, rng = heap.build(max_candidates, rng)
        heap, changes = heap.update(step, data)
        # jax.debug.print("finished iteration {} with {} updates", i, changes)
        return i + 1, changes <= delta * k * data.shape[0], heap, rng
    i, _, heap, rng = jax.lax.while_loop(cond, loop, (0, False, heap, rng))
    # jax.lax.cond(i < iters, lambda: jax.debug.print(
    #         "stopped early after {} iterations", i), lambda: None)
    return rng, heap.ascending()

TestingConfig = namedtuple(
        "Config",
        ("points", "neighbors", "max_candidates", "n_trees", "ndim", "seed"),
        defaults=(512, 8, 4, 2, 8, 0))

def test_step(*a, **kw):
    setup = TestingConfig(*a, **kw)
    rng = jax.random.key(setup.seed)
    rng, subkey = jax.random.split(rng)
    data = jax.random.normal(subkey, (setup.points, setup.ndim))
    rng, heap = aknn(
            setup.neighbors, rng, data, max_candidates=setup.max_candidates,
            n_trees=setup.n_trees)
    return data, heap

def npy_cache(uniq, *, ndim=16, path=None, **kw):
    import pathlib
    path = pathlib.Path.cwd() if path is None else pathlib.Path(path)
    path = path.parents[0] if path.is_file() else path
    assert path.is_dir()
    full = path / f"{uniq}.npz"
    if not full.is_file():
        data, heap = test_step(ndim=ndim, **kw)
        jnp.savez(full, data, heap)
    else:
        data, heap = jnp.load(full).values()
        heap = NNDHeap.tree_unflatten((), heap)
    return data, heap

if __name__ == "__main__":
    import sys
    heap = test_step(*map(int, sys.argv[1:]))
    print(f"{heap=}")
