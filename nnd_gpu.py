from functools import partial
import jax
import jax.numpy as jnp
from group import grouping, group_alias, dim_alias, marginalized
from avl import SingularAVL, MaxAVL, AVLs

euclidean = jax.jit(lambda x, y: jnp.sqrt(jnp.sum((x - y) ** 2)))

@jax.tree_util.register_pytree_node_class
class NNDHeap(
        group_alias(key="distances", secondary="indices"),
        dim_alias(trees="points"),
        MaxAVL,
        grouping(
            "NNDHeap", ("points", "size"),
            ("distances", "indices", "flags", "left", "right", "height"), (
                jnp.float32(jnp.inf), jnp.int32(-1), jnp.bool(False),
                jnp.int32(-1), jnp.int32(-1), jnp.int32(1)))):
    def build(self, limit, rng):
        # want to enable asyncronous writes but need deterministic order
        # original repo's implementation interleaves threads' memory access
        # instead, start by assigning random values to avoid prng order issues
        # first pass over indices sorted by flag: jax.scan carrying count so far
        # second pass pallas which can have out of order write b/c cache misses
        # reservoir sample referenced row, writing iff the prev count is smaller
        # requires atomic read/write; jax.experimental.pallas.atomic_max

        assert limit > 0, "limit should be strictly positive"

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
            return ref.at[(*idx.reshape(2, -1),)].max(
                    via.flatten(), mode="drop")
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

    def vet(self, step, bound, coords, links, data, idx, side):
        # args = (self, step, bound, coords, links, data, idx, side)
        # print(*(i.shape for i in args))
        # return
        def buffer(value, out, full):
            out = out.at[('key', 'secondary'), full].set(value)
            return full + 1, jax.lax.cond(
                    full == self.shape[1] - 1, lambda out: out.batched(),
                    lambda out: out, out)

        def replace(value, out, full):
            pos = out.max
            out = out.remove(pos)
            out = out.at[('key', 'secondary'), pos].set(value)
            out.insert(pos)
            return full, out

        # TODO: recalculate step data for coords, update loop bound, trim out
        def body(args):
            # primary/secondary bounds
            primary, secondary, value, out, full = args
            full, out = jax.lax.cond(
                    full < self.shape[1], buffer, replace, value, out, full)
            # TODO: remove sum
            return primary + 1000, secondary + 1000, value, out, full
        def cond(args):
            primary, secondary, value, out, full = args
            hi = self.key[self.max]
            return (
                    (primary < hi) | (
                        (jnp.isclose(primary, hi) &
                        (secondary < self.secondary[self.max])))
                    ) & ~self.contains(*value)
        def loop(a):
            coords, split, out, full = a
            out, full = jax.lax.while_loop(cond, body, (
                    *bound[(slice(None), *coords)],
                    step[(slice(None), *coords)], out, full))[-2:]
            return split[tuple(coords)], split, out, full
        out = SingularAVL(self.shape[1])
        for split, coords in zip(links, coords):
            out, full = jax.lax.while_loop(
                    lambda a: jnp.all(a[0] != -1), loop,
                    (coords, split, out, 0))[-2:]
        return jax.lax.cond(
                full < self.shape[1], lambda out: out.batched(),
                lambda out: out, out).tree_flatten()

@jax.tree_util.register_pytree_node_class
class Links(marginalized("splits", tail=jnp.int32(-1)), grouping(
        "Links", ("splits", "points", "size", "addresses"), ("head",))):
    @jax.jit
    def rebuild(self, step, bound, heap, data):
        order = jnp.tile(jnp.arange(self.shape[2]), (2, 1))
        side = jnp.tile(jnp.arange(2)[:, None], (1, self.shape[2]))
        return bound.vmap(in_axes=(0, None, 0, 0)).alongside(
                heap, step, self, in_axes=None).following(
                    data, order, side)

@jax.tree_util.register_pytree_node_class
class Bounds(grouping(
        "Bounds", ("splits", "points", "size"), ("distances", "indices"))):
    def following(self, heap, step, links, data, idx, side):
        return heap.vmap((0, 1, None, None, 0, 0)).alongside(
                step, self, in_axes=(None, None)).vet(
                    links.tail, links.head, data, idx, side)

class Candidates:
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
        init = jnp.full((self.spec.points, 2), -1)
        f = lambda x: jax.lax.scan(
                linker, init, (x, jnp.arange(self.spec.points)))
        res = jax.vmap(f)(jnp.stack(self))
        return Links.tree_unflatten((), res[::-1])

    # requires some re-computing, but low memory and high cache coherence
    def bound(self, idx0, idx1, data, heap, prune=False, dist=euclidean):
        row = heap.indirect[:, 0]
        tree = row.__class__
        children = heap.tree_flatten()[0]
        args = len(children)
        @partial(jax.vmap, in_axes=(0, 0, None, None, *(0,) * args))
        def row_row(x, y, data, *heap):
            return el_row(x, y, data, *heap)
        @partial(jax.vmap, in_axes=(0, None, None, None, *(None,) * args))
        def el_row(x, y, data, *heap):
            skip = x == -1
            res = el_el(x, y, data, *heap)
            out = jnp.where(skip, jnp.float32(jnp.inf)[None], res)
            lo = jnp.argmin(out)
            return out[lo], y[lo]
        @partial(jax.vmap, in_axes=(None, 0, None, None, *(None,) * args))
        def el_el(x, y, data, aux, *heap):
            d = dist(data[x], data[y])
            heap = tree.tree_unflatten(aux, heap)
            skip = (x == y) | (y == -1)
            if prune:
                skip |= heap.contains(d, y)
            return jnp.where(skip, jnp.inf, d)
        return row_row(self[idx0], self[idx1], data, heap.aux_data, *children)

    @partial(jax.jit, static_argnames=('prune', 'dist'))
    def bounds(self, data, heap, prune=False, dist=euclidean):
        res = tuple(self.bound(0, i, data, heap) for i in range(len(self)))
        return Bounds.tree_unflatten((), tuple(map(jnp.stack, zip(*res))))

class NNDCandidates(Candidates, grouping(
        "NNDCandidates", ("points", "size"), ("old", "new"))):
    pass

