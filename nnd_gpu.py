from functools import partial
import jax
import jax.numpy as jnp
from group import grouping

euclidean = jax.jit(lambda x, y: jnp.sqrt(jnp.sum((x - y) ** 2)))

class NNDHeap(grouping(
        "NNDHeap", ("points", "size"), ("distances", "indices", "flags"),
        (jnp.float32(jnp.inf), jnp.int32(-1), jnp.bool(False)))):
    pass

@jax.tree_util.register_pytree_node_class
class NNDHeap(NNDHeap):
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

    # requires some re-computing, but low memory and high cache coherence
    def bound(self, idx0, idx1, data, dist=euclidean):
        @partial(jax.vmap, in_axes=(0, None, None))
        def el_row(x, y, data):
            skip = x == -1
            out = jnp.where(skip, jnp.float32(jnp.inf)[None], el_el(x, y, data))
            lo = jnp.argmin(out)
            return y[lo], out[lo]
        @partial(jax.vmap, in_axes=(None, 0, None))
        def el_el(x, y, data):
            skip = (x == y) | (y == -1)
            # TODO: skip if y in x knn already
            return jnp.where(skip, jnp.inf, dist(data[x], data[y]))
        return jax.vmap(el_row, (0, 0, None))(self[idx0], self[idx1], data)

    @partial(jax.jit, static_argnames=('dist',))
    def bounds(self, data, dist=euclidean):
        return tuple(self.bound(0, i, data) for i in range(len(self)))

class NNDCandidates(Candidates, grouping(
        "NNDCandidates", ("points", "size"), ("old", "new"))):
    pass

