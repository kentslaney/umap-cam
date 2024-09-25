from functools import partial
import jax
import jax.numpy as jnp
from group import grouping, group_alias, dim_alias, marginalized, groupaux
from avl import MaxAVL, AVLs

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

@jax.tree_util.register_pytree_node_class
class Vetted(groupaux(dist=euclidean), AVLs):
    @partial(jax.jit)
    def pairs(self, coords, step, data, side):
        def body(el, ref):
            skip = (el == -1) | (el == ref)
            return jnp.where(skip, jnp.inf, self.dist(el, ref))
        (row, col), ref = coords, jnp.stack(step)[(side, *coords)]
        distances = jax.vmap(body, (0, None))(step[0, row], ref)
        indices = jnp.where(jnp.isfinite(distances), step[0, row], -1)
        return jax.lax.sort((distances, indices), num_keys=2)

    def vet(self, heap, step, bound, coords, links, data, side):
        args = (self, heap, step, bound, coords, links, data, side)
        print(*(i.shape for i in args))
        print(self.pairs(coords, step, data, side))
        def body(args):
            coords, out = args
            return coords, out
        return jax.lax.while_loop(
                lambda a: jnp.all(a[0] != -1), body, (coords, self))[1]

@jax.tree_util.register_pytree_node_class
class Links(
        marginalized("splits", "points", "addresses", tail=jnp.int32(-1)),
        grouping("Links", ("splits", "points", "size", "addresses"), ("head",))
        ):
    @jax.jit
    def rebuild(self, step, bound, heap, data, dist=euclidean):
        return jax.lax.scan(
                bound.indirect[:, 0].following, (
                    Vetted(self.spec.points, self.spec.size, dist=dist),
                    heap, step, data),
                (bound, self, jnp.arange(2)))[0][0].resolve()

    def follow(self, tail):
        # each point sampled a maximum of once per node
        dense, pos = jnp.full((self.spec.points, 2), -1), 0
        def body(args):
            tail, dense, pos = args
            return self.head[(*tail,)], dense.at[pos].set(tail), pos + 1
        return jax.lax.while_loop(
                lambda a: jnp.all(a[0] != -1), body, (tail, dense, pos))[1]

    def walk(self):
        split, ax = hasattr(self.spec, "splits"), self.spec.index("points")
        f = self.vmap().follow if split else self.follow
        return jax.vmap(f, in_axes=ax, out_axes=int(split))(self.tail)

    def show(self, dense, *data, all=False):
        lens = jnp.sum(jnp.all(dense != -1, axis=-1), axis=-1).T
        opt = jnp.get_printoptions()
        linewidth, edgeitems = opt['linewidth'], opt['edgeitems']
        pad = [
                "{:0" + str(len(str(getattr(self.spec, i)))) + "}"
                for i in ['points', 'size']]
        splits = hasattr(self.spec, "splits")
        def fmt(coords, last, *data):
            end = "" if last else " -> "
            s = [
                    ("{:.3f}" if jnp.isfinite(i) else "  {}")
                    if i.dtype == jnp.float32 else pad[0] for i in data]
            extra = (" " + " ".join(s)).format(*data) if data else ""
            return f"({pad[0]}, {pad[1]})".format(*coords) + extra + end
        def wraps(i, row, size, out, line, *data):
            upcoming = fmt(row[i], i == size - 1, *(
                    dat[(*row[i],)] for dat in data))
            if len(line) + len(upcoming) > linewidth:
                out, line = out + line + "\n", " " * (18 if splits else 12)
            line += upcoming
            return out, line
        def body(i, size, row):
            out, line = "", str(i).ljust(4) + "  "
            if splits:
                _line = line
                for i in range(self.shape[1]):
                    line = _line + str(size[i]).ljust(4) + "  " + \
                            str(i).ljust(4) + "  "
                    for j in jnp.arange(size[i]):
                        out, line = wraps(j, row[i], size[i], out, line, *(
                                k[i] for k in data))
                    out += line + "\n"
                out, line = out[:-1], ""
            else:
                line += str(size).ljust(4) + "  "
                for i in jnp.arange(size):
                    out, line = wraps(i, row, size, out, line, *data)
            print(out + line)
        extra, idx = ("side  ", (slice(None),)) if splits else ("", ())
        jax.debug.print(f"node  size  {extra}candidates", ordered=True)
        callback = lambda i, a: jax.debug.callback(
                body, i, lens[i], dense[(*idx, i)], ordered=True)
        if all or self.spec.points <= 2 * edgeitems:
            jax.lax.fori_loop(0, self.spec.points, callback, None)
        else:
            jax.lax.fori_loop(0, edgeitems, callback, None)
            multiplier = int(not splits or self.shape[1])
            hidden = (self.spec.points - 2 * edgeitems) * multiplier
            jax.debug.print(f"... ({hidden} more) ...")
            jax.lax.fori_loop(
                    self.spec.points - edgeitems,
                    self.spec.points, callback, None)

@jax.tree_util.register_pytree_node_class
class Bounds(grouping(
        "Bounds", ("splits", "points", "size"), ("distances", "indices"))):
    def following(self, carry, args):
        bounds, links, side = args
        out, heap, step, data = carry
        out = out.remap((0, 0, None, None, None)).alongside(
                heap, step, bounds, in_axes=(0, None, None, None)).vet(
                    links.tail, links.head, data, side)
        return (out, heap, step, data), None

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
            return row_el(x, y, data, *heap)
        @partial(jax.vmap, in_axes=(None, 0, None, None, *(None,) * args))
        def row_el(x, y, data, *heap):
            skip = y == -1
            out = el_el(x, y, data, *heap)
            out = jnp.where(skip, jnp.float32(jnp.inf)[None], out)
            lo = jnp.argmin(out)
            return out[lo], jnp.where(jnp.isfinite(out[lo]), x[lo], -1)
        @partial(jax.vmap, in_axes=(0, None, None, None, *(None,) * args))
        def el_el(x, y, data, aux, *heap):
            d = dist(data[x], data[y])
            heap = tree.tree_unflatten(aux, heap)
            skip = (x == y) | (x == -1)
            if prune:
                skip |= heap.contains(d, x)
            return jnp.where(skip, jnp.inf, d)
        return row_row(self[idx0], self[idx1], data, heap.aux_data, *children)

    @partial(jax.jit, static_argnames=('prune', 'dist'))
    def bounds(self, data, heap, prune=True, dist=euclidean):
        res = tuple(self.bound(0, i, data, heap) for i in range(len(self)))
        return Bounds.tree_unflatten((), tuple(map(jnp.stack, zip(*res))))

class NNDCandidates(Candidates, grouping(
        "NNDCandidates", ("points", "size"), ("old", "new"))):
    pass

