from functools import partial
import jax
import jax.numpy as jnp
from group import grouping, group_alias, dim_alias, marginalized, groupaux
from avl import MaxAVL, AVLs, Predecessors

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
            order = jax.random.permutation(subkey, order, 1, True)
            ordered = jax.lax.sort(jnp.stack((
                self.flags, order, self.indices), 0), num_keys=1)[2]
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
                    (jnp.arange(limit) < count) & (ref < 0),
                    forward, ref)
        ref = jax.vmap(jax.vmap(backfill))(counts, ordered, ref)
        update = self.indices[:, :limit] != ref[1, :, ::-1]
        unset = ((0, 0), (0, max(0, self.shape[2] - limit)))
        update = jnp.pad(update, unset, constant_values=True)
        update = self.at["flags"].set(self.flags & update)
        return update, NNDCandidates(*ref, data_points=self.spec.points), rng

    def randomize(self, data, rng, dist=euclidean):
        def outer(rng, i, existing, flags):
            idx = jax.random.choice(
                    rng, self.shape[1] - 1, (self.shape[2],), False)
            idx = jnp.where(existing == -1, idx + (idx >= i), existing)
            d = jax.vmap(dist, (0, None))(data[idx], data[i])
            _, flags = jax.lax.sort_key_val(d, existing == -1)
            d, idx = jax.lax.sort_key_val(d, idx)
            return d[::-1], idx[::-1], flags
        rng = jax.random.split(rng, self.shape[1] + 1)
        res = self.at[('distances', 'indices', 'flags'),].set(jax.vmap(outer)(
                rng[1:], jnp.arange(self.shape[1]), self.indices, self.flags))
        return res.remap().batched(), rng[0]

    def apply(self, update):
        def body(heap, primary, secondary):
            return heap.push(primary, secondary)
        def cond(heap, update):
            return jax.lax.cond(
                    update[1] != -1, body, lambda *a: heap, heap, *update), None
        return jax.lax.scan(cond, self, tuple(update[('key', 'secondary'),]))[0]

    def update(self, data, candidates, dist=euclidean):
        bounds = candidates.bounds(data, self)
        links = candidates.links()
        filtered = links.rebuild(candidates, bounds, self, data, dist)
        heap = self.remap('points').alongside(filtered, in_axes='trees').apply()
        return heap, jnp.sum(filtered.secondary != -1)

@jax.tree_util.register_pytree_node_class
class Filtered(groupaux(dist=euclidean), marginalized("trees", ceil=-1), AVLs):
    @partial(jax.jit)
    def pairs(self, coords, step, data, side):
        def body(el, ref):
            skip = (ref == -1) | (el == -1) | (el == ref)
            return jnp.where(skip, jnp.inf, self.dist(data[el], data[ref]))
        (row, col), ref = coords, jnp.stack(step)[(side, *coords)]
        distances = jax.vmap(body, (0, None))(step[0, row], ref)
        indices = jnp.where(jnp.isfinite(distances), step[0, row], -1)
        return jax.lax.sort((distances, indices), num_keys=2)

    def filter(self, heap, step, bound, coords, links, data, side):
        def push(out, ceil, *value):
            sign = out.sign(*value, out.max)
            over = heap.spec.size <= out.full + heap.full
            full = out.full >= out.spec.size
            update = (~full & over & (sign == 1))
            replace = full & (sign == -1)
            return jax.lax.cond(
                    replace, lambda ceil: (jax.lax.cond(
                        full, lambda: out.replace(*value), lambda: out), ceil),
                    lambda ceil: jax.lax.cond(
                        (sign == -1) | (heap.sign(*value, ceil.value) == -1),
                        lambda: (out.push(*value), ceil), lambda: (out, ceil)),
                    jax.lax.cond(update, lambda: ceil.next(), lambda: ceil))

        @jax.jit
        def recalculate(coords, out, ceil):
            pairs = out.pairs(coords, step, data, side)
            valid = jax.vmap(lambda *a: heap.sign(*a, heap.max))(*pairs)
            valid = (valid == -1) & (pairs[1] != -1)
            fit = jnp.sum(valid)
            valid = ~jax.vmap(heap.contains)(*pairs)
            return jax.lax.fori_loop(0, fit, lambda i, args: (*jax.lax.cond(
                    valid[i], lambda: push(*args[:2], args[2][i], args[3][i]),
                    lambda: args[:2]), *args[2:]), (out, ceil, *pairs))[:2]

        def body(args):
            coords, out, ceil = args
            lo = bound[(slice(None), *coords)]
            out, ceil = jax.lax.cond(
                    (heap.sign(*lo, heap.max) == -1) & out.includable(*lo),
                    recalculate, lambda *a: a[1:], *args)
            return links[(*coords,)], out, ceil
        ceil = Predecessors(heap, self.ceil)
        _, out, ceil = jax.lax.while_loop(
                lambda a: jnp.all(a[0] != -1), body, (coords, self, ceil))
        out.ceil = ceil.value
        return out

    def __repr__(self):
        return AVLs.__repr__(self) + "\nand dist = " + repr(self.dist)

@jax.tree_util.register_pytree_node_class
class Links(
        marginalized("splits", "data_points", "addresses", tail=jnp.int32(-1)),
        grouping("Links", ("splits", "points", "size", "addresses"), ("head",))
        ):
    @partial(jax.jit, static_argnames=("dist",))
    def rebuild(self, step, bound, heap, data, dist=euclidean):
        idx = jnp.arange(step.shape[0])
        return jax.lax.scan(
                bound.indirect[:, 0].following, (
                    Filtered(heap.spec.points, self.spec.size, dist=dist),
                    heap, step, data), (bound, self, idx))[0][0]

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

    def show(self, dense, *data, full=False, **kw):
        lens = jnp.sum(jnp.all(dense != -1, axis=-1), axis=-1).T
        opt = jnp.get_printoptions()
        linewidth, edgeitems = opt['linewidth'], opt['edgeitems']
        pad = [
                "{:0" + str(len(str(getattr(self.spec, i)))) + "}"
                for i in ['points', 'size']]
        splits = hasattr(self.spec, "splits")
        row_data, data = map(
                lambda x: tuple(filter(lambda x: x is not None, x)), zip(*[
                    (i, None) if i.ndim <= splits + 1 else (None, i)
                    for i in data]))
        names, values = zip(*sorted(kw.items())) if kw else ((), ())
        assert all(i.ndim <= splits + 1 for i in values)
        row_data += values
        row_data = [jnp.stack((i, i)) if i.ndim == 1 else i for i in row_data] \
                if splits else row_data
        names = tuple(
                f"info{i}" for i in range(len(row_data) - len(names))) + names
        names = [i.ljust(5) for i in names]
        def fmt_str(data):
            return ("{:.3f}" if jnp.isfinite(data) else "  {}") \
                    if data.dtype == jnp.float32 else pad[0]
        def fmt(coords, last, *data):
            end = "" if last else " -> "
            fmt_strs = [fmt_str(i) for i in data]
            extra = (" " + " ".join(fmt_strs)).format(*data) if data else ""
            return f"({pad[0]}, {pad[1]})".format(*coords) + extra + end
        def wraps(i, row, size, out, line, indent, *data):
            upcoming = fmt(row[i], i == size - 1, *(
                    dat[(*row[i],)] for dat in data))
            if len(line) + len(upcoming) > linewidth:
                out = out + line + "\n"
                line = " " * (indent + sum(map(len, names)) + 2 * len(row_data))
            line += upcoming
            return out, line
        def body(n, size, row):
            out, line = "", str(n).ljust(4) + "  "
            if splits:
                _line = line
                for i in range(self.shape[1]):
                    line = _line + str(size[i]).ljust(4) + "  " + \
                            str(i).ljust(4) + "  "
                    for j in range(len(row_data)):
                        info = row_data[j][i][n]
                        line += fmt_str(info).format(info).ljust(
                                len(names[j])) + "  "
                    for j in jnp.arange(size[i]):
                        out, line = wraps(j, row[i], size[i], out, line, 18, *(
                                k[i] for k in data))
                    out += line + "\n"
                out, line = out[:-1], ""
            else:
                line += str(size).ljust(4) + "  "
                for i in range(len(row_data)):
                    info = row_data[i][n]
                    line += fmt_str(info).format(info).ljust(
                            len(names[i])) + "  "
                for i in jnp.arange(size):
                    out, line = wraps(i, row, size, out, line, 12, *data)
            print(out + line)
        extra, idx = ("side  ", (slice(None),)) if splits else ("", ())
        for i in range(len(row_data)):
            extra += names[i] + "  "
        jax.debug.print(f"node  size  {extra}candidates", ordered=True)
        callback = lambda i, a: jax.debug.callback(
                body, i, lens[i], dense[(*idx, i)], ordered=True)
        if full or self.spec.points <= 2 * edgeitems:
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
        bounds = self.tree_unflatten(*bounds.tree_flatten()[::-1])
        out, heap, step, data = carry
        out = out.remap((0, 0, None, None, None)).alongside(
                heap, step, bounds, in_axes=(0, None, None)).filter(
                    links.tail, links.head, data, side)
        return (out, heap, step, data), None

class Candidates(groupaux("data_points")):
    # could be built iteratively and stored
    # jax.scan over rows to create a linked list of reverse neighbors
    def links(self):
        def linker(head, args):
            node, i = args
            oob = jnp.where(node == -1, self.data_points, node)
            res = jnp.where((node == -1)[:, None], -1, head[node])
            col = jnp.arange(self.spec.size)
            coords = jnp.stack((jnp.broadcast_to(i, self.spec.size), col), -1)
            return head.at[oob].set(coords, mode="drop"), res
        init = jnp.full((self.data_points, 2), -1)
        f = lambda x: jax.lax.scan(
                linker, init, (x, jnp.arange(self.spec.points)))
        res = jax.vmap(f)(jnp.stack(self))
        return Links.tree_unflatten((), res[::-1])

    # requires some re-computing, but low memory and high cache coherence
    def bound(self, idx0, idx1, data, heap, prune=True, dist=euclidean):
        tree = heap.indirect[:, 0].__class__
        args = len(heap.tree_flatten()[0])

        @partial(jax.vmap, in_axes=(0, 0, None))
        def row_row(x, y, data):
            return row_el(x, y, data)
        @partial(jax.vmap, in_axes=(None, 0, None))
        def row_el(x, y, data):
            skip = y == -1
            out = el_el(x, y, data, *heap[:, y].tree_flatten())
            out = jnp.where(skip, jnp.float32(jnp.inf)[None], out)
            lo = jnp.argmin(out)
            return out[lo], jnp.where(jnp.isfinite(out[lo]), x[lo], -1)
        @partial(jax.vmap, in_axes=(0, None, None, None, None))
        def el_el(x, y, data, aux, heap):
            heap = tree.tree_unflatten(heap, aux)
            d = dist(data[x], data[y])
            skip = (x == y) | (x == -1)
            if prune:
                skip |= heap.contains(d, x)
            return jnp.where(skip, jnp.inf, d)
        return row_row(self[idx0], self[idx1], data)

    @partial(jax.jit, static_argnames=('prune', 'dist'))
    def bounds(self, data, heap, prune=True, dist=euclidean):
        res = tuple(self.bound(
                0, i, data, heap, prune, dist) for i in range(len(self)))
        return Bounds.tree_unflatten((), tuple(map(jnp.stack, zip(*res))))

@jax.tree_util.register_pytree_node_class
class NNDCandidates(Candidates, grouping(
        "NNDCandidates", ("points", "size"), ("old", "new"))):
    pass

@jax.tree_util.register_pytree_node_class
class RPCandidates(groupaux("total"), Candidates, grouping(
        "RPCandidates", ("points", "size"))):
    pass

