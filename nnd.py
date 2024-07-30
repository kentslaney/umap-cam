from collections import namedtuple
from functools import partial
import jax.numpy as jnp
from jax.experimental import checkify
import jax

class GroupSetter:
    def __init__(self, group, idx):
        self.group, self.idx = group, idx + ((slice(None),) * 2)[len(idx):]
        self.idx = (self.group.ref(self.idx[0]),) + self.idx[1:]

    def set(self, value):
        sel = range(len(self.group))[self.idx[0]]
        sel, value = ((sel,), (value,)) if isinstance(sel, int) else (
                sel, value * len(sel) if len(value) == 1 else value)
        return self.group.tree_unflatten(self.group.aux_data, tuple(
            self.group[i].at[self.idx[1:]].set(value[sel.index(i)]) if i in sel
            else self.group[i] for i in range(len(self.group))))

    def __getattr__(self, key):
        f = getattr(self.group.indirect[self.idx], key)
        assert callable(f)
        return lambda *a, **kw: self.set(f(*a, **kw))

class Shunt:
    @classmethod
    def skip(cls):
        return super(cls.mro()[cls.mro().index(Group) - 1], cls)

    def __new__(cls, *a):
        return cls.skip().__new__(cls, *a)

    def tree_flatten(self):
        return self.skip().tree_flatten(self)

    @classmethod
    def tree_unflatten(cls,  *a, **kw):
        return cls.skip().tree_unflatten(*a, **kw)

class GroupIndirect:
    def __init__(self, group):
        self.group = group

    def __getitem__(self, idx):
        sliced = self.group[idx]
        wrapper, wrapped = self.group.__class__, sliced.__class__
        if wrapped != wrapper:
            class Wrapping(wrapped, Shunt, wrapper):
                @classmethod
                def tree_unflatten(cls,  *a, **kw):
                    return super().tree_unflatten(*a, **kw)
            children = sliced.tree_flatten()[0]
            sliced = Wrapping.tree_unflatten(self.group.aux_data, children)
        return sliced


class GroupAt:
    def __init__(self, group):
        self.group = group

    def __getitem__(self, i):
        return GroupSetter(self.group, i if isinstance(i, tuple) else (i,))

class Group:
    def __new__(cls, *a, **kw):
        assert hasattr(cls, "spec")
        return super().__new__(cls, *a, **kw)

    def tree_flatten(self):
        return tuple(self), ()

    aux_keys = ()
    @property
    def aux_data(self):
        return self.tree_flatten()[1]

    @property
    def aux_dict(self):
        return dict(zip(self.aux_keys, self.aux_data))

    def aux_keyed(self, keying):
        meta = groupaux(**self.aux_dict)
        class res(meta, keying):
            pass
        return type(keying.__name__, (meta, keying), {})

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        assert aux_data == ()
        return super().__new__(cls, *children)

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        idx = (self.ref(idx[0]),) + idx[1:]
        if isinstance(range(len(self))[idx[0]], int):
            res = super().__getitem__(idx[0])
            return res[idx[1:]] if len(idx) > 1 else res
        clsname = self.__class__.__name__ + "Slice"
        dims = [
                i for i, j in zip(self.spec._fields, idx[1:])
                if not isinstance(j, slice)]
        if len(dims) == 0 and range(len(self))[idx[0]] == range(len(self)):
            if len(idx) == 1:
                return self
            return self.tree_unflatten(
                    self.aux_data, tuple(i[idx[1:]] for i in self))
        dims = [i for i in self.spec._fields if i not in dims]
        dims = dims if isinstance(self.spec, Named) else len(dims)
        names = self._fields[idx[0]]
        names = names if isinstance(self, Named) else len(names)
        res = grouping(clsname, dims, names)
        return self.aux_keyed(res).tree_unflatten(self.aux_data, tuple(
                i[idx[1:]] if len(idx) > 1 else i
                for i in super().__getitem__(idx[0])))

    @property
    def at(self):
        return GroupAt(self)

    @property
    def indirect(self):
        return GroupIndirect(self)

    @property
    def shape(self):
        return (len(self),) + self[0].shape

    def ref(self, key):
        return self._fields.index(key) if isinstance(key, str) else key

class Named:
    def __repr__(self):
        if not any("\n" in repr(i) for i in self):
            return super().__repr__()
        def arr_repr(arr):
            if not isinstance(arr, jnp.ndarray):
                return repr(array)
            if arr.aval is not None and arr.aval.weak_type:
                dtype_str = f'dtype={arr.dtype.name}, weak_type=True)'
            else:
                dtype_str = f'dtype={arr.dtype.name})'
            return f"partial(Array, {dtype_str})(\n{arr._value})".replace(
                    "\n", "\n" + " " * 8)
        eq, sep = "=", ",\n\n" + " " * 4
        data = zip(self._fields, map(arr_repr, self))
        data = sep.join(eq.join(i) for i in data)
        return f"{self.__class__.__name__}(\n{' ' * 4}{data}\n)"

def nameable(clsname, names=None):
    if names is None or isinstance(names, int):
        class GroupSize(tuple):
            def __new__(cls, *a):
                assert names is None or len(a) == names
                return super().__new__(cls, a)

            @property
            def _fields(self):
                return range(len(self))
    else:
        class GroupSize(Named, namedtuple(clsname, names)):
            pass
    return type(clsname, (GroupSize,), {})

def grouping(clsname, dims=None, names=None, defaults=None):
    assert names is None or defaults is None or len(names) == len(defaults)
    amount = (
            (None if defaults is None else len(defaults)) if names is None
            else names if isinstance(names, int) else len(names))
    GroupSpec = nameable(f"{clsname}Spec", dims)
    Container = nameable(f"{clsname}Base", amount if names is None else names)
    class GroupBase(Group, Container):
        def __new__(cls, *a, **kw):
            if defaults is None:
                checking = (i for i in a + tuple(kw.values()))
                against = next(checking).shape
                assert dims is None or len(against) == (
                        dims if isinstance(dims, int) else len(dims))
                assert all(i.shape == against for i in checking)
                return super().__new__(cls, *a, **kw)
            return super().__new__(cls, *(
                    jnp.full(GroupSpec(*a, **kw), i) for i in defaults))

        @property
        def spec(self):
            return GroupSpec(*self.shape[1:])

        @classmethod
        def grouped(cls, *a, names=None, sized=True):
            assert names is None or sized is True
            assert all(isinstance(i, cls) for i in a)
            condition = f"Of{len(a)}" if sized or names is not None else ""
            return grouping(
                    f"{clsname}Group{condition}",
                    None if dims is None else (clsname,) + dims,
                    names if names is not None or not sized else len(a))(*a)

    return type(clsname, (GroupBase,), {})

class Heap(Group):
    @property
    def order(self):
        return self[0]

    def swapped(self, i0, i1):
        i0 = i0 if isinstance(i0, tuple) else (i0,)
        i1 = i1 if isinstance(i1, tuple) else (i1,)
        return self.at[:, *i0].set(self[:, *i1]).at[:, *i1].set(self[:, *i0])

    def sifted(self, i):
        sel = (lambda x, y: x, lambda x, y: y)
        def cond(args):
            heap, i, broken = args
            return ~broken & (i * 2 + 1 < heap.order.shape[0])

        def inner(args):
            heap, i, broken = args
            left, right, swap = i * 2 + 1, i * 2 + 2, i
            lpivot = heap.order[swap] < heap.order[left]
            swap = jax.lax.cond(lpivot, *sel, left,  swap)
            rpivot = (right < heap.order.shape[0]) & (
                    heap.order[swap] < heap.order[right])
            swap = jax.lax.cond(rpivot, *sel, right, swap)

            broken = swap == i
            heap, i = jax.lax.cond(
                    broken,
                    lambda heap, i, swap: (heap, i),
                    lambda heap, i, swap: (heap.swapped(swap, i), swap),
                    *(heap, i, swap))
            return heap, i, broken
        return jax.lax.while_loop(cond, inner, (self, i, False))[0]

    def ascending(self):
        def inner(i, heap):
            for j in range(self.shape[2] - 1, 0, -1):
                heap = heap.swapped((i, 0), (i, j)).at[:, i, :j].sifted(0)
            return heap
        return jax.lax.fori_loop(0, self.shape[1], inner, self)

    # low memory but O(size); PyNNDescent uses python sets in high memory mode
    # most devices probably vectorize it since it's a contiguous chunk of int32s
    # a binary search might be worth profiling but GPU jobs are usually IO bound
    def check(self, ins, idx):
        return jnp.all(ins[idx][None] != self[idx])

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
            pivot = jax.lax.cond(roob | flipped(), lambda: left, lambda: right)
            swapping = ~loob & (ins.order < heap.order[pivot])
            return (swapping,) + jax.lax.cond(
                    swapping,
                    lambda: (pivot, heap.at[:, i].set(heap[:, pivot])),
                    lambda: (i, heap))
        return res, jax.lax.cond(res, init, lambda a: a, self)

    def pusher(self, *a, **kw):
        return self.push(*a, **kw)[1]

def groupaux(*required, **defaults):
    order = required + tuple(sorted(defaults.keys()))
    others = lambda kw: {k: v for k, v in kw.items() if k not in order}
    class GroupAux:
        def __new__(cls, *a, **kw):
            assert all(k in kw for k in required), \
                    f"missing {','.join(set(required) - set(kw))}"
            obj = super().__new__(cls, *a, **others(kw))
            obj.aux_keys += order
            for k in order:
                setattr(obj, k, kw[k] if k in kw else defaults[k])
            return obj

        def __repr__(self):
            aux_data = zip(order, self.aux_data)
            aux_data = ", ".join("=".join(map(str, i)) for i in aux_data)
            aux_data = f" with {aux_data}" if aux_data else ""
            return f"<{super().__repr__()}{aux_data} at {hex(id(self))}>"

        def tree_flatten(self):
            children, aux_data = super().tree_flatten()
            aux_data += tuple(getattr(self, k) for k in order)
            return children, aux_data

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            assert len(aux_data) >= len(order), f"{aux_data} {order}"
            cutoff = len(aux_data) - len(order)
            obj = super().tree_unflatten(aux_data[:cutoff], children)
            obj.aux_keys += order
            for k, v in zip(order, aux_data[cutoff:]):
                setattr(obj, k, v)
            assert all(hasattr(obj, k) for k in required)
            return obj
    return GroupAux

euclidean = jax.jit(lambda x, y: jnp.sqrt(jnp.sum((x - y) ** 2)))

@jax.tree_util.register_pytree_node_class
class NNDHeap(Heap, grouping(
        "NNDHeap", ("points", "size"), ("distances", "indices", "flags"),
        (jnp.float32(jnp.inf), jnp.int32(-1), jnp.uint8(0)))):
    @partial(jax.jit, static_argnames=('limit',))
    def build(self, limit, rng):
        def init(i, args):
            for j in range(self.shape[2]):
                conts = self.indices[i, j] < 0
                args = jax.lax.cond(conts, lambda *a: a[2:], inner, j, i, *args)
            return args
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
            for j in range(limit):
                mask = cur.indices[i] == heap.new.indices[i, j]
                # reset flag
                cur = cur.at["flags"].set(jnp.where(mask, 0, cur.flags[i]))
            return cur
        cur = jax.lax.fori_loop(0, self.shape[1], end, self)
        return cur, Candidates(*heap[:, "indices"]), rng

    @partial(jax.jit, static_argnames=('dist',))
    def randomize(self, data, rng, dist=euclidean):
        def inner(j, args):
            i, heap, rng = args
            rng, subkey = jax.random.split(rng)
            idx = jax.random.randint(subkey, (), 0, self.shape[1] - 1)
            idx += idx >= i
            heap = heap.at[:, i].pusher(
                    dist(data[idx], data[i]), idx, jnp.uint8(1),
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

class Candidates(grouping("Candidates", ("points", "size"), ("old", "new"))):
    @partial(jax.jit, static_argnames=('apply', 'dist'))
    def updates(self, apply, thresholds, data, *out, dist=euclidean):
        def inner(idx):
            start, end = self["new"], self[idx]
            def loop(k, j, i, out):
                p, q = start[i, j], end[i, k]
                d = dist(data[p], data[q])
                cond = (d < thresholds[p]) | (d < thresholds[q])
                return jax.lax.cond(
                        cond, lambda: apply(p, q, d, *out),
                        lambda: out)
            return loop
        def checked(idx, start, f):
            def outer(j, *args):
                i, out = (j, *args)[-2:]
                def inner(k, args):
                    return jax.lax.cond(
                            self[idx, i, k] >= 0, lambda *a: (*a[1:-1], f(*a)),
                            lambda *a: a[1:], k, *args)
                return jax.lax.fori_loop(
                        start(j), self.shape[2], inner, (j, *args))[-1]
            return outer
        f = checked("new", lambda i: 0, lambda j, i, out: checked(
                "old", lambda i: 0, inner("old"))(j, i, checked(
                        "new", lambda i: i + 1, inner("new"))(j, i, out)))
        return jax.lax.fori_loop(0, self.shape[1], f, out)

if __name__ == "__main__":
    config = (5, 4, 3)
    rng = jax.random.key(0)
    heap = NNDHeap(*config[:2])
    data = jnp.arange(heap.shape[1])
    heap, rng = heap.randomize(data, rng)
    heap, step, rng = heap.build(config[2], rng)
    print(f"{heap=}", f"{step=}", sep="\n\n", end="\n\n")
    heap, changes = heap.update(step, data)
    print(f"{heap=}", f"{changes=}", sep="\n\n", end="\n\n")

