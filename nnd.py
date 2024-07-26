from collections import namedtuple
import jax.numpy as jnp
import jax

class GroupSetter:
    def __init__(self, group, idx):
        self.group, self.idx = group, idx + ((slice(None),) * 2)[len(idx):]

    def set(self, value):
        sel = range(len(self.group))[self.idx[0]]
        sel, value = ((sel,), (value,)) if isinstance(sel, int) else (
                sel, value * len(sel) if len(value) == 1 else value)
        return self.group.tree_unflatten(None, tuple(
            self.group[i].at[self.idx[1:]].set(value[sel.index(i)]) if i in sel
            else self.group[i] for i in range(len(self.group))))

    def __getattr__(self, key):
        slicing = sliced = self.group[self.idx]
        wrapper, wrapped = self.group.__class__, sliced.__class__
        if wrapped != wrapper:
            class Wrapping:
                def __new__(cls, *a):
                    return tuple.__new__(cls, a)
            slicing = wrapped.__new__(
                    type("Wrapper", (wrapped, Wrapping, wrapper), {}), *sliced)
        f = getattr(slicing, key)
        assert callable(f)
        return lambda *a, **kw: self.set(f(*a, **kw))

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
        return tuple(self), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return super().__new__(cls, *children)

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
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
            return self.tree_unflatten(None, tuple(i[idx[1:]] for i in self))
        dims = [i for i in self.spec._fields if i not in dims]
        dims = dims if isinstance(self.spec, Named) else len(dims)
        names = self._fields[idx[0]]
        names = names if isinstance(self, Named) else len(names)
        return grouping(clsname, dims, names)(*(
                i[idx[1:]] if len(idx) > 1 else i
                for i in super().__getitem__(idx[0])))

    @property
    def at(self):
        return GroupAt(self)

    @property
    def shape(self):
        return (len(self),) + self[0].shape

    @property
    def order(self):
        return self[0]

    def swapped(self, i0, i1):
        i0 = i0 if isinstance(i0, tuple) else (i0,)
        i1 = i1 if isinstance(i1, tuple) else (i1,)
        return self.at[:, *i0].set(self[:, *i1]).at[:, *i1].set(self[:, *i0])

class Named:
    pass

def nameable(clsname, names=None):
    if names is None or isinstance(names, int):
        class GroupSize(tuple):
            def __new__(cls, *a):
                assert names is None or len(a) == names
                return super().__new__(cls, a)

            @property
            def _fields(self):
                return range(len(self))
        return type(clsname, (GroupSize,), {})
    return type(clsname, (namedtuple(clsname, names), Named), {})

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

    def sorted(self):
        def inner(i, heap):
            for j in range(self.shape[2] - 1, 0, -1):
                heap = heap.swapped((i, 0), (i, j)).at[:, i, :j].sifted(0)
            return heap
        return jax.lax.fori_loop(0, self.shape[1], inner, self)

    def ref(self, key):
        return getattr(self, key) if isinstance(key, str) else self[key]

    def push(self, *value, checked=()):
        assert len(value) <= len(self), \
                f"can't push {len(value)} values to a group of {len(self)}"
        value = tuple(jnp.asarray(i) if isinstance(i, (int, float)) else i for i in value)
        ins = type("Ins", (Heap, self[:len(value), 0].__class__), {})(*value)
        res = (ins.order < self.order[0]) & jnp.all(jnp.asarray(tuple(
                jnp.all(ins.ref(i)[None] != self.ref(i)) for i in checked)))

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

@jax.tree_util.register_pytree_node_class
class NNDHeap(Heap, grouping(
        "NNDHeap", ("points", "size"), ("distances", "indices", "flags"),
        (jnp.float32(jnp.inf), jnp.int32(-1), jnp.uint8(0)))):
    def build(self, limit, rng):
        def init(i, args):
            for j in range(self.shape[2]):
                conts = self.indices[i, j] < 0
                args = jax.lax.cond(conts, lambda *a: a[2:], inner, j, i, *args)
            return args
        def inner(j, i, heap, rng):
            rng, subkey = jax.random.split(rng)
            d = jax.random.uniform(subkey)
            idx = self.indices[i, j]
            isn = jax.lax.cond(self.flags[i, j], lambda: 1, lambda: 0)
            heap = heap.at[isn, :, i].pusher(d, idx, checked=("indices",))
            heap = heap.at[isn, :, idx].pusher(d, i, checked=("indices",))
            return heap, rng
        heap, rng = jax.lax.fori_loop(0, self.shape[1], init, (self.grouped(
                *((self.__new__(self.__class__, self.spec.points, limit),) * 2),
                self, names=("old", "new", "cur")), rng))
        def end(i, heap):
            for j in range(limit):
                heap = heap.at[2, 2, jnp.where(
                    heap.cur.indices[i] == heap.new.indices[i, j])].set(0)
            return heap
        heap = jax.lax.fori_loop(0, self.shape[1], end, heap)
        return heap.cur, heap.new.indices, heap.old.indices, rng

    def randomize(self, dist, rng):
        def inner(j, args):
            i, heap, rng = args
            rng, subkey = jax.random.split(rng)
            idx = jax.random.randint(subkey, (), 0, self.shape[1])
            heap = heap.at[:, i].pusher(
                    dist(idx, i), idx, jnp.uint8(1), checked=("indices",))
            return heap, rng
        def init(i, args):
            heap, rng = args
            cond = heap.indices[i, 0] < 0.
            return jax.lax.cond(cond, lambda i, heap, rng: jax.lax.fori_loop(
                    jnp.sum(heap.indices[i] >= 0), heap.shape[1], inner,
                    (i, heap, rng)), lambda *a: a[1:], i, *args)
        return jax.lax.fori_loop(0, self.shape[1], init, (self, rng))

