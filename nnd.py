from collections import namedtuple
import jax.numpy as jnp
import jax

class GroupSetter:
    def __init__(self, group, idx, setter):
        self.group, self.idx, self.set = group, idx, setter

    def __getattr__(self, key):
        assert callable(getattr(self.group, key))
        return lambda *a, **kw: self.group.at[idx].set(
                getattr(self.group.slice(idx), key)(*a, **kw))

class GroupAt:
    def __init__(self, group):
        self.group = group

    def __getitem__(self, idx):
        def setter(*a):
            updates = (self.group[i].at[idx].set(x) for i, x in enumerate(a))
            static = (self.group[i] for i in range(len(a), len(self.group)))
            return self.group.tree_unflatten((), (*updates, *static))
        return GroupSetter(self.group, idx, setter)

class Group:
    def __new__(cls, *a, **kw):
        assert hasattr(cls, "spec")
        return super().__new__(cls, *a, **kw)

    def tree_flatten(self):
        return tuple(self), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return super().__new__(cls, *children)

    def slice(self, idx):
        return super().__new__(self.__class__, *(i[idx] for i in self))

    def at(self):
        return GroupAt(self)

    @property
    def shape(self):
        return self[0].shape

def namedgroup(clsname, dims, named, defaults):
    assert len(named) == len(defaults)
    GroupSpec = namedtuple(f"{clsname}Spec", dims)
    class GroupBase(Group, namedtuple(f"{clsname}Base", named)):
        def __new__(cls, *a, **kw):
            return super().__new__(cls, *(
                    jnp.full(GroupSpec(*a, **kw), i) for i in defaults))

        @property
        def spec(self):
            assert len(self) == len(GroupSpec._fields), "can't spec slice"
            return GroupSpec(self.shape)

    return type(clsname, (GroupBase,), {})

class Heap(Group):
    @property
    def order(self):
        return self[0]

    def swapped(self, i0, i1):
        return super().__new__(self.__class__, *(
                i.at[i0].set(i[i1]).at[i1].set(i[i0]) for i in self))

    def sifted(self, i):
        sel = (lambda x, y: x, lambda x, y: y)
        def cond(heap, i, broken):
            return not broken and i * 2 + 1 < heap.shape[0]

        def inner(heap, i, broken):
            left, right, swap = i * 2 + 1, i * 2 + 2, i
            lpivot = heap.order[swap] < heap.order[left]
            swap = jax.lax.cond(lpivot, *sel, left,  swap)
            rpivot = right < heap.shape[0] and \
                    heap.order[swap] < heap.order[right]
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
        heap = self
        for i in range(self.shape[0]):
            for j in range(self.shape[1] - 1, 0, -1):
                heap = heap.swapped((i, 0), (i, j)).sifted(0)
        return heap

    def ref(self, key):
        return getattr(self, key) if isinstance(type(key), str) else self[key]

    def push(self, *value, checked=()):
        lim = len(value)
        values = value + (None,) * (len(self) - lim)
        ins = super().__new__(self.__class__, *values)
        res = ins.order < self.order[0] and all((
                jnp.all(ins.ref(i)[None] != self.ref(i)) for i in checked))

        def init(heap):
            heap = heap.at[0].set(values)
            _, i, heap = jax.lax.while_loop(
                    lambda c, *a: c, inner, (True, 0, heap))
            return heap.at[i].set(values)

        def inner(continues, i, heap):
            left, right = 2 * i + 1, 2 * i + 2
            loob = left >= heap.shape[0]
            roob = right >= heap.shape[0]
            flipped = lambda: heap.order[left] >= heap.order[right]
            pivot = jax.lax.cond(roob or flipped(), lambda: left, lambda: right)
            swapping = not loob and ins.order < heap.order[pivot]
            return (swapping,) + jax.lax.cond(
                    swapping,
                    lambda: (pivot, heap.at[i].set(*heap.slice(i_swap)[:lim])),
                    lambda: (i, heap))
        return res, jax.lax.cond(res, init, lambda a: a, self)

@jax.tree_util.register_pytree_node_class
class NNDHeap(Heap, namedgroup(
        "Heap", ("points", "size"), ("distances", "indices", "flags"),
        (jnp.int32(-1), jnp.float32(jnp.inf), jnp.uint8(0)))):
    def build(self, max_candidates, rng_state):
        cur = Heap(self.spec.points, max_candidates)
        old = Heap(self.spec.points, max_candidates)
        for i in range(self.spec.points): # jit unrolling inefficiency; vmap
            for j in range(self.spec.size):
                pass

