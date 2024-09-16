from collections import namedtuple
import jax.numpy as jnp

class GroupSetter:
    def __init__(self, group, idx):
        self.group, self.idx = group, idx + ((slice(None),) * 2)[len(idx):]
        self.idx = (self.group.ref(self.idx[0]),) + self.idx[1:]

    def set(self, value):
        sel = range(len(self.group))[self.idx[0]]
        sel, value = ((sel,), (value,)) if isinstance(sel, int) else (
                sel, value * len(sel) if len(value) == 1 else value)
        return self.group.tree_unflatten(
                self.group.aux_data, tuple(
                    self.group[i].at[self.idx[1:]].set(value[sel.index(i)])
                    if i in sel else self.group[i]
                    for i in range(len(self.group))) + self.group.out)

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

    @property
    def out(self):
        return self.tree_flatten()[0][len(self):]

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        assert aux_data == ()
        if cls._dtypes is not None:
            children = tuple(jnp.asarray(j, dtype=i) for i, j in zip(
                    cls._dtypes, children))
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
                i for i, j in zip(self.spec._names, idx[1:])
                if not isinstance(j, slice)]
        if len(dims) == 0 and range(len(self))[idx[0]] == range(len(self)):
            if len(idx) == 1:
                return self
            return self.tree_unflatten(
                    self.aux_data, tuple(i[idx[1:]] for i in self) + self.out)
        dims = [i for i in self.spec._names if i not in dims]
        dims = dims if isinstance(self.spec, Named) else len(dims)
        names = self._names[idx[0]]
        names = names if isinstance(self, Named) else len(names)
        res = grouping(clsname, dims, names)
        return self.aux_keyed(res).tree_unflatten(self.aux_data, tuple(
                i[idx[1:]] if len(idx) > 1 else i
                for i in super().__getitem__(idx[0])) + self.out)

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

    @classmethod
    def where(cls, cond, x, y):
        assert x.aux_data == y.aux_data
        (xc, xa), (yc, _) = x.tree_flatten(), y.tree_flatten()
        return cls.tree_unflatten(xa, tuple(
                jnp.where(cond, i, j) for i, j in zip(xc, yc)))

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

    @property
    def _names(self):
        return self._fields

def nameable(clsname, names=None):
    if names is None or isinstance(names, int):
        class GroupSize(tuple):
            def __new__(cls, *a):
                assert names is None or len(a) == names
                return super().__new__(cls, a)

            @property
            def _names(self):
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
        _dtypes = defaults and tuple(i.dtype for i in defaults)
        def __new__(cls, *a, **kw):
            if defaults is None:
                checking = (i for i in a + tuple(kw.values()))
                against = next(checking).shape
                assert dims is None or len(against) == (
                        dims if isinstance(dims, int) else len(dims))
                assert all(i.shape == against for i in checking)
                return super().__new__(cls, *a, **kw)
            # if erroring double check cls is registered as a pytree node class
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

def groupaux(*required, **optional):
    order = required + tuple(sorted(optional.keys()))
    others = lambda kw: {k: v for k, v in kw.items() if k not in order}
    class GroupAux:
        def __new__(cls, *a, **kw):
            assert all(k in kw for k in required), \
                    f"missing {','.join(set(required) - set(kw))}"
            obj = super().__new__(cls, *a, **others(kw))
            obj.aux_keys += order
            for k in order:
                setattr(obj, k, kw[k] if k in kw else optional[k])
            return obj

        def __repr__(self):
            aux_data = zip(order, (getattr(self, i) for i in order))
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

def outgroup(*required, **optional):
    count = len(required) + len(optional)
    swap = lambda a, b: (a + b[len(b) - count:], b[:len(b) - count])
    class OutGroup(groupaux(*required, **optional)):
        def tree_flatten(self):
            return swap(*super().tree_flatten())

        @classmethod
        def tree_unflatten(self, aux_data, children):
            return super().tree_unflatten(*swap(aux_data, children))

        def aux_keyed(self, keying):
            keying = super().aux_keyed(keying)
            meta = outgroup(*required, **optional)
            class res(meta, keying):
                pass
            return type(keying.__name__, (meta, keying), {})
    return OutGroup

