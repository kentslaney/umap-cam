from collections import namedtuple
import jax
import jax.numpy as jnp

class GroupSetter:
    def __init__(self, group, idx):
        self.group, self.idx = group, idx + ((slice(None),) * 2)[len(idx):]

    def set(self, value):
        return self.group.setter(self.idx, value)

    def get(self):
        return self.group[self.idx]

    def add(self, value):
        return self.group.setter(self.idx, self.get() + value)

    def multiply(self, value):
        return self.group.setter(self.idx, self.get() * value)

    def divide(self, value):
        return self.group.setter(self.idx, self.get() / value)

    def power(self, value):
        return self.group.setter(self.idx, self.get() ** value)

    def min(self, value):
        return self.group.setter(self.idx, jnp.minimum(self.get(), value))

    def max(self, value):
        return self.group.setter(self.idx, jnp.maximum(self.get(), value))

    def __getattr__(self, key):
        f = getattr(self.group.indirect[self.idx], key)
        assert callable(f)
        return lambda *a, **kw: self.set(f(*a, **kw))

registered = {}
class GroupIndirect:
    def __init__(self, group):
        self.group = group

    def __getitem__(self, idx):
        sliced = self.group[idx]
        wrapper, wrapped = self.group.__class__, sliced.__class__
        if wrapped != wrapper:
            wraps = type("wraps", (), {
                k: getattr(wrapper, k) for k in dir(wrapper)
                if not hasattr(wrapped, k)})
            class Wrapping(wrapped, wraps):
                @classmethod
                def tree_unflatten(cls,  *a, **kw):
                    return super().tree_unflatten(*a, **kw)

            hashable = (wrapper, *self.group.subgroup(idx)[1:])
            if hashable in registered:
                Wrapping = registered[hashable]
            else:
                registered[hashable] = jax.tree_util.register_pytree_node_class(
                        Wrapping)

            children = sliced.tree_flatten()[0]
            sliced = Wrapping.tree_unflatten(self.group.aux_data, children)
        return sliced

class GroupAt:
    def __init__(self, group):
        self.group = group

    def __getitem__(self, i):
        return GroupSetter(self.group, i if isinstance(i, tuple) else (i,))

class GroupMap:
    def __init__(self, group, remap, in_axes=0, *a, **kw):
        self.group, self.remap = group, remap
        self.in_axes, self.a, self.kw = in_axes, a, kw
        self.sliced, self.flat, self.side_axes, self.aux_data = ((),) * 4

    def alongside(self, *a, in_axes=0):
        in_axes = (in_axes,) * len(a) if isinstance(
                in_axes, int) or in_axes is None else in_axes
        sliced, flat, aux, in_axes = zip(*(
                self._axis(*i) for i in zip(in_axes, a)))
        self.flat, self.sliced = self.flat + flat, self.sliced + sliced
        self.side_axes += sum(in_axes, ())
        self.aux_data += aux
        return self

    def __getattr__(self, key):
        assert callable(getattr(self.group, key))
        return lambda *a: self._map(key, a)

    def _axis(self, in_axes, group, *a):
        _in_axes = (in_axes,) if isinstance(in_axes, int) else in_axes
        _in_axes = (None,) if _in_axes is None else _in_axes
        _in_axes = (group.ref(_in_axes[0]),) + tuple(_in_axes[1:])

        if _in_axes[0] is None:
            sliced = group.__class__
        else:
            sliced = (*(slice(None),) * (_in_axes[0] + 1), 0)
            sliced = group.indirect[sliced].__class__
        children, aux = group.tree_flatten()

        _in_axes = _in_axes[:1] * len(children) + _in_axes[1:]
        _in_axes += (in_axes,) * (len(a) if isinstance(in_axes, int) else 0)
        return sliced, children, aux, _in_axes

    def _unflatten(self, a):
        sizes, bounds, total = map(len, ((),) + self.flat), [], 0
        for i in sizes:
            total += i
            bounds.append(total)
        return tuple(
                sliced.tree_unflatten(aux, a[begin:end])
                for sliced, aux, begin, end in zip(
                    self.sliced, self.aux_data, bounds[:-1], bounds[1:]))

    def _map(self, key, a):
        sliced, flat, aux, in_axes = self._axis(self.in_axes, self.group, *a)
        in_axes = in_axes[:len(flat)] + self.side_axes + in_axes[len(flat):]
        _flat, aux_out = flat + sum(self.flat, ()), ()
        def g(*a):
            nonlocal aux_out
            subset = (sliced.tree_unflatten(aux, a[:len(flat)]),)
            subset += self._unflatten(a[len(flat):len(_flat)])
            res = getattr(sliced, key)(*subset, *a[len(_flat):])
            if self.remap:
                res, aux_out = sliced.tree_flatten(res)
                return res
            return res
        res = jax.vmap(g, in_axes, *self.a, **self.kw)(*_flat, *a)
        if self.remap:
            return self.group.tree_unflatten(aux_out, res)
        return res

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
        return keying

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

    def subgroup(self, idx):
        assert idx[0] != None
        expanding = sum(i == Ellipsis for i in idx)
        if expanding:
            assert expanding == 1
            expanding = idx.index(Ellipsis)
            new_axes = idx.count(None)
            idx = (
                    idx[:expanding] +
                    (slice(None),) * (self.ndim - len(idx) + 1 + new_axes) +
                    idx[expanding + 1:])
        idx = (self.ref(idx[0]),) + idx[1:]
        fields = tuple(range(len(self)))
        sel = tuple(fields[i] for i in idx[0]) \
                if isinstance(idx[0], tuple) else fields[idx[0]]
        return idx, sel, tuple(fields)

    def outslice(self, idx, value=None):
        return ()

    def dict_dim(self, idx):
        assert isinstance(idx, dict)
        mapping = {k: self.spec.index(k) for k, v in idx.items()}
        assert len(mapping.values()) == len(set(mapping.values()))
        mapping = {mapping[k] + 1: v for k, v in idx.items()}
        return tuple(
                mapping.get(i, slice(None))
                for i in range(max(mapping.keys()) + 1))

    def __getitem__(self, idx):
        if isinstance(idx, dict):
            idx = self.dict_dim(idx)
        if not isinstance(idx, tuple):
            idx = (idx,)
        idx, sel, fields = self.subgroup(idx)
        if isinstance(sel, int):
            res = super().__getitem__(idx[0])
            return res[idx[1:]] if len(idx) > 1 else res
        clsname = self.__class__.__name__ + "Slice"
        old_axes = [i for i in idx[1:] if i is not None]
        new_axes = idx.count(None)
        # unflattened dims (fields or indices)
        dims = [
                i for i, j in zip(self.spec._names, old_axes)
                if not isinstance(j, slice) and (
                    not isinstance(j, jax.Array) or j.ndim == 0)]
        if len(dims) == 0 and sel == fields and new_axes == 0:
            if len(idx) == 1:
                return self
            return self.tree_unflatten(
                    self.aux_data, tuple(i[idx[1:]] for i in self) +
                    self.outslice(idx))
        # dim names if available
        if new_axes:
            dims = len(dims) + new_axes
        else:
            dims = [i for i in self.spec._names if i not in dims]
            dims = dims if isinstance(self.spec, Named) else len(dims)
        # fields if available
        names = [self._names[i] for i in sel]
        names = names if isinstance(self, Named) else len(names)
        res = grouping(clsname, dims, names)
        return self.aux_keyed(res).tree_unflatten(self.aux_data, tuple(
                self[i][idx[1:]] if len(idx) > 1 else self[i]
                for i in sel) + self.outslice(idx))

    @property
    def at(self):
        return GroupAt(self)

    @property
    def indirect(self):
        return GroupIndirect(self)

    def setter(self, idx, value):
        idx, sel, fields = self.subgroup(idx)
        # broadcast unwrapped values if needed
        sel, value = ((sel,), (value,)) if isinstance(sel, int) else (
                sel, value * len(sel) if len(value) == 1 else value)
        return self.tree_unflatten(
                self.aux_data, tuple(
                    self[i].at[idx[1:]].set(value[sel.index(i)])
                    if i in sel else self[i]
                    for i in range(len(self))) + self.outslice(idx, value))

    @property
    def shape(self):
        return (len(self),) + self[0].shape

    @property
    def ndim(self):
        return len(self.shape)

    def ref(self, key):
        if isinstance(key, (list, tuple)):
            assert all(isinstance(i, (str, int)) for i in key)
            return tuple(map(self.ref, key))
        return self._fields.index(key) if isinstance(key, str) else key

    @classmethod
    def where(cls, cond, x, y):
        assert x.aux_data == y.aux_data
        (xc, xa), (yc, _) = x.tree_flatten(), y.tree_flatten()
        return cls.tree_unflatten(xa, tuple(
                jnp.where(cond, i, j) for i, j in zip(xc, yc)))

    def vmap(self, *a, **kw):
        return GroupMap(self, False, *a, **kw)

    def remap(self, *a, **kw):
        return GroupMap(self, True, *a, **kw)

def group_alias(**kw):
    class GroupAliased:
        def aux_keyed(self, keying):
            return type(keying.__name__, (
                    GroupAliased, super().aux_keyed(keying)), {})

        def ref(self, key):
            return super().ref(
                    kw.get(key, key) if isinstance(key, str) else key)

        def __getattr__(self, key):
            if key in kw:
                return getattr(self, kw[key])

    return GroupAliased

def dim_alias(**kw):
    class Alias(Named):
        def __init__(self, ln, **kw):
            self.ln, self.kw = ln, kw

        def __getattr__(self, key):
            return getattr(self.ln, self.kw.get(key, key))

        def index(self, key):
            return self.ln.index(self.kw.get(key, key))

    class DimAliased:
        def aux_keyed(self, keying):
            return type(keying.__name__, (
                    DimAliased, super().aux_keyed(keying)), {})

        @property
        def spec(self):
            return Alias(super().spec, **kw)

    return DimAliased

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

    @property
    def index(self):
        return self._names.index

def nameable(clsname, names=None):
    if names is None or isinstance(names, int):
        class GroupSize(tuple):
            def __new__(cls, *a):
                assert names is None or len(a) == names
                return super().__new__(cls, a)

            @property
            def _names(self):
                return range(len(self))

            @property
            def index(self):
                return self._names.index
    else:
        class GroupSize(Named, namedtuple(clsname, names)):
            pass
    return type(clsname, (GroupSize,), {})

def grouping(clsname, dims=None, names=None, defaults=None):
    assert names is None or defaults is None or (
            (names if isinstance(names, int) else len(names)) == len(defaults))
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

        @property
        def order(self):
            return order

        def aux_keyed(self, keying):
            return type(keying.__name__, (
                    GroupAux, super().aux_keyed(keying)), {})

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
        def __new__(cls, *a, **kw):
            return super().__new__(cls, *a, **kw).deaux()

        def tree_flatten(self):
            return swap(*super().tree_flatten())

        @classmethod
        def tree_unflatten(self, aux_data, children):
            return super().tree_unflatten(*swap(aux_data, children)).deaux()

        def deaux(self):
            self.aux_keys = self.aux_keys[:len(self.aux_keys) - count]
            return self

        def outslice(self, idx, value=None):
            other = super().outslice(idx, value)
            return other + tuple(getattr(self, i) for i in super().order)

        def aux_keyed(self, keying):
            return type(keying.__name__, (
                    OutGroup, super().aux_keyed(keying)), {})
    return OutGroup

def marginalized(*axes, **defaults):
    assert defaults
    class Margin(outgroup(**defaults)):
        def __new__(cls, *a, **kw):
            obj = super().__new__(cls, *a, **kw)
            shape = [getattr(obj.spec, i) for i in axes if hasattr(obj.spec, i)]
            for k in defaults.keys():
                setattr(obj, k, jnp.full(shape, getattr(obj, k)))
            return obj

        @property
        def axes(self):
            return [i for i in axes if hasattr(self.spec, i)]

        @property
        def used(self):
            return tuple(1 + self.spec.index(i) for i in self.axes)

        @property
        def lo(self):
            return min(self.used) if self.used else None

        def outslice(self, idx, value=None):
            seen = super().outslice(idx, value)
            other, owned = seen[:-len(defaults)], seen[-len(defaults):]
            lo = self.lo
            if isinstance(idx, tuple) and lo is not None and len(idx) > lo:
                slices = tuple(idx[i] for i in self.used if i < len(idx))
                if value is None:
                    owned = tuple(i[slices] for i in owned)
                else:
                    owned = tuple(
                            i.at[slices].set(getattr(value, k))
                            if hasattr(value, k) else i
                            for k, i in zip(super().order, owned))
            return other + owned

        def aux_keyed(self, keying):
            return type(keying.__name__, (
                    Margin, super().aux_keyed(keying)), {})
    return Margin

def interface(dims=None, names=None, defaults=None):
    class GroupInterface:
        def __init__(self, *a, **kw):
            super().__init__()
            if isinstance(dims, int):
                assert dims <= len(self.spec)
            elif dims is not None:
                assert all(hasattr(self.spec, i) for i in dims)
            if isinstance(names, int):
                assert names <= len(self)
            elif names is not None:
                assert all(self.ref(i) or 1 for i in names)
            if defaults is not None and names is not None:
                assert all(
                        getattr(self, i).dtype == j.dtype
                        for i, j in zip(names, defaults))
                if not any(isinstance(i, jax.core.Tracer) for i in self):
                    assert all(
                            jnp.all(getattr(self, i) == j)
                            for i, j in zip(names, defaults)
                            if isinstance(j, jax.Array) and
                            not jnp.isnan(j) and jnp.isfinite(j))
    return GroupInterface

