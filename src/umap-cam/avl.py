import jax, math
import jax.numpy as jnp
from group import grouping, outgroup, marginalized, interface
from functools import partial

class Chars:
    BAR = "\u2502"
    DASH = "\u2500"
    BAR_DASH = "\u251C"
    DASH_BAR = "\u252C"
    ELL = "\u2514"
    MT = " "

    side = lambda childless: Chars.DASH if childless else Chars.DASH_BAR
    top = lambda last: Chars.ELL if last else Chars.BAR_DASH
    space = lambda last: Chars.MT if last else Chars.BAR
    seq = (top, side, space)

class LazyGetter:
    def __init__(self, f):
        self.f = f

    def __getitem__(self, i):
        return self.f(i)

@jax.tree_util.register_pytree_node_class
class SearchPath(outgroup(height=0), grouping(
        "SearchPath", ("size",), ("path", "sign"),
        (jnp.int32(-1), jnp.int32(0)))):
    @property
    def start(self):
        return self.shape[1] - self.height

class AVLsInterface(marginalized("trees", root=jnp.int32(-1)), interface(
        ("size",), ("key", "secondary", "left", "right", "height"), (
            jnp.float32, jnp.int32, jnp.int32(-1), jnp.int32(-1),
            jnp.int32(1)))):
    def right_rotate(self, y):
        x = self.left[y]
        z = self.right[x]

        self = self.at['right', x].set(y)
        self = self.at['left', y].set(z)
        self = self.at['height', y].set(self.measured[y])
        self = self.at['height', x].set(self.measured[x])
        return self, x

    def left_rotate(self, x):
        y = self.right[x]
        z = self.left[y]

        self = self.at['left', y].set(x)
        self = self.at['right', x].set(z)
        self = self.at['height', x].set(self.measured[x])
        self = self.at['height', y].set(self.measured[y])
        return self, y

    @property
    def depth(self):
        return LazyGetter(lambda x: jnp.where(x == -1, 0, self.height[x]))

    @property
    def balance(self):
        return LazyGetter(lambda x: jnp.where(x == -1, 0, (
                self.depth[self.left[x]] - self.depth[self.right[x]])))

    @property
    def measured(self):
        return LazyGetter(lambda x: 1 + jnp.maximum(
                self.depth[self.left[x]], self.depth[self.right[x]]))

    def cmp(self, x, y):
        return self.sign(self.key[x], self.secondary[x], y)

    def sign(self, primary, secondary, y):
        return jnp.where(
                jnp.isclose(primary, self.key[y]),
                jnp.where(
                    secondary == self.secondary[y], 0,
                    jnp.where(secondary > self.secondary[y], 1, -1)),
                jnp.where(primary > self.key[y], 1, -1))

    @property
    def bound(self):
        phi = (1 + math.sqrt(5)) / 2
        log_phi = math.log(phi)
        b = math.log(5) / log_phi / 2 - 2
        return math.trunc(math.log(self.spec.size + 2) / log_phi + b)

    @jax.jit
    def path(self, x):
        node = self.root
        out = SearchPath(self.bound)
        def body(args):
            node, out = args
            sign = self.cmp(x, node)
            out = out.at[:, out.height].set((node, sign))
            out.height += 1
            node = jnp.where(sign == 0, -1, jnp.where(
                    sign == 1, self.right[node], self.left[node]))
            return node, out
        node, out = jax.lax.while_loop(
                lambda a: a[0] != -1, body, (node, out))
        return out[:, ::-1]

    @jax.jit
    def search(self, primary, secondary):
        node = self.root
        def body(args):
            node, _, _ = args
            sign = self.sign(primary, secondary, node)
            update = jnp.where(sign == 0, -1, jnp.where(
                    sign == 1, self.right[node], self.left[node]))
            return update, node, sign
        _, node, sign = jax.lax.while_loop(
                lambda a: a[0] != -1, body, (node, node, 0))
        return (node, sign)

    def contains(self, primary, secondary):
        node, sign = self.search(primary, secondary)
        return (node != -1) & (sign == 0)

    def set_left(self, ins, root):
        return self.at['left', root].set(ins)

    def set_right(self, ins, root):
        return self.at['right', root].set(ins)

    def set_root(self, ins, *a):
        self.root = ins
        return self

    def pre_balance(self, cond, root, balance):
        return jax.lax.cond(
                cond,
                lambda t, root, balance: jax.lax.cond(
                    balance == 1,
                    lambda t, root: t.__class__.set_left(
                        *t.left_rotate(t.left[root]), root),
                    lambda t, root: t.__class__.set_right(
                        *t.right_rotate(t.right[root]), root),
                    t, root),
                lambda t, *a: t,
                self, root, balance)

    def re_balance(self, root, balance):
        return jax.lax.cond(
                balance == 0,
                lambda t, root, balance: (t, root),
                lambda t, root, balance: jax.lax.cond(
                    balance == 1,
                    lambda t, root: t.right_rotate(root),
                    lambda t, root: t.left_rotate(root),
                    t, root),
                self, root, balance)

    @jax.jit
    def insert(self, x):
        def body(i, args):
            path, x, t, y = args
            root, sign = path[:, i]
            t = jax.lax.cond(sign == 1, t.set_right, t.set_left, y, root)
            t = t.at['height', root].set(t.measured[root])

            balance = t.balance[root]
            balance = jnp.where(balance > 1, 1, jnp.where(balance < -1, -1, 0))
            side = jnp.where(balance == 0, 1, t.cmp(
                x, jnp.where(balance == 1, t.left[root], t.right[root])))

            t = t.pre_balance(balance == side, root, balance)
            return (path, y) + t.re_balance(root, balance)

        path = self.path(x)
        _, _, t, x = jax.lax.fori_loop(
                path.start, path.shape[1], body, (path, x, self, x))
        t.root = x
        return t

    @jax.jit
    def remove(self, x):
        def branchless(path, t, x):
            child = jnp.where(self.left[x] == -1, self.right[x], self.left[x])
            path = path.at['path', path.start].set(child)
            path.height -= child == -1
            return path, t, child
        def split(path, t, x):
            idx, node = path.start, t.right[x]
            path = jax.lax.while_loop(
                    lambda a: a[0] != -1, successor, (node, path, t))[1]
            child = path.path[path.start]
            grandchild = t.right[child]
            path = path.at[:, idx].set((child, 1))
            path = path.at[:, path.start].set((grandchild, 0))
            t = t.at['left', path.path[path.start + 1]].set(grandchild)
            dereferenced = jnp.where(t.right[x] == child, -1, t.right[x])
            t = t.at[('left', 'right'), child].set((t.left[x], dereferenced))
            path.height -= grandchild == -1
            return path, t, child
        def successor(args):
            node, path, t = args
            path.height += 1
            path = path.at[:, path.start].set((node, -1))
            return t.left[node], path, t
        path, _x = self.path(x), x
        # checkify.check(path[path.start] == 0, "node not found")
        parent = path.start + 1
        path, t, x = jax.lax.cond(
                (self.left[x] == -1) | (self.right[x] == -1),
                branchless, split, path, self, x)
        t = jax.lax.cond(
                parent == path.shape[1],
                t.__class__.set_root, lambda t, x, path: jax.lax.cond(
                    path.sign[parent] == 1, t.set_right, t.set_left,
                    x, path.path[parent]), t, x, path)
        def body(i, args):
            path, t, x = args
            root, sign = path[:, i]
            t = jax.lax.cond(
                    x == -1, lambda t, *a: t,
                    lambda t, x, root: jax.lax.cond(
                        sign == 1, t.set_right, t.set_left, x, root),
                    t, x, root)
            t = t.at['height', root].set(t.measured[root])

            balance = t.balance[root]
            balance = jnp.where(balance > 1, 1, jnp.where(balance < -1, -1, 0))
            side = jnp.where(balance == 0, 0,
                t.balance[jnp.where(balance == 1, t.left[root], t.right[root])])
            double = jnp.abs(balance - side) == 2

            t = t.pre_balance(double, root, balance)
            return (path,) + t.re_balance(root, balance)

        t = t.at[('left', 'right', 'height'), _x].set((-1, -1, 1))
        _, t, x = jax.lax.fori_loop(
                path.start, path.shape[1], body, (path, t, -1))
        t.root = x
        return t

    def arena(self):
        jax.debug.print(
                "{}\n{}\n{}", jnp.arange(self.shape[1]), self.left, self.right)

    tsv = lambda t, x: "\t".join(map(str, (
            x, f"{t.key[x]:.5f}", t.secondary[x], t.height[x])))
    def tsv_table(self, init):
        names = ["index", "primary", "secondary", "height"]
        res = [i.split("\t") for i in init.split("\n")]
        res = [i + [""] * max(len(names) - len(i), 0) for i in res]
        res = [names] + res
        size = list(map(max, zip(*((len(j) for j in i) for i in res))))
        return "\n".join((" " * 4).join(
                getattr(i[j], "ljust" if j == 0 else "rjust")(size[j])
                for j in range(len(i))) for i in res)

    def walk(self, value=None, hide=False, root=None, f=None, g=None, alt=None):
        if hasattr(self.spec, "trees"):
            return "\n".join(
                self.indirect[:, i].walk(self.indirect[:, i].tsv, hide, root)
                for i in range(self.spec.trees))
        start = value is None and root is None
        if root is None:
            assert self.acyclic()
        root = self.root if root is None else root
        f = (lambda x: "\n" + x) if f is None else f
        mt = lambda n: n == -1 or self.left[n] == -1 and self.right[n] == -1
        g = (lambda x: Chars.side(mt(self.root)) + x) if g is None else g
        if root == -1:
            return "" if hide or alt == -1 else g(" ")

        _l, _r = l, r = self.left[root], self.right[root]
        l, r = (r == -1, mt(l), r == -1), (1, mt(r), 1)
        l, r = (l, r) if hide else ((0, l[1], 0), (1, r[1], 1))
        l, r = (tuple(f(i) for f, i in zip(Chars.seq, j)) for j in (l, r))
        fl, gl = lambda x: f(l[2] + x), lambda x: l[0] + l[1] + x
        fr, gr = lambda x: f(r[2] + x), lambda x: r[0] + r[1] + x
        h = lambda x: f(x) if x else ""

        node = self.tsv(root) if value is None else \
                self[value, root] if isinstance(value, str) else value(root)
        res = f"{g(str(node))}" + (
                h(self.walk(value, hide, self.left[root], fl, gl, _r)) +
                h(self.walk(value, hide, self.right[root], fr, gr, _l)))
        return self.tsv_table(res) if start else res

    def __repr__(self):
        edgeitems = jnp.get_printoptions()['edgeitems']
        if hasattr(self.spec, "trees") and self.spec.trees > 2 * edgeitems:
            top = self.indirect[:, :edgeitems].walk()
            mid = f"\n...\n({self.spec.trees - 2 * edgeitems} more)\n...\n"
            bottom = self.indirect[:, -edgeitems:].walk()
            return self.tsv_table(top + mid + bottom)
        return self.tsv_table(self.walk())

    def acyclic(self):
        q = jnp.full(self.spec.size, -1).at[0].set(self.root)
        def body(node):
            return self.left[node], self.right[node]
        def loop(i, args):
            q, res = args
            l, r = jax.vmap(lambda n: jax.lax.cond(
                n != -1, body, lambda n: (n, -1), n))(q)
            q = jnp.concatenate((l, r))
            s, q = jax.lax.sort_key_val(q == -1, q)
            return q[:self.spec.size], res & s[self.spec.size]
        q, res = jax.lax.fori_loop(0, self.spec.size, loop, (q, True))
        res &= jnp.all(q == -1)
        return res

    @jax.jit
    def batched(self):
        filled = self.secondary != -1
        pop = jnp.sum(filled)
        idx = jnp.arange(self.spec.size)
        order = (1 - filled, self.key, self.secondary, idx)
        order = jax.lax.sort(order, num_keys=2)[3]
        bound = jnp.int32(jnp.ceil(jnp.log2(pop + 1)))

        pos, size = idx + 1, jnp.full(self.spec.size, pop)
        left, right = jnp.zeros((2, self.spec.size), dtype=jnp.int32)
        height = jnp.ones(self.spec.size, dtype=jnp.int32)

        def body(i, args):
            pos, left, right, height, size = args
            size -= (height == 1) * 1
            large = -(-size // 2)
            size = large - (size % 2) * (pos > large)
            pos = pos % (large + 1)
            left += (left == 0) * (pos == 0) * (((large - 1) // 2) + 1)
            right += (right == 0) * (pos == 0) * (1 - ((1 - size) // 2))
            height += (height == 1) * (pos == 0) * jnp.int32(
                    jnp.ceil(jnp.log2(large + 1)))
            return pos, left, right, height, size
        pos, left, right, height, size = jax.lax.fori_loop(
                1, bound, body, (pos, left, right, height, size))
        right -= size == 0
        filled *= -1
        height = height & filled | (~filled & 1)

        left = (idx - left) - (idx + 1) * (left == 0)
        right = (idx + right) - (idx + 1) * (right == 0)
        left, right = left | ~filled, right | ~filled
        left, right = (jnp.where(i == -1, -1, order[i]) for i in (left, right))
        self = self.at[('left', 'right', 'height'), order].set(
                (left, right, height))
        self.root = order[-((1 - pop) // 2)]
        return self

    def row(self, i=None, min_space=2, secondaries=False, sep=" "):
        assert "trees" not in self.spec or i is not None
        if i is not None:
            self = self[:, i]
        res = "" if i is None else f"row: {i}    "
        min_size = 2 # for null "-1" key
        size = max(len(str(self.spec.size)), min_size)
        if secondaries:
            size = max(size, *(len(str(int(i))) for i in self.secondary))
        fmt = "{:0" + str(size) + "}"
        res += "root: " + fmt.format(int(self.root)) + "\n"
        fmt = " " * (min_space - size) + fmt
        info = (jnp.arange(self.spec.size), self.left, self.right)
        res += "\n".join(sep.join(fmt.format(int(i)) for i in j) for j in info)
        if secondaries:
            res += "\n" + sep.join(
                    str(int(i)).rjust(max(min_space, size))
                    for i in self.secondary)
        return res

class MaxAVL(
        marginalized("trees", max=jnp.int32(-1), full=jnp.int32(0)),
        AVLsInterface):
    @jax.jit
    def insert(self, x):
        self = AVLsInterface.insert(self, x)
        self.max = jnp.where(self.max == -1, x, jnp.where(
                self.cmp(x, self.max) == 1, x, self.max))
        return self

    @jax.jit
    def remove(self, x):
        self = AVLsInterface.remove(self, x)
        self.max = jax.lax.cond(
                x == self.max,
                lambda t: jax.lax.while_loop(
                    lambda a: a[0] != -1,
                    lambda a: (a[2].right[a[0]], a[0], a[2]),
                    (t.root, t.root, t))[1],
                lambda t: t.max,
                self)
        return self

    def batched(self):
        self = AVLsInterface.batched(self)
        self.max = jnp.nanargmax(self.key)
        self.full = self.spec.size - jnp.sum(self.secondary == -1)
        return self

    def __repr__(self):
        edgeitems = jnp.get_printoptions()['edgeitems']
        if self.max.size > 2 * edgeitems:
            end = "[" + " ".join(map(str, self.max[:edgeitems])) + " ... (" + \
                    str(self.max.size - 2 * edgeitems) + " more) ... " + \
                    " ".join(map(str, self.max[-edgeitems:])) + "]"
        else:
            end = str(self.max)
        return AVLsInterface.__repr__(self) + "\nwith max = " + end

    @jax.jit
    def replace(self, primary, secondary):
        pos = self.max
        self = self.remove(pos)
        self = self.at[("key", "secondary"), pos].set((primary, secondary))
        return self.insert(pos)

    def includable(self, primary, secondary):
        return (
                ((self.full < self.spec.size) | (
                    self.sign(primary, secondary, self.max) == -1)) &
                (secondary != -1) & ~self.contains(primary, secondary))

    @partial(jax.jit, static_argnames=("checked",))
    def push(self, primary, secondary, checked=True):
        def add(t, primary, secondary):
            t = t.at[("key", "secondary"), t.full].set((primary, secondary))
            t = t.insert(t.full)
            t.full += 1
            return t
        def body(t):
            return jax.lax.cond(
                    t.full < t.spec.size, add, t.__class__.replace,
                    t, primary, secondary)
        return jax.lax.cond(
                self.includable(primary, secondary),
                body, lambda t: t, self) if checked else body(self)

    @property
    def key_max(self):
        if hasattr(self.spec, "trees"):
            return jnp.stack([self['key', *i] for i in jnp.stack(
                (jnp.arange(self.spec.points), self.max), -1)])
        else:
            return self.key[self.max]

@jax.tree_util.register_pytree_node_class
class Predecessors:
    def __init__(self, parent, node=-1, path=None):
        self.parent = parent
        node = jnp.where(node == -1, parent.max, node)
        self.path = parent.path(node) if path is None else path

    @jax.jit
    def next(self):
        def hasnt_left(level):
            return (
                    (level < self.path.spec.size) &
                    (self.path.sign[level] != 1) &
                    ~((self.path.sign[level] == 0) & (
                        self.parent.left[self.path.path[level]] != -1)))
        level = jax.lax.while_loop(
                hasnt_left, lambda level: level + 1, self.path.start)
        # if level is 0 stop; else set child to 1 or else 0
        def step(step=1):
            assert step in (-1, 1)
            dir = 'right' if step == 1 else 'left'
            def f(args):
                level, path = args
                node = getattr(self.parent, dir)[path.path[level]]
                return jax.lax.cond(
                        node == -1,
                        lambda: (level, path.at['sign', level].set(0)),
                        lambda: (
                            level - 1, path.at[:, level - 1].set((node, 1))))
            return f
        mov = lambda a: (a[0] >= 0) & (a[1].sign[a[0]] != 0)
        def down():
            nonlocal level
            path = self.path.at['sign', level].add(-1)
            level, path = jax.lax.cond(
                    mov((level, path)), step(-1), lambda a: (level, path),
                    (level, path))
            level, path = jax.lax.while_loop(mov, step(), (level, path))
            path.height = path.spec.size - level
            return __class__(self.parent, path=path)
        return jax.lax.cond(level >= self.path.spec.size, lambda: self, down)

    @property
    def value(self):
        return self.path.path[self.path.start]

    def tree_flatten(self):
        return (self.parent, self.path), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], path=children[1])

@jax.tree_util.register_pytree_node_class
class SingularAVL(MaxAVL, grouping(
        "AVL", ("size",), ("key", "secondary", "left", "right", "height"), (
            jnp.float32(jnp.nan), jnp.int32(-1), jnp.int32(-1), jnp.int32(-1),
            jnp.int32(1)))):
    pass

@jax.tree_util.register_pytree_node_class
class AVLs(MaxAVL, grouping(
        "AVLs", ("trees", "size"),
        ("key", "secondary", "left", "right", "height"), (
            jnp.float32(jnp.nan), jnp.int32(-1), jnp.int32(-1), jnp.int32(-1),
            jnp.int32(1)))):
    pass

