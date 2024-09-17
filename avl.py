import jax, math
import jax.numpy as jnp
from group import grouping, outgroup
from functools import partial

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

@jax.tree_util.register_pytree_node_class
class AVLs(outgroup(root=jnp.int32(-1)), grouping(
        "AVL", ("size",),# ("trees", "size"),
        ("key", "secondary", "left", "right", "height"), (
            jnp.float32(jnp.nan), jnp.int32(-1), jnp.int32(-1), jnp.int32(-1),
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
        return jnp.where(
                self.key[x] != self.key[y],
                jnp.where(self.key[x] > self.key[y], 1, -1),
                jnp.where(
                    self.secondary[x] == self.secondary[y], 0,
                    jnp.where(self.secondary[x] > self.secondary[y], 1, -1)))

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
            node, x, out = args
            sign = self.cmp(x, node)
            out = out.at[:, out.height].set((node, sign))
            out.height += 1
            node = jnp.where(sign == 0, -1, jnp.where(
                    sign == 1, self.right[node], self.left[node]))
            return node, x, out
        node, _, out = jax.lax.while_loop(
                lambda a: a[0] != -1, body, (node, x, out))
        return out[:, ::-1]

    @staticmethod
    def set_left(t, ins, root):
        return t.at['left', root].set(ins)

    @staticmethod
    def set_right(t, ins, root):
        return t.at['right', root].set(ins)

    @staticmethod
    def set_root(t, ins, *a):
        t.root = ins
        return t

    def pre_balance(self, cond, root, balance):
        return jax.lax.cond(
                cond,
                lambda t, root, balance: jax.lax.cond(
                    balance == 1,
                    lambda t, root: t.set_left(
                        *t.left_rotate(t.left[root]), root),
                    lambda t, root: t.set_right(
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
            t = jax.lax.cond(sign == 1, t.set_right, t.set_left, t, y, root)
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
            t = t.at['left', child].set(t['left', x])
            t = t.at['right', child].set(t['right', x])
            path.height -= grandchild == -1
            return path, t, child
        def successor(args):
            node, path, t = args
            path.height += 1
            path = path.at[:, path.start].set((node, -1))
            return t.left[node], path, t
        path = self.path(x)
        # checkify.check(path[path.start] == 0, "node not found")
        parent = path.start + 1
        path, t, x = jax.lax.cond(
                (self.left[x] == -1) | (self.right[x] == -1),
                branchless, split, path, self, x)
        t = jax.lax.cond(
                parent == path.shape[1],
                t.set_root, lambda t, x, path: jax.lax.cond(
                    path.sign[parent] == 1, t.set_right, t.set_left,
                    t, x, path.path[parent]), t, x, path)
        def body(i, args):
            path, t, x = args
            root, sign = path[:, i]
            t = jax.lax.cond(
                    x == -1, lambda t, *a: t,
                    lambda t, x, root: jax.lax.cond(
                        sign == 1, t.set_right, t.set_left, t, x, root),
                    t, x, root)
            t = t.at['height', root].set(t.measured[root])

            balance = t.balance[root]
            balance = jnp.where(balance > 1, 1, jnp.where(balance < -1, -1, 0))
            side = jnp.where(balance == 0, -1,
                jnp.where(balance == 1, t.left[root], t.right[root]))
            double = jnp.abs(balance - t.balance[side]) == 2

            t = t.pre_balance(double, root, balance)
            return (path,) + t.re_balance(root, balance)

        _, t, x = jax.lax.fori_loop(
                path.start, path.shape[1], body, (path, t, -1))
        t.root = x
        return t

    def walk(self, value='key', root=None, transform=None):
        root = self.root if root is None else root
        transform = (lambda x: x) if transform is None else transform
        if root == -1:
            return ""
        return f"{transform(self[value, root])} " + (
                self.walk(value, self.left[root], transform) +
                self.walk(value, self.right[root], transform))

