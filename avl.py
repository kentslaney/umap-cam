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
    pass

@jax.tree_util.register_pytree_node_class
class AVLs(outgroup(root=jnp.int32(-1)), grouping(
# class AVLs(groupaux(root=-1), grouping(
        "AVL", ("size",),# ("trees", "size"),
        ("key", "secondary", "left", "right", "height"), (
            jnp.float32(jnp.nan), jnp.int32(-1), jnp.int32(-1), jnp.int32(-1),
            jnp.int32(0)))):
    def right_rotate(self, y):
        x = self.left[y]
        z = self.right[x]

        self = self.at['right', x].set(y)
        self = self.at['left', y].set(z)
        self = self.at['height', y].set(1 + jnp.maximum(
            self.height[z], self.height[self.right[y]]))
        self = self.at['height', x].set(1 + jnp.maximum(
            self.height[self.left[x]], self.height[y]))
        return self, x

    def left_rotate(self, x):
        y = self.right[x]
        z = self.left[y]

        self = self.at['left', y].set(x)
        self = self.at['right', x].set(z)
        self = self.at['height', x].set(1 + jnp.maximum(
            self.height[self.left[x]], self.height[z]))
        self = self.at['height', y].set(1 + jnp.maximum(
            self.height[x], self.height[self.right[y]]))
        return self, y

    @property
    def balance(self):
        return LazyGetter(lambda x: (
                self.height[self.left[x]] - self.height[self.right[x]]))

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
        _, order = jax.lax.sort_key_val(out.path != -1, jnp.arange(self.bound))
        return out[:, order[::-1]]

    @jax.jit
    def insert(self, x):
        def left(t, ins, root):
            return t.at['left', root].set(ins)
        def right(t, ins, root):
            return t.at['right', root].set(ins)
        def body(i, args):
            path, x, t, y = args
            root, sign = path[:, i]
            t = jax.lax.cond(sign == 1, right, left, t, y, root)
            t = t.at['height', root].set(1 + jnp.maximum(
                    t.height[t.left[root]], t.height[t.right[root]]))

            balance = t.balance[root]
            balance = jnp.where(balance > 1, 1, jnp.where(balance < -1, -1, 0))
            side = jnp.where(balance == 0, 1, t.cmp(
                x, jnp.where(balance == 1, t.left[root], t.right[root])))

            t = jax.lax.cond(
                    balance == side,
                    lambda t, root, side: jax.lax.cond(
                        side == 1,
                        lambda t, root: left(
                            *t.left_rotate(t.left[root]), root),
                        lambda t, root: right(
                            *t.right_rotate(t.right[root]), root),
                        t, root),
                    lambda t, root, side: t,
                    t, root, side)

            return (path, y) + jax.lax.cond(
                    balance == 0,
                    lambda t, root, balance: (t, root),
                    lambda t, root, balance: jax.lax.cond(
                        balance == 1,
                        lambda t, root: t.right_rotate(root),
                        lambda t, root: t.left_rotate(root),
                        t, root),
                    t, root, balance)

        path = self.path(x)
        t = self.at['height', x].set(1)
        _, _, t, x = jax.lax.fori_loop(0, path.height, body, (path, x, t, x))
        t.root = x
        return t

