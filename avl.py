import jax
import jax.numpy as jnp
from group import Group, grouping, outgroup

class LazyGetter:
    def __init__(self, f):
        self.f = f

    def __getitem__(self, i):
        return self.f(i)

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
        phi = (1 + jnp.sqrt(5)) / 2
        log_phi = jnp.log(phi)
        b = jnp.log(5) / log_phi / 2 - 2
        return jnp.int32(jnp.log(self.spec.size + 2) / log_phi + b)

    @jax.jit
    def insert(self, x):
        def left(t, ins, root):
            return t.at['left', root].set(ins)
        def right(t, ins, root):
            return t.at['right', root].set(ins)

        def base(t, ins, root, depth):
            return t.at['height', ins].set(1), ins
        @partial(jax.jit, static_argnames=("depth",))
        def recurse(t, ins, root, depth):
            sign = t.cmp(ins, root)
            return jax.lax.cond(
                    sign == 0,
                    lambda t, root, ins, sign, depth: (t, root),
                    valid,
                    t, ins, root, sign, depth - 1)
        @partial(jax.jit, static_argnames=("depth",))
        def valid(t, ins, root, sign, depth):
            t = jax.lax.cond(
                    sign == 1,
                    lambda t, ins, root: right(
                        *_insert(t, ins, t.right[root], depth), root),
                    lambda t, ins, root: left(
                        *_insert(t, ins, t.left[root], depth), root),
                    t, ins, root)


            t = t.at['height', root].set(1 + jnp.maximum(
                    t.height[t.left[root]], t.height[t.right[root]]))
            balance = t.balance[root]
            balance = jnp.where(balance > 1, 1, jnp.where(balance < -1, -1, 0))

            t = jax.lax.cond(
                    balance == sign,
                    lambda t, root, sign: jax.lax.cond(
                        sign == 1,
                        lambda t, root: left(
                            *t.left_rotate(t.left[root]), root),
                        lambda t, root: right(
                            *t.right_rotate(t.right[root]), root),
                        t, root),
                    lambda t, root, sign: t,
                    t, root, sign)

            return jax.lax.cond(
                    balance == 0,
                    lambda t, root, balance: (t, root),
                    lambda t, root, balance: jax.lax.cond(
                        balance == 1,
                        lambda t, root: t.right_rotate(root),
                        lambda t, root: t.left_rotate(root),
                        t, root),
                    t, root, balance)

        @partial(jax.jit, static_argnames=("depth",))
        def _insert(t, ins, root, depth):
            if depth == 0:
                return t, root
            return jax.lax.cond(root == -1, base, recurse, t, ins, root)
        def init(t, ins):
            t, root = _insert(t, ins, t.root, t.bound)
            t = t.at['root'].set(root)
            return t

        first = self.root == -1
        self = jnp.where(first, self.at['root'].set(x), self)
        return jax.lax.cond(first, lambda t, x: t, init, self, x)

