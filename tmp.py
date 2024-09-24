import jax, os
import jax.numpy as jnp
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parents[0]))
import avl, group
from importlib import reload
group = reload(group)
avl = reload(avl)

from avl import AVLs, SingularAVL

# https://www.geeksforgeeks.org/introduction-to-avl-tree/

# t = AVLs(2, 32)
# keys = jnp.concatenate(
#         (jnp.arange(1, 6) * 10, jnp.asarray((25,))), dtype=jnp.float32)
# t = t.at['key', 0, :keys.size].set(keys)
# for i in range(keys.size):
#     t = t.at[:, 0].insert(i)
#
# keys = jnp.asarray([9, 5, 10, 0, 6, 11, -1, 1, 2])
# t = t.at['key', 1, :keys.size].set(keys)
# for i in range(keys.size):
#     t = t.at[:, 1].insert(i)

# print(t.indirect[:, 0].search(25, -1))

# t = AVLs(2, 32)
# t = t.at['secondary', 0, :15].set(1)
# t = t.at['secondary', 1, :26].set(1)
# t = t.at['secondary', 2:].set(1)

t = SingularAVL(32)

rng = jax.random.key(0)
# idx = None
# for i in range(t.shape[1]):
subkey, rng = jax.random.split(rng)
t = t.at['key'].set(jax.random.normal(subkey, (32,)))
t = t.at['secondary'].set(1)
t = t.batched()
print(t.walk())

for i in range(32):
    subkey, rng = jax.random.split(rng)
    x = jax.random.normal(subkey, ())
    t = t.replace(x, i + 2)
print(t.walk())

# for i in range(t.spec.trees):
#     for j in range(t.spec.size):
#         if t.secondary[i, j] != -1:
#             t = t.at[:, i].insert(j)
# print(t)

