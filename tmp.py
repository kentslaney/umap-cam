import jax, os
import jax.numpy as jnp
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parents[0]))
import avl, group
from importlib import reload
group = reload(group)
avl = reload(avl)

from avl import AVLs

# https://www.geeksforgeeks.org/introduction-to-avl-tree/

t = AVLs(2, 32)
keys = jnp.concatenate(
        (jnp.arange(1, 6) * 10, jnp.asarray((25,))), dtype=jnp.float32)
# t = t.at['key', 0, :keys.size].set(keys)
# for i in range(keys.size):
#     t = t.at[:, 0].insert(i)
#
# print(t.indirect[:, 0].walk(transform=int))
# print(t.indirect[:, 0].search(25, -1))

t = t.at['secondary', 0, :15].set(1).at[:, 0].batched()
t = t.at['secondary', 1, :26].set(1).at[:, 1].batched()
print(t)
print(t[:, 0, t.root[0]])
print(t[:, 1, t.root[1]])

# keys = jnp.asarray([9, 5, 10, 0, 6, 11, -1, 1, 2])
# t = t.at['key', 1, :keys.size].set(keys)
# for i in range(keys.size):
#     t = t.at[:, 1].insert(i)
#
# print(t.indirect[:, 1].walk(transform=int))
# print(t.max[1])
# t = t.at[:, 1].remove(5)
# print(t.indirect[:, 1].walk(transform=int))
# print(t.max[1])
#
