import jax, os
import jax.numpy as jnp
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parents[0]))
import avl, group
from importlib import reload
group = reload(group)
avl = reload(avl)

from avl import AVLs

# s = AVLs(16)
# keys = jnp.concatenate(
#         (jnp.arange(1, 6) * 10, jnp.asarray((25,))), dtype=jnp.float32)
# s = s.at['key', :keys.size].set(keys)
# for i in range(keys.size):
#     s = s.insert(i)
# print(s)

t = AVLs(16)
keys = jnp.asarray([9, 5, 10, 0, 6, 11, -1, 1, 2])
t = t.at['key', :keys.size].set(keys)
for i in range(keys.size):
    t = t.insert(i)

print(t.walk(transform=int))
t = t.remove(2)
print(t.walk(transform=int))

