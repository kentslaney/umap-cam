import jax, os
import jax.numpy as jnp
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parents[0]))
import avl, group
from importlib import reload
group = reload(group)
avl = reload(avl)

from avl import AVLs

t = AVLs(16)
keys = jnp.concatenate(
        (jnp.arange(1, 6) * 10, jnp.asarray((25,))), dtype=jnp.float32)
t = t.at['key', :6].set(keys)
for i in range(6):
    t = t.insert(i)
