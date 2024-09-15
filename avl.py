import jax
import jax.numpy as jnp
from group import Group, grouping, groupaux

class AVLs(groupaux("root"), grouping("AVL", ("trees", "size"), (
        "key", "left", "right", "height"))):
    pass
