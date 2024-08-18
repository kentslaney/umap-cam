from scipy.optimize import minimize

fun = lambda x: print(x) or (x[0] - 1)**2 + (x[1] - 2.5)**2


cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
        {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
        {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})

bnds = ((1.5, None), (0, None))

res = minimize(fun, (2, 0), method='L-BFGS-B', bounds=bnds)
print(res)

import jax
import jax.numpy as jnp
from jaxopt import ScipyBoundedMinimize

# n_features = (1, 2)
# def fun(w, X, y):
#     return jnp.sum((w @ X - y) ** 2)
#
# rng0, rng1 = jax.random.split(jax.random.key(0))
# X = jax.random.normal(rng0, (2, 3))
# y = jax.random.normal(rng1, (3,))

w_init = jnp.asarray((2., 0))
lbfgsb = ScipyBoundedMinimize(fun=fun, method="l-bfgs-b")
lower_bounds = jnp.asarray((1.5, 0))
upper_bounds = jnp.ones_like(w_init) * jnp.inf
bounds = (lower_bounds, upper_bounds)
lbfgsb_sol = lbfgsb.run(w_init, bounds=bounds).params
print(lbfgsb_sol)

# solve for optimal update step? optimal position within distance?
# under relaxation
# CFD pressure type model to encourage using the entire color space?

