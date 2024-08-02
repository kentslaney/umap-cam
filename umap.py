from functools import partial
from nnd import npy_cache, grouping
from jax.experimental import sparse
from jax.scipy import optimize
import jax.numpy as jnp
import jax
import debug

class SmoothKNN(grouping("SmoothKNN", names=("sigma", "rho"))):
    pass

def itp(f, lo, hi, f_lo, f_hi, n_0=64, eps=1e-5):
    k_1, k_2 = 1 / 3 / (hi - lo), 2
    n_half = jnp.ceil(jnp.log2(hi - lo) / (2 * eps))
    n_max = n_half + n_0
    def loop(args):
        i, lo, hi, f_lo, f_hi = args
        mid, r = (lo + hi) / 2, eps * 2 ** (n_max - i) - (hi - lo) / 2
        delta = k_1 * (hi - lo) ** k_2
        interpolation = (hi * f_lo - lo * f_hi) / (f_lo - f_hi)
        sigma = jnp.sign(mid - interpolation)
        truncated = jnp.where(
                jnp.abs(mid - interpolation) >= delta,
                interpolation + delta * sigma, mid)
        projected = jnp.where(
                jnp.abs(mid - truncated) <= r,
                truncated, mid - sigma * r)
        x, y = projected, f(projected)
        return (i + 1, *jax.lax.cond(
                y > 0, lambda: (lo, x, f_lo, y), lambda: (x, hi, y, f_hi)))
    return jnp.mean(jnp.asarray(jax.lax.while_loop(
        lambda a: a[2] - a[1] > 2 * eps, loop, (0, lo, hi, f_lo, f_hi))[1:3]))

@jax.jit
def smooth_knn(distances, n_0=64, local_connectivity=1.0, bandwidth=1.0):
    idx, interpolation = jnp.int32(local_connectivity), local_connectivity % 1
    neighbors = jnp.sum(distances > 0, axis=1)
    duplicates = distances.shape[1] - neighbors
    rho = jnp.where(
            neighbors >= local_connectivity, jax.lax.cond(
                idx == 0, lambda: interpolation * distances[:, 0], lambda: (
                    distances[:, idx - 1] * (1 - interpolation) +
                    distances[:, idx] * interpolation)),
            jnp.maximum(jnp.max(distances, 1), 0))
    delta = jnp.maximum(0, distances - rho[..., None])
    target = jnp.log2(distances.shape[1]) * bandwidth
    opt = lambda x, d=delta: jnp.sum(jnp.exp(-d / x[..., None]), -1) - target
    bound = jnp.ones_like(rho)
    bound, initial = jax.vmap(lambda x, d: jax.lax.while_loop(
            lambda y: y[1] <= 0,
            lambda y: (y[0] * 2, opt(y[0] * 2, d=d)),
            (x, opt(x, d))))(bound, delta)
    solver = jax.vmap(lambda x, y, d, lo: itp(
            partial(opt, d=d), 1e-9, x, lo, y, n_0))
    return SmoothKNN(solver(bound, initial, delta, duplicates - target), rho)

def memberships(knn, smoothed=None):
    smoothed = smooth_knn(knn.distances) if smoothed is None else smoothed
    smoothed = smoothed[:, :, None]
    res = jnp.exp((knn.distances - smoothed.rho) / smoothed.sigma)
    return knn.at["distances"].set(res)

@jax.jit
def simplices(members):
    idx = jnp.arange(members.shape[1])[:, None]
    idx = jnp.broadcast_to(idx, members.shape[1:])
    idx = jnp.stack((idx, members.indices), 2).reshape(-1, 2)
    data = members.distances.flatten()
    return sparse.BCOO(
            (data, idx), shape=(members.shape[1], members.shape[1]),
            unique_indices=True)

@partial(jax.jit, static_argnames=("k",))
def sparse_pca(rng, arr, k):
    assert arr.ndim == 2 and arr.shape[0] == arr.shape[1]
    rng, subkey = jax.random.split(rng)
    x = jax.random.normal(subkey, (arr.shape[0], k))
    theta, u, i = sparse.sparsify(sparse.linalg.lobpcg_standard)(arr, x)
    return rng, u

# not specified in the paper, but in the official implementation, with the note:
# we add a little noise to avoid local minima for optimization to come
def noisy_scale(rng, data, hi=10.0, noise=0.0001):
    data *= hi / jnp.max(jnp.abs(data))
    data += noise * jax.random.normal(rng, data.shape)
    lo = jnp.min(data, 0)
    return rng, hi * (data - lo) / (jnp.max(data, 0) - lo)

@partial(jax.jit, static_argnames=("n_components", "n_epochs"))
def initialize(rng, heap, n_components, n_epochs=None):
    graph = simplices(memberships(heap))
    default_epochs = 500 if graph.shape[0] <= 10_000 else 200
    n_epochs = default_epochs if n_epochs is None else n_epochs
    norm = n_epochs if n_epochs > 10 else default_epochs
    graph.data = jnp.where(
            graph.data < jnp.max(graph.data) / norm, 0., graph.data)
    rng, embedding = noisy_scale(*sparse_pca(rng, graph, n_components))
    n_samples = n_epochs * graph.data / jnp.max(graph.data)
    graph.data = jnp.where(n_samples > 0, n_epochs / n_samples, -1)
    return rng, embedding, graph, n_epochs

"""
from the original documentation:
    min_dist: float (optional, default 0.1)
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points
        on the manifold are drawn closer together, while larger values will
        result on a more even dispersal of points. The value should be set
        relative to the ``spread`` value, which determines the scale at which
        embedded points will be spread out.

    spread: float (optional, default 1.0)
        The effective scale of embedded points. In combination with ``min_dist``
        this determines how clustered/clumped the embedded points are.
"""
@partial(jax.jit, static_argnames=("a", "b"))
def fit_ab(spread=1.0, min_dist=0.1, a=None, b=None):
    ndim = int(a is None) + int(b is None)
    if ndim == 0:
        return a, b
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))
    def lsq(ab):
        _a, _b = ab[0] if a is None else a, ab[1] if b is None else b
        return jnp.sum((curve(x, ab[0], ab[1]) - y) ** 2)

    x = jnp.linspace(0, spread * 3, 300)
    y = jnp.where(x < min_dist, 1., jnp.exp(-(x - min_dist) / spread))
    return optimize.minimize(lsq, jnp.ones((ndim,)), method="BFGS")[0]

@partial(jax.jit, static_argnames=("move_other",))
def optimize_embedding(
        rng, embedding, graph, n_epochs, a, b, gamma=1.,
        move_other=False, negative_sample_rate=5):
    args = (embedding,) * 2
    def cond(freq, n):
        return n % freq < 1
    def epoch(i, n, rng, head_embedding, tail_embedding):
        alpha = 1 - n / n_epochs
        j, k = graph.indices[i]
        current, other = head_embedding[j], tail_embedding[k]
        dist2 = jnp.sum((current - other) ** 2)
        coeff = jnp.where(dist2 > 0, -2 * a * b * jnp.pow(dist2, b - 1) / (
                a * jnp.pow(dist2, b) + 1), 0)
        coeff = jnp.clip(coeff * (current - other))
        head_embedding = head_embedding.at[j].set(current + coeff * alpha)
        if move_other:
            tail_embedding = tail_embedding.at[k].set(other - coeff * alpha)
        def negative(p, args):
            rng, head_embedding, tail_embedding = args
            rng, subkey = jax.random.split(rng)
            k = jax.random.randint(subkey, (), 0, tail_embedding.shape[0] - 1)
            other = tail_embedding[k]
            dist2 = jnp.sum((current - other) ** 2)
            coeff = jnp.where(dist2 > 0, 2 * gamma * b / (0.001 + dist2) / (
                    a * jnp.pow(dist2, b) + 1), 0)
            coeff = jnp.clip(coeff * (current - other))
            head_embedding = head_embedding.at[j].set(current + coeff * alpha)
            return rng, head_embedding, tail_embedding
        return jax.lax.fori_loop(0, negative_sample_rate, negative, (
                rng, head_embedding, tail_embedding))
    args = jax.lax.fori_loop(0, n_epochs, lambda n, a: jax.lax.fori_loop(
            0, graph.data.shape[0], lambda i, a: jax.lax.cond(
                graph.data[i] > 0 & cond(graph.data[i], n),
                lambda i, n, a: epoch(i, n, *a),
                lambda i, n, a: a, i, n, a), a), (rng, *args))
    return args

if __name__ == "__main__":
    from nnd import test_step
    data, heap = npy_cache("test_step")
    init = initialize(jax.random.key(0), heap, 3)
    print(optimize_embedding(*init, *fit_ab()))

