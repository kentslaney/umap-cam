from functools import partial
from jax.experimental import sparse
from jax.scipy import optimize, linalg
import jax.numpy as jnp
import jax

from group import grouping, groupaux, outgroup

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
    res = jnp.exp(-(knn.distances - smoothed.rho) / smoothed.sigma)
    res = knn.at["distances"].set(res)
    bound = knn.distances[knn.indices.flatten(), -1].reshape(knn.indices.shape)
    return res.at["flags"].set(knn.distances <= bound)

@jax.jit
def simplices(members, fuzziness=1.):
    idx = jnp.arange(members.shape[1])[:, None]
    idx = jnp.broadcast_to(idx, members.shape[1:])
    idx = jnp.stack((idx, members.indices), 2).reshape(-1, 2)
    data = members.distances.flatten()
    bidirectional = members.flags.flatten()
    def t(idx, flag):
        lookup, row = members[:2, idx[1]]
        first = jnp.argmax(row == idx[0])
        value = jnp.where(first > 0, lookup[first], jnp.where(
                row[0] == idx[0], lookup[0], 0))
        return jnp.where(flag, value, 0)
    transpose = jax.vmap(t)(idx, bidirectional)
    def op(m, t):
        p = m * t
        return fuzziness * (m + t - p) + (1 - fuzziness) * p
    res = jax.vmap(op)(data, transpose)
    flip = jnp.where(transpose == 0, data, 0)
    idx = jnp.concatenate((idx, idx[:, ::-1]))
    res = jnp.concatenate((res, fuzziness * flip))
    return sparse.BCOO(
            (res, idx), shape=(members.shape[1], members.shape[1]))

@partial(jax.jit, static_argnames=("k",))
def sparse_pca(rng, arr, k):
    assert arr.ndim == 2 and arr.shape[0] == arr.shape[1]
    rng, subkey = jax.random.split(rng)
    x = jax.random.normal(subkey, (arr.shape[0], k))
    theta, u, i = sparse.sparsify(sparse.linalg.lobpcg_standard)(arr, x)
    return rng, u # different than below

def dense_pca(rng, arr, k):
    # FP errors accumulate once arr is large enough, but other problems first
    arr -= jnp.mean(arr, axis=0, keepdims=True)
    u, s, vh = linalg.svd(arr, full_matrices=False)
    return rng, u[:, :k] * s[:k]

scale = 1

# not specified in the paper, but in the official implementation, with the note:
# we add a little noise to avoid local minima for optimization to come
def noisy_scale(rng, data, hi=1.0 * scale, noise=0.00001 * scale):
    data *= hi / jnp.max(jnp.abs(data))
    data += noise * jax.random.normal(rng, data.shape)
    lo = jnp.min(data, 0)
    return rng, hi * (data - lo) / (jnp.max(data, 0) - lo)

# weights depend on n_epochs
@jax.tree_util.register_pytree_node_class
class Adjacencies(outgroup("entries", "size"), groupaux("n_epochs"), grouping(
        "Adjacencies", names=("head", "tail", "weight"))):
    @classmethod
    def from_sparse(cls, graph, n_epochs=None):
        default_epochs = 500 if graph.shape[0] <= 10_000 else 200
        n_epochs = default_epochs if n_epochs is None else n_epochs
        norm = n_epochs if n_epochs > 10 else default_epochs
        data = jnp.where(
                graph.data < jnp.max(graph.data) / norm, 0., graph.data)
        n_samples = n_epochs * data / jnp.max(data)
        weights = jnp.where(n_samples > 0, n_epochs / n_samples, -1)
        nulls = weights < 0
        entries, order = weights.shape[0] - jnp.sum(nulls), jnp.argsort(nulls)
        args = (*graph.indices[order].T, weights[order])
        return cls(
                *args, n_epochs=n_epochs, entries=entries, size=graph.shape[0])

    def iter(self, callback, args):
        return jax.lax.fori_loop(0, self.entries, callback, args)

    def filtered(self, cond, callback, *args):
        return self.iter(lambda i, a: jax.lax.cond(cond(
                self.weight[i]), callback, lambda i, *a: a, i, *a), args)

    @classmethod
    def literal(cls, graph):
        nulls = graph.data == 0
        order = jnp.argsort(nulls)
        entries = graph.data.shape[0] - jnp.sum(nulls)
        return cls(
                *graph.indices[order].T, graph.data[order], size=graph.shape[0],
                entries=entries, n_epochs=0)

    def sorted(self):
        order = jnp.argsort(self.head * self.size + self.tail)
        _, order = jax.lax.sort_key_val(order >= self.entries, order)
        return self.__class__(
                self.head[order], self.tail[order], self.weight[order],
                n_epochs=self.n_epochs, entries=self.entries, size=self.size)

@partial(jax.jit, static_argnames=("n_components", "n_epochs"))
def initialize(rng, data, heap, n_components, n_epochs=None):
    assert n_components <= heap.shape[2]
    graph = simplices(memberships(heap))
    adj = Adjacencies.from_sparse(graph, n_epochs)
    rng, embedding = noisy_scale(*dense_pca(rng, data, n_components))
    return rng, embedding, adj

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
def fit_ab(spread=0.1 * scale, min_dist=0.01 * scale, a=None, b=None):
    ndim = int(a is None) + int(b is None)
    if ndim == 0:
        return a, b
    def curve(x, a, b):
        return 1.0 / (1.0 + a * (10 / scale * x) ** (2 * b))
    def lsq(ab):
        _a, _b = ab[0] if a is None else a, ab[1] if b is None else b
        return jnp.sum((curve(x, ab[0], ab[1]) - y) ** 2)

    x = jnp.linspace(0, spread * 3, 300)
    y = jnp.where(x < min_dist, 1., jnp.exp(-(x - min_dist) / spread))
    return optimize.minimize(lsq, jnp.ones((ndim,)), method="BFGS")[0]

class BaseOptimizer:
    order = (("a", "b"), (
            "move_other", "gamma", "negative_sample_rate", "verbose"))
    def __init__(
            self, spread=0.1 * scale, min_dist=0.01 * scale, a=None, b=None,
            move_other=True, gamma=1, negative_sample_rate=5, verbose=True):
        self.a, self.b = fit_ab(
                spread, min_dist, a, b) if a is None or b is None else (a, b)
        self.move_other, self.verbose = move_other, verbose
        self.gamma, self.negative_sample_rate = gamma, negative_sample_rate

    def tree_flatten(self):
        return tuple(tuple(getattr(self, j) for j in i) for i in self.order)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        matched = zip(cls.order, (children, aux_data))
        return cls(**{j: k for i in matched for j, k in zip(*i)})

    @staticmethod
    def dist(current, other):
        return (10 / scale) ** 2 * jnp.sum((current - other) ** 2)

    def phi(self, current, other):
        dist = self.dist(current, other)
        dist = jnp.where(dist > 0, dist, 0)
        return 1 / (1 + self.a * dist ** self.b)

    def positive_loss(self, current, other):
        return jnp.log(self.phi(current, other))

    def negative_loss(self, current, other):
        phi = self.phi(current, other)
        phi = jnp.where(phi < 1, phi, 0)
        return self.gamma * jnp.log(1 - phi)

    def step(self, i, n, rng, head_embedding, tail_embedding, adj):
        raise NotImplementedError

    iterator = Adjacencies
    def epoch(self, f, *args):
        return f(self.step, *args)

    @jax.jit
    def optimize(self, rng, embedding, adj):
        args = (embedding,) * 2
        def cond(freq, n):
            return n % freq < 1
        def update(n):
            print("", n, end="\r")
        def loop(n, a):
            if self.verbose:
                jax.debug.callback(update, n)
            f = partial(self.iterator.filtered, adj, lambda x: cond(x, n))
            wrapper = lambda g, *a, **kw: f(
                    lambda i, n, *a: g(i, n, *a), *a, **kw)
            return self.epoch(wrapper, n, *a)
        return jax.lax.fori_loop(0, adj.n_epochs, loop, (rng, *args, adj))[:-1]

@jax.tree_util.register_pytree_node_class
class Optimizer(BaseOptimizer):
    @staticmethod
    def clip(grad):
        return jnp.clip(grad, -0.04 * scale ** 2, 0.04 * scale ** 2)

    def negative_sample(self, rng, current, tail_embedding):
        rng, subkey = jax.random.split(rng)
        k = jax.random.randint(subkey, (), 0, tail_embedding.shape[0])
        other = tail_embedding[k]
        return self.negative_loss(current, other)

    def sample(self, head_embedding, tail_embedding, rng, j, k):
        current, other = head_embedding[j], tail_embedding[k]
        loss = self.positive_loss(current, other)
        rng, *subkeys = jax.random.split(rng, self.negative_sample_rate + 1)
        loss += jnp.sum(jax.vmap(
                lambda rng: self.negative_sample(
                    rng, current, jax.lax.stop_gradient(tail_embedding)),
                )(jnp.stack(subkeys)))
        return loss, rng

    def step(self, i, n, rng, head_embedding, tail_embedding, adj):
        alpha = 1 - n / adj.n_epochs
        grad, rng = jax.grad(self.sample, (0, 1), True)(
                head_embedding, tail_embedding, rng, adj.head[i], adj.tail[i])
        positive, negative = map(self.clip, grad)
        head_embedding += positive * alpha
        if self.move_other:
            tail_embedding += negative * alpha
        return rng, head_embedding, tail_embedding, adj

class CommutativeAdjacencies(Adjacencies):
    def filtered(self, cond, callback, *args, mt=None):
        return jax.vmap(lambda i, j: jax.lax.cond(
                cond(j), callback, lambda i, *a: a if mt is None else mt,
                i, *args))(jnp.arange(self.shape[1]), self.weight)

@jax.tree_util.register_pytree_node_class
class AccumulatingOptimizer(Optimizer):
    iterator = CommutativeAdjacencies
    def epoch(self, f, n, rng, head, tail, adj):
        rng, *subkeys = jax.random.split(rng, adj.shape[1] + 1)
        subkeys = jnp.stack(subkeys)
        rng_wrapper = lambda i, n, *a: self.step(i, n, subkeys[i], *a)
        delta_head, delta_tail = map(partial(jnp.sum, axis=0), f(
                rng_wrapper, n, head, tail, adj,
                mt=tuple(jnp.zeros((2, *head.shape)))))
        return rng, head + delta_head, tail + delta_tail, adj

    def step(self, i, n, rng, head_embedding, tail_embedding, adj):
        alpha = 1 - n / adj.n_epochs
        grad, rng = jax.grad(self.sample, (0, 1), True)(
                head_embedding, tail_embedding, rng, adj.head[i], adj.tail[i])
        positive, negative = map(self.clip, grad)
        head = positive * alpha
        tail = negative * alpha if self.move_other else jnp.zeros_like(head)
        return head, tail

# TODO: constrained optimization solution just needs the shape to be right
#   ... scale bounds to domain and apply gradient to (soft-)boundary points
#       the best option is probably just adding the rescale factor to the loss
#       it adds a hyperparameter trading off space efficiency vs accuracy though
#       penalize soft boundary via beta distribution non-linearity
if __name__ == "__main__":
    # from sklearn.datasets import load_digits
    # from nnd import aknn, NNDHeap
    # import pathlib
    # data = load_digits().data
    # path = pathlib.Path.cwd() / "digits.npy"
    # if not path.is_file():
    #     rng = jax.random.key(0)
    #     rng, heap = aknn(14, rng, data)
    #     jnp.save(path, heap)
    # else:
    #     heap = jnp.load(path)
    #     heap = NNDHeap.tree_unflatten((), heap)
    #
    # rng = jax.random.key(0)
    # rng, embed, adj = initialize(rng, data, heap, 2)
    # rng, lo, hi = Optimizer().optimize(rng, embed, adj)

    from nnd import npy_cache
    data, heap = npy_cache("test_step", neighbors=14)
    rng, embed, adj = initialize(jax.random.key(0), data, heap, 2)
    rng, lo, hi = AccumulatingOptimizer().optimize(rng, embed, adj)
    # print(lo)
    # exit(0)

    import matplotlib.pyplot as plt
    fig0 = plt.figure()
    ax0 = fig0.add_subplot(1, 1, 1)
    ax0.scatter(*lo.T)
    plt.show()
