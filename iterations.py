import jax.numpy as jnp
import jax

from umap import Optimizer

jax.config.update("jax_debug_nans", True)

# repetitive code iterating on the existing layout method

@jax.tree_util.register_pytree_node_class
class AccumulatingOptimizer(Optimizer):
    def negative_sample(self, grad, rng, current, tail_embedding, j):
        rng, subkey = jax.random.split(rng)
        k = jax.random.randint(subkey, (), 0, tail_embedding.shape[0] - 1)
        other = tail_embedding[k + (k >= j)]
        coeff = self.clip(jax.grad(self.negative_loss, 1)(current, other))
        return grad + coeff, rng, current, tail_embedding, j

    def epoch(self, i, n, rng, head_embedding, tail_embedding, adj):
        alpha, j, k = 1 - n / adj.n_epochs, adj.head[i], adj.tail[i]
        current, other = head_embedding[j], tail_embedding[k]
        positive = self.clip(jax.grad(self.positive_loss, 1)(current, other))
        grad, rng, *_ = jax.lax.fori_loop(
                0, self.negative_sample_rate,
                lambda p, a: self.negative_sample(*a),
                (positive, rng, current, tail_embedding, j))
        head_embedding = head_embedding.at[j].set(current + grad * alpha)
        if self.move_other:
            tail_embedding = tail_embedding.at[k].set(other - positive * alpha)
        return rng, head_embedding, tail_embedding, adj

@jax.tree_util.register_pytree_node_class
class GlobalClipOptimizer(Optimizer):
    def negative_sample(self, loss, rng, current, tail_embedding):
        rng, subkey = jax.random.split(rng)
        k = jax.random.randint(subkey, (), 0, tail_embedding.shape[0])
        other = tail_embedding[k]
        loss += self.negative_loss(current, other)
        return loss, rng, current, tail_embedding

    def sample(self, current, other, rng, tail_embedding):
        positive = self.positive_loss(current, other)
        loss, rng, *_ = jax.lax.fori_loop(
                0, self.negative_sample_rate,
                lambda p, a: self.negative_sample(*a),
                (positive, rng, current, tail_embedding))
        return loss, rng

    def epoch(self, i, n, rng, head_embedding, tail_embedding, adj):
        alpha, j, k = 1 - n / adj.n_epochs, adj.head[i], adj.tail[i]
        current, other = head_embedding[j], tail_embedding[k]
        grad, rng = jax.grad(self.sample, (0, 1), True)(
                current, other, rng, tail_embedding)
        positive, negative = map(self.clip, grad)
        head_embedding = head_embedding.at[j].set(current + positive * alpha)
        if self.move_other:
            tail_embedding = tail_embedding.at[k].set(other + negative * alpha)
        return rng, head_embedding, tail_embedding, adj

if __name__ == "__main__":
    from nnd import npy_cache
    from umap import initialize
    data, heap = npy_cache("test_step")
    init = initialize(jax.random.key(0), heap, 3)
    rng, lo, hi = GlobalClipOptimizer().optimize(*init)
    print(lo)

