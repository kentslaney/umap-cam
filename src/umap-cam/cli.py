import argparse, pathlib
project_dir = pathlib.Path(__file__).parents[2]

def image_cli(parser):
    parser.add_argument("npy")
    parser.add_argument(
            "--job-dir", default=project_dir / "jobs", type=pathlib.Path)
    parser.add_argument("--quiet", action="store_false", dest="verbose")
    parser.add_argument("--k", type=int, default=15)
    parser.set_defaults(call=image)

def image(args):
    import jax
    import jax.numpy as jnp
    from nnd_avl import aknn, NNDResult
    from umap import initialize
    from color import CAM16Optimizer

    job_dir = args.job_dir / pathlib.Path(args.npy).stem
    job_dir.mkdir(exist_ok=True)
    im = jnp.load(args.npy)
    shape, im = im.shape, jnp.reshape(im, (-1, im.shape[-1]))
    knn_path, umap_path = job_dir / "knn.npy", job_dir / "umap.npy"
    if knn_path.is_file():
        print("reusing stored AKNN result...")
        heap = NNDResult.tree_unflatten((), jnp.load(knn_path))
    else:
        print("starting AKNN...")
        _, heap = aknn(args.k, jax.random.key(0), im, verbose=True)
        jnp.save(knn_path, jnp.stack(heap))
    rng, embed, adj = initialize(jax.random.key(1), im, heap, n_components=3)
    optimizer = CAM16Optimizer(verbose=True)
    rng, lo, hi = optimizer.optimize(rng, embed, adj)
    lo = lo.reshape(shape)
    jnp.save(umap_path, lo)

def reformat_cli(parser):
    parser.add_argument("embeddings")
    parser.add_argument("neighbors")
    parser.add_argument(
            "--job-dir", default=project_dir / "jobs", type=pathlib.Path)
    parser.set_defaults(call=reformat)

def reformat(args):
    import jax.numpy as jnp
    job_dir = args.job_dir / pathlib.Path(args.embeddings).stem
    job_dir.mkdir(exist_ok=True)
    data, indices = jnp.load(args.embeddings), jnp.load(args.neighbors)
    flags = jnp.zeros_like(indices)
    points = data[indices.flatten()].reshape(*indices.shape, -1)
    distances = jnp.sum((data[:, None, :] - points) ** 2, -1)
    jnp.save(job_dir / "knn.npy", jnp.stack((distances, indices, flags)))

helper = lambda parser: lambda args: parser.parse_args(["-h"])

def main(parser):
    subparser = parser.add_subparsers()
    image_cli(subparser.add_parser("image"))
    reformat_cli(subparser.add_parser("reformat"))
    parser.set_defaults(call=helper(parser))
    args = parser.parse_args()
    args.call(args)

if __name__ == "__main__":
    main(argparse.ArgumentParser())
