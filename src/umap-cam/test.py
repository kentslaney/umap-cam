import jax
import jax.numpy as jnp
import sys, unittest, pathlib, hashlib, inspect, argparse, importlib
from functools import partial, wraps

def fold(rng, uniq: str):
    uniq = bytes(uniq, "UTF-8")
    rng, subkey = jax.random.split(rng)
    salt = jax.random.bits(subkey, 4)
    salt = b''.join(i.to_bytes(4) for i in list(map(int, salt)))
    dk = hashlib.pbkdf2_hmac('sha256', uniq, salt, 1_000) # insecure
    dk = jnp.uint32(int.from_bytes(dk[:4]))
    return jax.random.fold_in(rng, dk), dk

class RouteResult:
    def __init__(self, routes):
        self.routes = routes

    def __getitem__(self, i):
        f = self.routes[i]
        return getattr(inspect.getmodule(f),
                f.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0])

    def __call__(self, i):
        return self[i](self.routes[i].__name__)

class Depends:
    def __init__(self, path, seed=0, options: (str, None)=None):
        self.path = pathlib.Path(path).resolve()
        rng = jax.random.key(seed)
        self._rng = rng if options is None else fold(rng, options)[0]
        self.edges, self.outputs, self.cache_rng = {}, {}, {}
        self.failed, self.complete, self.validated = set(), set(), set()
        self.routes, self.names = {}, {}
        self.classes = RouteResult(self.routes)
        self.ro_cache = False

    prefix = "test_"
    def __call__(self, *on, rng=False, **renamed):
        needed = tuple((*on, *renamed.values()))
        assert not any(i.startswith(self.prefix) for i in needed)
        def decorator(f):
            @wraps(f)
            def wrapper(*a, **kw):
                deps, a, kw = self.bind_deps(uniq, f, a, kw)
                if rng:
                    subkey, seed = self.rng(uniq, deps)
                    self.cache_rng[uniq] = seed
                    self.validated.add(uniq)
                    a, kw = self.bind(f, 'rng', subkey, a, kw)
                try:
                    res = f(*a, **kw)
                except AssertionError:
                    self.failed.add(uniq)
                    raise
                self.complete.add(uniq)
                self.outputs[uniq] = 0 if res is None else res

            assert f.__name__.startswith(self.prefix)
            uniq = f.__name__[len(self.prefix):]
            self.edges[uniq] = list(sorted(needed))
            self.names[uniq] = dict(zip(renamed.values(), renamed.keys()))
            self.routes[uniq] = wrapper
            return wrapper
        return decorator

    rng_prefix = "cached_"
    def load(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.suffix.endswith(".npz"):
            self.path = self.path.parent / self.path.basename + ".npz"
        if not self.path.is_file():
            return
        outputs = dict(jnp.load(str(self.path)))
        self.outputs = {
                k[len(self.prefix):]: v for k, v in outputs.items()
                if k.startswith(self.prefix)}
        self.cache_rng = {
                k[len(self.rng_prefix):]: v for k, v in outputs.items()
                if k.startswith(self.rng_prefix)}
        fold = [k for k in outputs if k.startswith("_" + self.prefix)]
        sizes = [[k[:-len(i)] for k in outputs if k.endswith(i)] for i in fold]
        for k, i in zip(fold, sizes):
            aux_size, child_size = map(max, zip(*(
                (0, 1 + int(j.split("_", 2)[1]))[
                    ::-1 if j.startswith("aux_") else 1]
                for j in i if j)))
            aux = [outputs[f"aux_{i}{k}"] for i in range(aux_size)]
            children = [outputs[f"child_{i}{k}"] for i in range(child_size)]
            if outputs[k] == "tuple":
                res = tuple(children)
            else:
                mod, name = str(outputs[k]).split("-", 1)
                mod = importlib.import_module(mod)
                for name in name.split("."):
                    mod = getattr(mod, name)
                res = mod.tree_unflatten(*map(tuple, (aux, children)))
            self.outputs[k[len(self.prefix) + 1:]] = res

    def save(self):
        if self.ro_cache:
            return
        outputs = {self.prefix + k: v for k, v in self.outputs.items()}
        res = {self.rng_prefix + k: v for k, v in self.cache_rng.items()}
        for k, v in outputs.items():
            if not hasattr(v, "tree_flatten"):
                if isinstance(v, tuple):
                    res["_" + k] = "tuple"
                    for i in range(len(v)):
                        res[f"child_{i}_{k}"] = v[i]
                else:
                    res[k] = v
                continue
            res["_" + k] = (
                    inspect.getmodule(v.__class__).__name__ + "-" +
                    v.__class__.__qualname__)
            children, aux_data = v.tree_flatten()
            for i in range(len(children)):
                res[f"child_{i}_{k}"] = children[i]
            for i in range(len(aux_data)):
                res[f"aux_{i}_{k}"] = aux_data[i]
        jnp.savez(str(self.path), **res)

    def topological(self, uniq=None):
        unseen = set(self.edges)
        def body(uniq):
            if uniq in unseen:
                unseen.remove(uniq)
                return sum((body(i) for i in self.edges[uniq]), []) + [uniq]
            return []
        uniq = sorted(unseen) if uniq is None else (uniq,)
        return sum((body(i) for i in uniq), [])

    def rng(self, uniq, deps=None):
        deps = self.topological(uniq) if deps is None else deps
        return fold(self._rng, "-".join(deps))

    TEST_FAILED, TEST_PASSED, TEST_CACHED, TEST_STALE, TEST_AWOL = range(5)
    def status(self, uniq):
        res = self.TEST_FAILED if uniq in self.failed else \
                self.TEST_PASSED if uniq in self.complete else \
                self.TEST_CACHED if uniq in self.outputs else self.TEST_AWOL
        if res is self.TEST_CACHED and uniq in self.cache_rng and \
                uniq not in self.validated:
            _, seed = self.rng(uniq)
            if self.cache_rng.get(uniq, None) == seed:
                self.validated.add(uniq)
            else:
                return self.TEST_STALE
        return res

    def matches(self, deps, status, query):
        match = [i for i, j in zip(deps, status) if j is query]
        end = "and " if len(match) > 1 else ""
        return ", ".join(match[:-1] + [end + match[-1]]) if match else None

    def ensure(self, uniq, f, a, kw):
        # unittest.skip raises unittest.SkipTest
        deps = self.topological(uniq)
        status = tuple(map(self.status, deps[:-1]))
        if self.TEST_AWOL in status:
            awol = self.matches(deps, status, self.TEST_AWOL)
            raise RuntimeError(f"test {uniq} run before {awol}")
        elif self.TEST_FAILED in status:
            failed = self.matches(deps, status, self.TEST_FAILED)
            failed = f"dependency tests {failed} for {uniq} failed"
            unittest.skip(failed)(f)(*a, **kw)
        elif self.TEST_STALE in status:
            stale = self.matches(deps, status, self.TEST_STALE)
            stale = f"cached results for dependency tests {stale} are stale"
            unittest.skip(stale)(f)(*a, **kw)
        return deps

    @staticmethod
    def bind(f, arg, value, a, kw, sig=None):
        if arg in kw:
            return a, kw
        sig = inspect.getfullargspec(f) if sig is None else sig
        if arg in sig.args and len(a) > (pos := sig.args.index(arg)):
            a.insert(value, pos)
        else:
            kw = {**kw, arg: value}
        return a, kw

    def bind_deps(self, uniq, f, a, kw):
        deps = self.ensure(uniq, f, a, kw)
        sig = inspect.getfullargspec(f)
        for dep in deps:
            if dep in sig.args + sig.kwonlyargs or dep in self.names[uniq]:
                name = self.names[uniq].get(dep, dep)
                a, kw = self.bind(f, name, self.outputs[dep], a, kw, sig)
        return deps, a, kw

    @property
    def caching(self):
        class DependsTearDown(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                self.load()

            @classmethod
            def tearDownClass(cls):
                self.save()
        return DependsTearDown

    @property
    def loader(self):
        order = self.topological()
        default = unittest.defaultTestLoader.sortTestMethodsUsing
        class DependsLoader(unittest.TestLoader):
            @staticmethod
            def sortTestMethodsUsing(a, b):
                a, b = (
                        i[len(self.prefix):] if i.startswith(self.prefix) else i
                        for i in (a, b))
                for i in (a, b):
                    if i not in order:
                        order.append(i)
                a, b = order.index(a), order.index(b)
                return (a > b) - (a < b)
        return DependsLoader()

    # doesn't accept CLI args but can initialize the cache
    def suite(self):
        suite = unittest.TestSuite()
        for i in self.topological():
            suite.addTest(self.classes(i))
        unittest.TextTestRunner().run(suite)

    def as_needed(self, names):
        updated = False
        for dep in names:
            if dep not in depends.outputs:
                depends.classes(dep).run()
                updated = True
            elif dep not in self.validated:
                _, seed = self.rng(dep)
                if self.cache_rng.get(dep, None) == seed:
                    self.validated.add(dep)
                else:
                    depends.classes(dep).run()
                    updated = True
        if updated:
            depends.save()

class Options:
    hashing = (
            "points", "k_neighbors", "max_candidates", "ndim", "n_trees",
            "n_nnd_iter")
    names = ("seed",) + hashing
    points = 32
    k_neighbors = 15
    max_candidates = 7
    ndim = 8
    n_trees = 2
    n_nnd_iter = 3
    seed = 0

    @property
    def hashed(self) -> str:
        return "_".join(str(getattr(self, i)) for i in self.hashing)

    @property
    def rng(self):
        return self.seed, self.hashed

opt = Options()
file_dir = pathlib.Path(__file__).parent
project_dir = file_dir / pathlib.os.pardir / pathlib.os.pardir
depends = Depends(project_dir / "test_cache.npz", *opt.rng)
file_dir = str(file_dir)

class FlatTest:
    @classmethod
    def setUpClass(cls):
        if file_dir not in sys.path:
            sys.path.insert(0, file_dir)
        if hasattr(super(), "setUpClass"):
            super().setUpClass()

class TestPipelinedUMAP(FlatTest, depends.caching):
    @depends(rng=True)
    def test_data(self, rng):
        return jax.random.normal(rng, (opt.points, opt.ndim))

    @depends("data", rng=True)
    def test_rpt(self, rng, data):
        from rpt import forest
        rng, total, trees = forest(rng, data, opt.n_trees, opt.max_candidates)
        trees.block_until_ready()
        return total, trees

    @depends("data", rng=True)
    def test_heap(self, rng, data):
        from nnd import NNDHeap
        heap = NNDHeap(opt.points, opt.k_neighbors)
        heap, rng = heap.randomize(data, rng)
        heap.distances.block_until_ready()
        return heap

    @depends("data", rng=True)
    def test_heap_avl(self, rng, data):
        from nnd_avl import NNDHeap
        heap = NNDHeap(opt.points, opt.k_neighbors)
        heap, rng = heap.randomize(data, rng)
        heap.distances.block_until_ready()
        self.assertTrue(jnp.all(heap.indices < heap.spec.points))
        return heap

    @depends("heap", forest="rpt")
    def test_rpt_heap(self, forest, heap, data):
        from nnd import RPCandidates
        total, trees = forest
        trees = RPCandidates(trees, total=total)
        heap, count = heap.update(data, trees)
        heap.distances.block_until_ready()
        return heap

    @depends(forest="rpt", heap="heap_avl")
    def test_rpt_heap_avl(self, forest, heap, data):
        from nnd_avl import RPCandidates
        total, trees = forest
        trees = RPCandidates(trees, total=total, data_points=data.shape[0])
        heap, count = heap.update(data, trees)
        heap.distances.block_until_ready()
        return heap

    @depends(rng=True, heap="rpt_heap")
    def test_heap_build(self, rng, heap, data):
        for i in range(opt.n_nnd_iter):
            heap, step, rng = heap.build(opt.max_candidates, rng)
            heap, changes = heap.update(data, step)
        heap.distances.block_until_ready()
        return heap

    @depends(rng=True, heap="rpt_heap_avl")
    def test_heap_avl_build(self, rng, heap, data):
        for i in range(opt.n_nnd_iter):
            heap, step, rng = heap.build(opt.max_candidates, rng)
            heap, changes = heap.update(data, step)
        heap.distances.block_until_ready()
        return heap

    @depends(rng=True, heap="rpt_heap_avl")
    def test_heap_avl_initial_changes(self, rng, heap, data):
        heap, step, rng = heap.build(opt.max_candidates, rng)
        heap, changes = heap.update(data, step)
        self.assertNotEqual(changes, 0)

    @depends(rng=True, heap="heap_avl_build")
    def test_umap(self, rng, heap, data):
        from umap import initialize, AccumulatingOptimizer
        rng, embed, adj = initialize(rng, data, heap, n_components=2)
        optimizer = AccumulatingOptimizer(verbose=False)
        rng, lo, hi = optimizer.optimize(rng, embed, adj)
        std = jnp.stack(tuple(map(jnp.std, (lo, hi))))
        delta = 2 * (std[1] - std[0]) / jnp.sum(std)
        lo.block_until_ready()
        return lo

    @depends(rng=True, heap="heap_avl_build")
    def test_constrained_umap(self, rng, heap, data):
        from umap import initialize, ConstrainedOptimizer
        rng, embed, adj = initialize(rng, data, heap, n_components=3)
        optimizer = ConstrainedOptimizer(verbose=False, constrained_cols=2)
        rng, lo, hi = optimizer.optimize(rng, embed, adj)
        lo.block_until_ready()
        return lo

    @depends(rng=True, heap="heap_avl_build")
    def test_cam16_umap(self, rng, heap, data):
        from umap import initialize
        from color import CAM16Optimizer
        rng, embed, adj = initialize(rng, data, heap, n_components=4)
        optimizer = CAM16Optimizer(verbose=False)
        rng, lo, hi = optimizer.optimize(rng, embed, adj)
        lo.block_until_ready()
        return lo

    @depends(rng=True)
    def test_build_flipped(self, rng):
        max_candidates, k_neighbors = opt.k_neighbors, opt.max_candidates
        from nnd_avl import NNDHeap
        rng, subkey = jax.random.split(rng)
        data = jax.random.normal(subkey, (opt.points, opt.ndim))
        heap = NNDHeap(data.shape[0], k_neighbors)
        heap, rng = heap.randomize(data, rng)
        heap, step, rng = heap.build(max_candidates, rng)
        heap, changes = heap.update(data, step)
        self.assertNotEqual(changes, 0)

    @depends("data", rng=True)
    def test_aknn(self, rng, data):
        from nnd_avl import aknn
        rng, heap = aknn(
                opt.k_neighbors, rng, data, max_candidates=opt.max_candidates,
                n_trees=opt.n_trees)
        heap.distances.block_until_ready()
        return heap

    def test_avl_removals(self):
        from avl import SingularAVL
        tree = SingularAVL.tree_unflatten((), (
            [14.247807, 31.511902, 43.335896, 43.451122, 44.079475, 44.474712,
             14.247807, 44.079475, 28.495613, 34.655445, 34.899857, 34.971416,
             36.138622, 42.485290, 43.289722, 44.158806, 44.429718, 35.510563,
             36.864616, 41.291645, 42.071370, 42.284748, 42.673176, 44.237990,
             30.886890, 31.701735, 44.429718, 26.115130, 30.886890, 43.197224,
             43.474130, 43.554560],
            [  2, 121,  17,  81, 104,  57,   1, 103,  47,  68,  21, 122,  97,
              16, 162,  59,  54,  11, 158,  76, 107,  14,  69,  24,  87,  90,
              53,  70,  88, 147, 113, 100],
            [ 6,  8, -1, 12,  7, -1, -1, -1,  0, 25,  9, -1,  1, 19, 22, 31, 26,
             -1, -1, 18, -1, -1, -1, -1, -1, -1, -1, -1, 24, -1, -1, 30],
            [27, 10, -1, 15, -1, -1, -1, -1, 28, -1, 11, 17, 13, 14,  2, 16,  5,
             -1, -1, 20, 21, -1, 29, -1, -1, -1, -1, -1, -1, -1, -1,  4],
            [ 2,  4,  1,  6,  2,  1,  1,  1,  3,  2,  3,  2,  5,  4,  3,  4,  2,
              1,  1,  3,  2,  1,  2,  1,  1,  1,  1,  1,  2,  1,  1,  3],
            jnp.int32(3), jnp.int32(32), jnp.int32(5)))
        tree = tree.remove(16)
        self.assertTrue(tree.acyclic())

class TestDigitsIntegration(FlatTest, depends.caching):
    @depends()
    def test_digits_polyfill_aknn(self):
        from sklearn.datasets import load_digits
        from nnd_polyfill import aknn
        data = load_digits().data
        _, heap = aknn(15, None, data)
        heap.distances.block_until_ready()
        return heap

    @depends(rng=True)
    def test_digits_avl_aknn(self, rng):
        from sklearn.datasets import load_digits
        from nnd_avl import aknn
        data = load_digits().data
        rng, heap = aknn(15, rng, data)
        heap.distances.block_until_ready()
        return heap

    @depends(rng=True, heap="digits_avl_aknn")
    def test_digits_avl_umap(self, rng, heap):
        from sklearn.datasets import load_digits
        from umap import initialize, AccumulatingOptimizer
        data = load_digits().data
        rng, embed, adj = initialize(rng, data, heap, n_components=2)
        optimizer = AccumulatingOptimizer(verbose=False)
        rng, lo, hi = optimizer.optimize(rng, embed, adj)
        lo.block_until_ready()
        return lo

    @depends(rng=True)
    def test_digits_rpt(self, rng):
        from sklearn.datasets import load_digits
        from nnd_avl import RPCandidates, NNDHeap
        data = load_digits().data
        heap = NNDHeap(data.shape[0], opt.k_neighbors)
        heap, rng = heap.randomize(data, rng)
        rng, trees = RPCandidates.forest(
                rng, data, opt.n_trees, opt.max_candidates)
        heap, _ = heap.update(data, trees)
        heap.distances.block_until_ready()
        return total, trees

class TestCAM16(FlatTest, unittest.TestCase):
    @staticmethod
    def domain_bounds(lo=0, hi=1, points=100):
        inputs = jnp.linspace(lo, hi, points + 1)
        shape = (inputs.shape[0],) * 2
        row = jnp.broadcast_to(inputs[:, None], shape)
        col = jnp.broadcast_to(inputs[None, :], shape)
        lo, hi = jnp.full(shape, lo), jnp.full(shape, hi)
        faces = (
                (lo, row, col), (row, lo, col), (row, col, lo),
                (hi, row, col), (row, hi, col), (row, col, hi))
        return jnp.concatenate(tuple(map(
                lambda x: jnp.stack(x).T.reshape(-1, 3), faces)))

    def test_domain(self):
        from color import ViewingConditions
        vc = ViewingConditions()
        rgb = self.domain_bounds(-0.1, 1.1, 120)
        res = vc.broadcast(rgb)
        failures = jnp.any(jnp.isnan(res) | jnp.isinf(res), axis=1)
        self.assertEqual(jnp.sum(failures), 0)

    def test_boundaries(self):
        from color import ViewingConditions
        vc = ViewingConditions()
        rgb = jnp.asarray([[0.1, 0.4, 0.3], [0, 0.4, 0.3], [-0.1, 0.4, 0.3]])
        cam = vc.broadcast(rgb)
        area_2 = jnp.sum((jnp.cross(cam[1] - cam[0], cam[2] - cam[0])) ** 2)
        self.assertAlmostEqual(area_2, 0, 3)

    def test_scale(self):
        from color import ViewingConditions, CAM16Optimizer
        vc = ViewingConditions()
        rng = jax.random.key(0)
        rng, subkey = jax.random.split(rng)
        rgb = jax.random.uniform(subkey, (10_000, 3))
        cam = vc.broadcast(rgb)
        samples = jnp.int32(rgb.shape[0] ** 2 / 2 * 0.01)
        rng, subkey = jax.random.split(rng)
        cross = jax.random.randint(subkey, (samples, 2), 0, rgb.shape[0])
        deltas = jax.vmap(vc.delta)(*jnp.unstack(cam[cross], axis=1))
        reference = jnp.linalg.norm(rgb[cross][:, 1] - rgb[cross][:, 0], axis=1)
        sparse = vc.broadcast(self.domain_bounds(points=5))
        cross = jnp.stack([i.flatten()
                for i in jnp.meshgrid(*(jnp.arange(sparse.shape[0]),) * 2)])
        sparse = jax.vmap(vc.delta)(*jnp.unstack(sparse[cross.T], axis=1))

        distribution = (jnp.mean(deltas), jnp.std(deltas), jnp.max(sparse))
        identity = (jnp.mean(reference), jnp.std(reference), jnp.max(reference))
        default = CAM16Optimizer().color_scale
        self.assertAlmostEqual(distribution[0] / identity[0], default, 0)

def visualize(embedded):
    import matplotlib.pyplot as plt
    plt.scatter(*embedded.T)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ro-cache", "-r", action="store_true")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--perfetto", action="store_true")
    parser.add_argument("--manual-profile", nargs="?", default=False)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--umap-suite", "-u", action="store_true")
    group.add_argument("--debug", "-d", default=False, nargs="?")
    group.add_argument("--digits", action="store_true")
    group.add_argument("--profile", nargs="+")
    args, argv = parser.parse_known_args()
    if args.ro_cache:
        depends.ro_cache = True
    if args.manual_profile is not False:
        jax.profiler.start_server(args.manual_profile or 9999)
    if args.profile or args.digits or args.debug is not False:
        depends.load()
        FlatTest.setUpClass()
    if args.profile:
        for uniq in args.profile:
            depends.as_needed(depends.topological(uniq)[:-1])
            with jax.profiler.trace(
                    project_dir / "jobs", create_perfetto_link=args.perfetto):
                depends.classes(uniq).run()
        depends.save()
    elif args.digits:
        depends.as_needed(depends.topological("digits_avl_umap"))
        visualize(depends.outputs["digits_avl_umap"])
    elif args.debug:
        depends.as_needed(depends.topological(args.debug)[:-1])
        depends.classes(args.debug).debug()
        depends.save()
    elif args.umap_suite:
        depends.suite()
    else:
        unittest.main(argv=sys.argv[:1] + argv, testLoader=depends.loader)
    if args.interactive:
        for k, v in depends.outputs.items():
            globals()[k] = v

