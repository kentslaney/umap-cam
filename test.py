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

class Depends:
    def __init__(self, path, seed=0, options: (str, None)=None):
        self.path = pathlib.Path(path).resolve()
        rng = jax.random.key(seed)
        self._rng = rng if options is None else fold(rng, options)[0]
        self.edges, self.outputs, self.cache_rng = {}, {}, {}
        self.failed, self.complete, self.validated = set(), set(), set()
        self.routes = {}
        self.ro_cache = False

    prefix = "test_"
    def __call__(self, *on, rng=False):
        assert not any(i.startswith(self.prefix) for i in on)
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
            self.edges[uniq] = list(sorted(on))
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
                        res[f"child_{i}_{k}"] = children[i]
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
        end = " and " if len(match) > 1 else ""
        return ", ".join(match[:-1] + [end + match[-1]]) if match else None

    def ensure(self, uniq):
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
            unittest.skip(failed)(f)(*a, **kw)
        return deps

    @staticmethod
    def bind(f, arg, value, a, kw, sig=None):
        assert arg not in kw
        sig = inspect.getfullargspec(f) if sig is None else sig
        if arg in sig.args and len(a) > (pos := sig.args.index(arg)):
            a.insert(value, pos)
        else:
            kw = {**kw, arg: value}
        return a, kw

    def bind_deps(self, uniq, f, a, kw):
        deps = self.ensure(uniq)
        sig = inspect.getfullargspec(f)
        for dep in deps:
            if dep in sig.args or dep in sig.kwonlyargs:
                a, kw = self.bind(f, dep, self.outputs[dep], a, kw, sig)
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

    # BROKEN: the latest versions of Python 3 seem to ignore the order method
    # https://stackoverflow.com/q/4095319#comment120033036_22317851
    @property
    def loader(self):
        order = self.topological()
        class DependsLoader(unittest.TestLoader):
            @staticmethod
            def sortTestMethodsUsing(a, b):
                a, b = a[len(self.prefix):], b[len(self.prefix):]
                a, b = order.index(a), order.index(b)
                return (a > b) - (a < b)
        return DependsLoader()

    # doesn't accept CLI args but can initialize the cache
    def suite(self):
        suite = unittest.TestSuite()
        for i in self.topological():
            f = self.routes[i]
            test_case = getattr(inspect.getmodule(f),
                    f.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0])
            suite.addTest(test_case(f.__name__))
        unittest.TextTestRunner().run(suite)

class Options:
    hashing = (
            "points", "k_neighbors", "max_candidates", "ndim", "n_trees",
            "n_nnd_iter", "n_components")
    names = ("seed",) + hashing
    points = 32
    k_neighbors = 15
    max_candidates = 7
    ndim = 4
    n_trees = 2
    n_nnd_iter = 3
    n_components = 2
    seed = 0

    @property
    def hashed(self) -> str:
        return "_".join(str(getattr(self, i)) for i in self.hashing)

    @property
    def rng(self):
        return self.seed, self.hashed

opt = Options()
file_dir = pathlib.Path(__file__).parent
depends = Depends(file_dir / "test_cache.npz", *opt.rng)
file_dir = str(file_dir)

class FlatTest:
    @classmethod
    def setUpClass(cls):
        if file_dir not in sys.path:
            sys.path.insert(0, file_dir)
        super().setUpClass()

class TestPipelinedUMAP(FlatTest, depends.caching):
    @depends(rng=True)
    def test_data(self, rng):
        return jax.random.normal(rng, (opt.points, opt.ndim))

    @depends("data", rng=True)
    def test_rpt(self, rng, data):
        from rpt import forest
        rng, total, trees = forest(rng, data, opt.n_trees, opt.max_candidates)
        return total, trees

    @depends("data", rng=True)
    def test_heap(self, rng, data):
        from nnd import NNDHeap
        heap = NNDHeap(opt.points, opt.k_neighbors)
        heap, rng = heap.randomize(data, rng)
        return heap

    @depends("data", rng=True)
    def test_heap_avl(self, rng, data):
        from nnd_avl import NNDHeap
        heap = NNDHeap(opt.points, opt.k_neighbors)
        heap, rng = heap.randomize(data, rng)
        return heap

    @depends("rpt", "heap")
    def test_rpt_heap(self, rpt, heap, data):
        from nnd import RPCandidates
        total, trees = rpt
        trees = RPCandidates(trees, total=total)
        heap, count = heap.update(data, trees)
        return heap

    @depends("rpt", "heap_avl")
    def test_rpt_heap_avl(self, rpt, heap_avl, data):
        from nnd_avl import RPCandidates
        total, trees = rpt
        trees = RPCandidates(trees, total=total, data_points=data.shape[0])
        heap_avl, count = heap_avl.update(data, trees)
        return heap_avl

    @depends("rpt_heap", rng=True)
    def test_heap_build(self, rng, rpt_heap, data):
        for i in range(opt.n_nnd_iter):
            rpt_heap, step, rng = rpt_heap.build(opt.max_candidates, rng)
            rpt_heap, changes = rpt_heap.update(data, step)
        return rpt_heap

    @depends("rpt_heap_avl", rng=True)
    def test_heap_avl_build(self, rng, rpt_heap_avl, data):
        for i in range(opt.n_nnd_iter):
            rpt_heap_avl, step, rng = rpt_heap_avl.build(
                    opt.max_candidates, rng)
            rpt_heap_avl, changes = rpt_heap_avl.update(data, step)
        return rpt_heap_avl

    @depends("heap_avl_build", rng=True)
    def test_umap(self, rng, heap_avl_build, data):
        from umap import initialize, AccumulatingOptimizer
        rng, embed, adj = initialize(rng, data, heap_avl_build, n_components=2)
        optimizer = AccumulatingOptimizer(verbose=False)
        rng, lo, hi = optimizer.optimize(rng, embed, adj)
        std = jnp.stack(tuple(map(jnp.std, (lo, hi))))
        delta = 2 * (std[1] - std[0]) / jnp.sum(std)
        return lo

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
        assert jnp.sum(failures) == 0

    def test_boundaries(self):
        from color import ViewingConditions
        vc = ViewingConditions()
        rgb = jnp.asarray([[0.1, 0.4, 0.3], [0, 0.4, 0.3], [-0.1, 0.4, 0.3]])
        cam = vc.broadcast(rgb)
        area_2 = jnp.sum((jnp.cross(cam[1] - cam[0], cam[2] - cam[0])) ** 2)
        self.assertAlmostEqual(area_2, 0, 3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--umap-suite", action="store_true")
    parser.add_argument("--ro-cache", action="store_true")
    args, argv = parser.parse_known_args()
    if args.ro_cache:
        depends.ro_cache = True
    if args.umap_suite:
        depends.suite()
    else:
        unittest.main(argv=sys.argv[:1] + argv)

