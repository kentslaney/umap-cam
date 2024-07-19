from collections import namedtuple
import jax
import jax.numpy as jnp

WhitePoint = namedtuple("WhitePoint", ["x_w", "y_w", "z_w"])

# https://en.wikipedia.org/wiki/Standard_illuminant#Illuminant_series_D
def series_D(t, y_w=1.):
    t *= 1.438776877 / 1.4380
    _t = 1_000 / t
    if 4_000 <= t <= 7_000:
        x = 0.244063 + 0.09911 * _t + 2.9678 * _t ** 2 - 4.6070 * _t ** 3
    elif 7_000 < t <= 25_000:
        x = 0.237040 + 0.24748 * _t + 1.9018 * _t ** 2 - 2.0064 * _t ** 3
    else:
        raise Exception
    y = -3.000 * x ** 2 + 2.870 * x - 0.275
    return WhitePoint(y_w / y * x, y_w, y_w / y * (1 - x - y))

# Noon light
WhitePoint.D65 = series_D(6_500)
WhitePoint.D65_2DEG  = WhitePoint(0.95047, 1., 1.08883)
WhitePoint.D65_10DEG = WhitePoint(0.94811, 1., 1.07304)
# Horizon light
WhitePoint.D50 = series_D(5_000)

ColorSpec = namedtuple("ColorSpec", [
        "gamma", "x_r", "y_r", "x_g", "y_g", "x_b", "y_b"])

class Gamma:
    def __init__(self, gamma, a=0.):
        self.gamma, self.a = gamma, a
        self.x = a / (gamma - 1)
        self.phi = (1 + a) ** gamma * (gamma - 1) ** (gamma - 1) / \
                (a ** (gamma - 1) * gamma ** gamma)

    def __call__(self, v):
        return jnp.select(
                [v < self.x,   True],
                [v / self.phi, ((v + self.a) / (1 + self.a)) ** self.gamma])

# http://www.brucelindbloom.com/WorkingSpaceInfo.html#Specifications
ColorSpec.S_RGB = ColorSpec(
        Gamma(2.4, 0.055), 0.6400, 0.3300, 0.3000, 0.6000, 0.1500, 0.0600)

# https://www.researchgate.net/publication/318152296
M_16 = jnp.asarray([[ 0.401288, 0.650173, -0.051461],
                    [-0.250268, 1.204414,  0.045854],
                    [-0.002079, 0.048952,  0.953127]])

M_AB = jnp.asarray([[1,      -12. / 11,  1. / 11],
                    [1. / 9,  1.  / 9,  -2. / 9 ]])

CIECAM02Surround = namedtuple("CIECAM02Surround", ["f", "c", "n_c"])
CIECAM02Surround.AVERAGE = CIECAM02Surround(1.0, 0.69,  1.0)
CIECAM02Surround.DIM     = CIECAM02Surround(0.9, 0.59,  0.9)
CIECAM02Surround.DARK    = CIECAM02Surround(0.8, 0.525, 0.8)

h_i = jnp.asarray([20.14,  90.00, 164.25, 237.53, 380.14]) * jnp.pi / 180.
e_i = jnp.asarray([ 0.8,    0.7,    1.0,    1.2,    0.8])
achromatic = jnp.asarray([2, 1, 1. / 20])
t_norm = jnp.asarray([1, 1, 21. / 20])

JChQMsH = namedtuple(
        "JChQMsH", ["j", "c", "h", "q", "m", "s", "hq", "h_"], defaults=[None])
Jab = namedtuple("Jab", ["j", "a", "b"])

class ViewingConditions:
    # https://en.wikipedia.org/wiki/SRGB#Viewing_environment
    y_b = 0.2 # luminance factor of the background
    e_w = 64 # illuminance of reference white in lux
    spec = ColorSpec.S_RGB
    white = WhitePoint.D65_2DEG
    surroundings = CIECAM02Surround.AVERAGE
    def __init__(
            self, y_b=None, e_w=None, spec=None, white=None, surroundings=None):
        self.y_b = self.y_b if y_b is None else y_b
        self.e_w = self.e_w if e_w is None else e_w
        self.spec = self.spec if spec is None else spec
        self.white = self.white if white is None else white
        self.surroundings = self.surroundings if surroundings is None \
                else surroundings

        self.l_a = self.e_w * self.y_b / (jnp.pi * self.white.y_w)
        self.xyz_w = jnp.asarray(self.white)[:, None] / self.white.y_w
        self.rgb_w = M_16 @ self.xyz_w

        self.d = self.surroundings.f
        self.d *= (1 - (1/3.6) * jnp.exp((-self.l_a - 42) / 92))
        self.d = jnp.clip(self.d, 0, 1)

        self.d_rgb = self.d / self.rgb_w + 1 - self.d
        self.k = 1 / (5 * self.l_a + 1)
        self.f_l = (
                0.2 * self.k ** 4 * (5 * self.l_a) +
                0.1 * (1 - self.k ** 4) ** 2 * (5 * self.l_a) ** (1./3))
        self.f_l_4 = self.f_l ** 0.25
        self.n = self.y_b / self.white.y_w
        self.z = 1.48 + jnp.sqrt(self.n)
        self.n_bb = 0.725 * (1 / self.n) ** 0.2
        self.n_cb = self.n_bb

        self.rgb_wc = self.d_rgb * self.rgb_w
        rescaled = (self.f_l * self.rgb_wc) ** 0.42
        self.rgb_aw = 400 * (rescaled / (rescaled + 27.13)) + 0.1
        self.a_w = ((achromatic @ self.rgb_aw - 0.305) * self.n_bb)

    # http://brucelindbloom.com/Eqn_RGB_XYZ_Matrix.html
    _m_rgb_xyz = None
    @property
    def m_rgb_xyz(self):
        if self._m_rgb_xyz is None:
            x = jnp.asarray([self.spec.x_r, self.spec.x_g, self.spec.x_b])
            y = jnp.asarray([self.spec.y_r, self.spec.y_g, self.spec.y_b])
            xyz = jnp.asarray((x / y, jnp.ones(3), (1 - x - y) / y))
            s = jnp.linalg.inv(xyz) @ self.xyz_w
            self._m_rgb_xyz = xyz * s.transpose()
        return self._m_rgb_xyz

    def cam16(self, rgb):
        assert rgb.ndim == 2 and rgb.shape[0] == 3
        cones = M_16 @ self.m_rgb_xyz @ self.spec.gamma(rgb)
        rgb_c = self.d_rgb * cones
        sign = jnp.sign(rgb_c)
        rescaled = (sign * self.f_l * rgb_c) ** 0.42
        rgb_a = sign * 400 * (rescaled / (rescaled + 27.13)) + 0.1
        a, b = M_AB @ rgb_a
        h_rad = jnp.arctan2(b, a) % (2 * jnp.pi)
        hprime = jnp.select([h_rad < h_i[0], True], [h_rad + 2 * jnp.pi, h_rad])
        i = jnp.searchsorted(h_i, hprime, side="right") - 1
        e_t = (jnp.cos(h_rad + 2) + 3.8) / 4
        endpoint = lambda n: (hprime - h_i[i + n]) / e_i[i + n]
        h = 100 * (i + endpoint(0) / (endpoint(0) - endpoint(1)))

        a_ = ((achromatic @ rgb_a - 0.305) * self.n_bb)
        assert not jnp.any(a_ < 0)

        j = 100 * (a_ / self.a_w) ** (self.surroundings.c * self.z)
        q = (
                (4 / self.surroundings.c) * (j / 100) ** 0.5 *
                (self.a_w + 4) * self.f_l_4)
        e = (50_000. / 13) * self.surroundings.n_c * self.n_cb * e_t
        t = e * jnp.sqrt(a ** 2 + b ** 2) / (t_norm @ rgb_a)

        c = t ** 0.9 * (j / 100) ** 0.5 * (1.64 - 0.29 ** self.n) ** 0.73
        m = c * self.f_l_4
        s = 100 * (m / q) ** 0.5

        return JChQMsH(j, c, h_rad * 180 / jnp.pi, q, m, s, h, h_rad)

    # d rgb (axis 0) / d jch (axis 1)
    # TODO: jax.jacobian instead?
    _partial = None
    @property
    def partial(self):
        if self._partial is None:
            def closure(var):
                def getter(rgb):
                    assert rgb.size == 3
                    return getattr(self.cam16(rgb), var)[0]
                return jax.grad(getter)
            partials = tuple(map(closure, ("j", "c", "h")))
            def mapping(rgb):
                return jnp.concat([i(rgb) for i in partials], axis=1)
            self._partial = mapping
        return self._partial

    def ucs(self, jch):
        j = 1.7 * jch.j / (1 + 0.007 * jch.j)
        m = jnp.log(1 + 0.0228 * jch.m) / 0.0228
        a = m * jnp.cos(jch.h_)
        b = m * jnp.sin(jch.h_)
        return Jab(j, a, b)

    def delta(self, jab0, jab1):
        x, y = map(jnp.asarray, (jab0, jab1))
        eprime = jnp.sqrt(jnp.sum((x - y) ** 2))
        return 1.41 * eprime ** 0.63

    def bound(self, rgb):
        oob = jnp.any((rgb < 0) + (rgb > 1), axis=0)
        clipped = jnp.clip(rgb, 0, 1)

