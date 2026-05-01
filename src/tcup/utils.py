import jax
import jax.numpy as jnp
import jax.scipy.special as jspec

_SIGMA_68_CDF = 0.5 * (1 + jspec.erf(1 / jnp.sqrt(2)))


@jax.jit
def peak_height(nu):
    log_t = 0.5 * jnp.log(2)
    log_t -= 0.5 * jnp.log(nu)
    log_t += jspec.gammaln((nu + 1) / 2)
    log_t -= jspec.gammaln(nu / 2)
    t = jnp.where(nu == 0.0, 0.0, jnp.exp(log_t))
    t = jnp.where(jnp.isinf(nu), 1.0, t)
    return t


@jax.jit
def normality(nu):
    t = peak_height(nu)
    c = peak_height(1)  # Cauchy peak height
    return (t - c) / (1 - c)


@jax.jit
def t_cdf(nu, x):
    # Using https://encyclopediaofmath.org/wiki/Student_distribution
    # The above has a source for this in terms of the incomplete Beta function
    # It's only for x > 0 so I modified it to work for all x
    # If x < 0, return integral
    # If x == 0, return 1/2
    # If x > 0, return 1 - integral
    integral = 0.5 * jspec.betainc(0.5 * nu, 0.5, nu / (nu + x**2))
    return (jnp.sign(x) + 1) / 2 - jnp.sign(x) * integral


@jax.jit
def outlier_frac(nu, outlier_sigma=3):
    normal_outlier_frac = 1 - jspec.erf(outlier_sigma / jnp.sqrt(2))
    omega = jspec.betainc(0.5 * nu, 0.5, nu / (nu + outlier_sigma**2))
    omega = jnp.where(nu == 0, 0.0, omega)
    omega = jnp.where(jnp.isinf(nu), normal_outlier_frac, omega)
    return omega


def _sigma_68_impl(nu):
    nu = jnp.asarray(nu, dtype=jnp.result_type(nu, 1.0))
    high = jnp.maximum(2.0, 20.0 / jnp.power(nu, 3))
    low = jnp.zeros_like(high)

    def body(_, bounds):
        low, high = bounds
        mid = 0.5 * (low + high)
        cdf_mid = t_cdf(nu, mid)
        low = jnp.where(cdf_mid < _SIGMA_68_CDF, mid, low)
        high = jnp.where(cdf_mid < _SIGMA_68_CDF, high, mid)
        return low, high

    low, high = jax.lax.fori_loop(0, 80, body, (low, high))
    sigma = 0.5 * (low + high)
    return jnp.where(jnp.isinf(nu), 1.0, sigma)


@jax.custom_jvp
def sigma_68(nu):
    return _sigma_68_impl(nu)


@sigma_68.defjvp
def _sigma_68_jvp(primals, tangents):
    (nu,) = primals
    (nu_dot,) = tangents
    nu = jnp.asarray(nu, dtype=jnp.result_type(nu, 1.0))
    step = jnp.maximum(1e-4, 1e-4 * nu)
    step = jnp.minimum(step, 0.5 * nu)
    nu_lo = jnp.maximum(nu - step, jnp.finfo(nu.dtype).tiny)
    nu_hi = nu + step
    sigma = _sigma_68_impl(nu)
    grad = (_sigma_68_impl(nu_hi) - _sigma_68_impl(nu_lo)) / (nu_hi - nu_lo)
    return sigma, grad * nu_dot


sigma_68 = jax.jit(sigma_68)
