import os

# Enable genuine parallel MCMC chains on CPU. By default, JAX only exposes
# ONE device regardless of how many CPU cores are actually available -
# numpyro.set_host_device_count() tells XLA to treat the CPU as N virtual
# devices instead, which numpyro's MCMC(..., num_chains=N) then runs
# chains across in parallel rather than falling back to sequential
# execution (the "not enough devices... chains will be drawn sequentially"
# warning users would otherwise see every time). This MUST happen before
# XLA's backend actually initializes (its first real computation) - it's
# safe here even though `import numpyro`/`import jax` happen next, since
# those imports alone don't trigger full backend initialization, only the
# first actual computation does.
#
# Configurable via the JMJAX_NUM_DEVICES environment variable (set from R
# via jmjax_setup(num_devices=...) or Sys.setenv() before the first
# jm_fit() call - it has no effect once the backend has already been
# imported in this session). Defaults to the number of CPU cores detected
# here if not explicitly set.
_num_devices = int(os.environ.get("JMJAX_NUM_DEVICES", os.cpu_count() or 4))

# ---------------------------------------------------------------------------
# Floating-point precision. DEFAULTS TO float64, unlike JAX itself.
#
# JAX defaults to float32 and silently DOWNCASTS any float64 array handed to
# it. That default suits neural networks; it is wrong for this package, for
# three reasons measured on real fits rather than assumed:
#
#   1. The maximum-likelihood path is the worst affected, and least
#      visibly. common.py hands scipy's L-BFGS-B a float64 gradient buffer
#      (`np.array(grad, dtype=np.float64)`), but under float32 the VALUES in
#      it carry only ~1e-7 relative accuracy while scipy's default
#      tolerances assume ~1e-15. The Hessian is computed the same way and
#      then INVERTED for the covariance matrix, which is where single
#      precision hurts most - so the reported standard errors are a
#      float64-shaped container holding float32-quality numbers.
#
#   2. NUTS pays for gradient noise in leapfrog steps. At n = 8,000,
#      float64 was 4.24x FASTER end-to-end despite doing more work per
#      operation, because float32 noise drove the sampler into roughly four
#      times as many steps. ESS for alpha went from 4 to 1188.
#
#   3. exp() overflows float32 at ~88 and float64 at ~709. The hazard is
#      exp(linear predictor), and NUTS explores wild regions during warmup,
#      so 88 is reachable on ordinary data.
#
# R has no single-precision numeric type - `numeric` IS a double - so
# float64 is also what a user calling this from R already assumes they are
# getting.
#
# Set JMJAX_ENABLE_X64=0 to opt out (from R: jmjax_setup(enable_x64 =
# FALSE)). Like JMJAX_NUM_DEVICES this MUST be set before the backend is
# first imported in a session; it cannot be changed afterwards without
# restarting R.
#
# Set through the environment BEFORE `import numpyro` (which imports jax),
# because that is the only point at which jax reads it for free.
# ---------------------------------------------------------------------------
_x64_requested = os.environ.get("JMJAX_ENABLE_X64", "1").strip().lower() \
    not in ("0", "false", "no", "off")
os.environ["JAX_ENABLE_X64"] = "1" if _x64_requested else "0"

import numpyro
numpyro.set_host_device_count(_num_devices)

import jax

# Belt and braces. The environment variable above is authoritative only if
# nothing else in this Python session imported jax first - another
# reticulate-using package could have. config.update() still works in that
# case, so state the intent explicitly rather than inheriting someone
# else's default.
jax.config.update("jax_enable_x64", _x64_requested)


def get_precision():
    """Return the precision actually in force: 'float64' or 'float32'.

    Read live from jax rather than from the request above, so that a
    disagreement between what was asked for and what is running is
    visible to the caller instead of being reported as the request.
    """
    return "float64" if jax.config.jax_enable_x64 else "float32"

from . import weibull_model
from . import spline_model
from . import mcmc_model

__all__ = ["weibull_model", "spline_model", "mcmc_model", "get_precision"]
