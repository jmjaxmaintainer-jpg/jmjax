# ==============================================================================
# Minimal reproduction of the RW2 scan failure, with NO jmjax involved.
#
# Run it with the venv's own python, from a terminal:
#
#   ~/.virtualenvs/r-jmjax/bin/python dev/repro_scan_float0.py
#
# WHY NOT THROUGH R. reticulate::py_last_error() returned NULL, so the
# Python traceback never reached us. Running Python directly means Python
# prints its own traceback natively and nothing can swallow it.
#
# WHAT IT ISOLATES. jmjax's penalized spline prior builds the RW2
# recursion with
#
#   numpyro_scan(rw2_step, (W01[0], W01[1]), None, length = n_splines - 2)
#
# a two-scalar carry, xs=None, explicit length. Variant 1 below is that
# and nothing else - nine coefficients, a trivial likelihood, five warm-up
# draws. If it fails with the same
#
#   TypeError: body_fun output and input must have identical types
#   ... int32[] weak_type vs float0[]
#
# then the bug is in numpyro/jax and jmjax is only the messenger. If it
# passes, the trigger is something about how jmjax uses it and the model
# has to be built up until it breaks.
#
# THE OTHER VARIANTS ARE CANDIDATE ONE-LINE FIXES, so a failure here comes
# with a remedy attached rather than just a diagnosis:
#
#   2  xs = zeros(K) instead of xs=None + length=K. numpyro's scan takes a
#      different path when it has to infer the length, and that path is
#      where a loop counter would be threaded through the carry.
#   3  the vectorized closed form jmjax already implements - the known
#      working configuration, included as a control so a clean run proves
#      the harness itself is sound.
#   4  scan under a plain jax.grad of the log density, no NUTS. Separates
#      "automatic differentiation through scan is broken" from "NUTS's use
#      of it is broken", which point at different fixes.
# ==============================================================================

import sys, traceback

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan as numpyro_scan
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import initialize_model

print("================ versions ================")
print(f"  python    {sys.version.split()[0]}")
for m in ("jax", "jaxlib", "numpyro", "ml_dtypes", "numpy"):
    try:
        print(f"  {m:9s} {__import__(m).__version__}")
    except Exception as e:                       # noqa: BLE001
        print(f"  {m:9s} ? ({e})")
print(f"  x64 enabled: {jax.config.jax_enable_x64}")
print(f"  platform   : {jax.default_backend()}")

K = 9          # spline coefficients, as with n_interior_knots = 5, ord = 4
N_REST = K - 2
Y = jnp.array(1.0)


def _prior():
    """The shared head: smoothing precision and the two unpenalized coefficients."""
    tau_w = numpyro.sample("tau_w", dist.Gamma(1.0, 0.005))
    sigma_w = 1.0 / jnp.sqrt(tau_w)
    W01 = numpyro.sample("W01", dist.Normal(0.0, 10.0).expand([2]))
    return sigma_w, W01


def model_scan_length(y):
    """Variant 1 - exactly what jmjax does today."""
    sigma_w, W01 = _prior()

    def rw2_step(carry, _):
        w_prev2, w_prev1 = carry
        z_step = numpyro.sample("z_step", dist.Normal(0.0, 1.0))
        w_next = 2.0 * w_prev1 - w_prev2 + sigma_w * z_step
        return (w_prev1, w_next), w_next

    _, w_rest = numpyro_scan(rw2_step, (W01[0], W01[1]), None, length=N_REST)
    W = jnp.concatenate([W01, w_rest])
    numpyro.sample("obs", dist.Normal(W.sum(), 1.0), obs=y)


def model_scan_xs(y):
    """Variant 2 - same recursion, but scan is given a real xs to iterate over."""
    sigma_w, W01 = _prior()

    def rw2_step(carry, _):
        w_prev2, w_prev1 = carry
        z_step = numpyro.sample("z_step", dist.Normal(0.0, 1.0))
        w_next = 2.0 * w_prev1 - w_prev2 + sigma_w * z_step
        return (w_prev1, w_next), w_next

    _, w_rest = numpyro_scan(rw2_step, (W01[0], W01[1]), jnp.zeros(N_REST))
    W = jnp.concatenate([W01, w_rest])
    numpyro.sample("obs", dist.Normal(W.sum(), 1.0), obs=y)


def model_vectorized(y):
    """Variant 3 - jmjax's closed form. The known-good control."""
    sigma_w, W01 = _prior()
    z_step = numpyro.sample("z_step", dist.Normal(0.0, 1.0).expand([N_REST]))
    s2 = jnp.cumsum(jnp.cumsum(z_step))
    k_offset = jnp.arange(1, N_REST + 1)
    w_rest = W01[1] + k_offset * (W01[1] - W01[0]) + sigma_w * s2
    W = jnp.concatenate([W01, w_rest])
    numpyro.sample("obs", dist.Normal(W.sum(), 1.0), obs=y)


def run_nuts(label, model):
    print(f"\n---- {label} ----")
    try:
        mcmc = MCMC(NUTS(model), num_warmup=5, num_samples=5,
                    num_chains=1, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(0), y=Y)
        s = mcmc.get_samples()
        print(f"  OK - sites: {sorted(s.keys())}")
        return True
    except Exception:                             # noqa: BLE001
        print("  FAILED\n")
        traceback.print_exc()
        return False


def run_grad(label, model):
    """Variant 4 - differentiate the log density directly. No sampler."""
    print(f"\n---- {label} ----")
    try:
        init = initialize_model(jax.random.PRNGKey(0), model, model_kwargs={"y": Y})
        potential_fn = init[1]
        z = init[0].z
        val, grad = jax.value_and_grad(potential_fn)(z)
        finite = all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree_util.tree_leaves(grad))
        print(f"  OK - potential {float(val):.4f}, gradients finite: {finite}")
        return True
    except Exception:                             # noqa: BLE001
        print("  FAILED\n")
        traceback.print_exc()
        return False


print("\n================ variants ================")
ok1 = run_nuts("1  scan, xs=None + length=K   (what jmjax does)", model_scan_length)
ok2 = run_nuts("2  scan, xs=zeros(K)          (candidate fix)", model_scan_xs)
ok3 = run_nuts("3  vectorized closed form     (known-good control)", model_vectorized)
ok4 = run_grad("4  scan under plain jax.grad, no NUTS", model_scan_length)

print("\n\n================ VERDICT ================")
if not ok1 and ok3:
    print("  Reproduced with no jmjax in the picture, so this is numpyro/jax")
    print("  on this platform, not a jmjax regression. The traceback above")
    print("  names the function.")
    if ok2:
        print("\n  AND variant 2 passed: giving scan a real xs instead of")
        print("  xs=None + length= avoids it. That is a one-line change in")
        print("  mcmc_model.py and keeps the scan implementation available")
        print("  rather than retiring it.")
    else:
        print("\n  Variant 2 failed too, so the length= form is not the")
        print("  trigger and scan is unusable here. Default should move to")
        print("  the vectorized closed form.")
    if ok4 and not ok1:
        print("\n  Variant 4 passed while 1 failed: differentiating through")
        print("  scan is fine and the mismatch is introduced by NUTS's")
        print("  machinery around it, not by AD itself.")
    elif not ok4:
        print("\n  Variant 4 failed too: AD through numpyro's scan is broken")
        print("  here outright, independent of any sampler.")
elif ok1:
    print("  Variant 1 PASSED here, so a bare RW2 scan is fine and the")
    print("  trigger is something jmjax adds around it - the subject plate,")
    print("  the survival likelihood, or the initialisation. Next step is to")
    print("  grow this model toward jmjax's until it breaks.")
else:
    print("  Even the vectorized control failed, so the harness or the")
    print("  install is the problem rather than scan. Check the versions")
    print("  printed at the top before reading anything else into this.")
