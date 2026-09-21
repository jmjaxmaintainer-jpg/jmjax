# Why 79 tests skipped on a fresh machine, and what changed

## Symptom

On a new Mac, `devtools::test()` reported **0 failures, 89 passing, 79 skipped**,
every skip reading `Python backend not available (run jmjax::jmjax_setup() to
enable this test)` — while the `r-jmjax` virtualenv, complete with
jax 0.4.30 / jaxlib / numpyro 0.15.0, had been built **during that same run**.

## Cause

An ordering bug in `.get_backend()` (`R/zzz.R`), not a user-workflow mistake.

reticulate binds to exactly one Python interpreter per R session, at the first
`py_*` call, and cannot be re-pointed afterwards. `use_python(required = TRUE)`
checks `is_python_initialized()` and calls
`stop("failed to initialize requested version of Python")` when a different
interpreter is already attached (reticulate `R/use_python.R`, the
`"another version of Python ('%s') has already been initialized"` branch).
`py_run_string()` is such a call: it has no `ensure_python_initialized()` guard
in the R wrapper, but the C++ side acquires the GIL through
`_initialize_python_and_PyGILState_Ensure()`, which calls
`ensure_python_initialized()` (reticulate `src/python.cpp`).

The old `.get_backend()` probed first and chose second:

1. Fresh machine, no `r-jmjax` venv → `.onLoad()`'s
   `use_virtualenv(required = FALSE)` is skipped, so reticulate has no hint.
2. `reticulate::py_run_string(add_path_code)` — **binds Python** to whatever the
   system offers (Homebrew/system `python3`).
3. `import("jmjax_backend")` fails: `inst/python/jmjax_backend/__init__.py`
   does `import numpyro` at module level, and the bound interpreter has none.
4. `virtualenv_exists()` is FALSE → `jmjax_setup()` runs and **builds the venv**
   — this is the venv that appeared mid-run.
5. `jmjax_setup()` ends with `use_virtualenv(JMJAX_VENV_NAME, required = TRUE)`
   → throws, because step 2 already attached a different interpreter.
6. The error was swallowed by `backend_available()`'s `tryCatch(..., FALSE)`.

Every later call repeated steps 4–6 down the `else` branch, so all 79 skipped
with a message naming the one explanation that was not true.

Restarting R does fix it — but only because the venv now exists, so `.onLoad()`
sets the hint before anything touches Python. The **first** run on any fresh
machine could never have worked.

## Fix

`.get_backend()` now chooses the environment **before** the first `py_*` call:

- `py_available(initialize = FALSE)` — is anything attached yet?
- `py_exe()` — which interpreter *would* reticulate use? (honours
  `RETICULATE_PYTHON` and any earlier `use_*()`; does not start Python)
- `.python_has_backend_deps()` — a **subprocess** `importlib.util.find_spec`
  check on that interpreter. Deliberately not `py_module_available()`, which
  would start Python in this session and recreate the bug. `find_spec` only
  looks on `sys.path`, so it costs milliseconds rather than a jax import.

Then: if that interpreter has jax/numpyro, use it (preserves the documented
support for a user's own conda env — the Windows case). Otherwise create the
venv if absent, **rebuild it if present but missing its dependencies**
(`jmjax_setup()` installs only on create, so a half-finished install used to be
unrecoverable), and select it. Only then is Python touched.

If Python was already attached by something else, the import failure now raises
a message naming the interpreter and saying to restart, instead of an
`ImportError`.

`tests/testthat/helper-python.R` now reports the error text in the skip reason
and caches the result, so the next failure of this kind is visible in the log.

## `Config/reticulate` — checked and rejected

The suggestion to declare the Python dependencies in DESCRIPTION via
`Config/reticulate` would **not** have prevented this, and would not help CI.

reticulate acts on that field from `configure_environment()`
(`R/python-packages.R`), which returns early when:

- Python is not yet initialized — so it can never influence which interpreter
  gets chosen; it runs *after* the binding, from `initialize_python()`;
- the session is **non-interactive**, unless called with `force = TRUE` *and*
  `NOT_CRAN=true` — so it never fires under `R CMD check` or
  `devtools::test()`;
- the attached interpreter is not already a virtualenv or conda env.

And it installs into whatever environment is already bound — which in the
failure above was the system Python, where pip would refuse (externally
managed). It is a convenience for interactive users, not a provisioning
mechanism. `SystemRequirements` documents the pins instead, and CI provisions
explicitly.

## Known edge

If `RETICULATE_PYTHON` points at an interpreter without jax/numpyro, it takes
precedence over the package's `use_virtualenv()` and the import still fails.
reticulate warns about the override, and the error names the interpreter, but
the variable has to be unset by hand.
