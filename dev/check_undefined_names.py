"""Report names a function loads but never binds.

This is the NameError class of bug that `python -m py_compile` cannot catch:
the file compiles fine, and the error only appears when that line runs. A
real instance shipped in this package - `orthogonalize_report` was computed
in fit_nuts() but read in _package_result(), which takes it as a parameter -
and it broke every MCMC fit until it was caught by running one.

Approximate by design, and deliberately biased toward silence: closure
variables, comprehension targets and star-imports are all treated as bound,
so it under-reports rather than crying wolf. A clean run is weak evidence;
a dirty run is strong evidence.

  python3 dev/check_undefined_names.py inst/python/jmjax_backend/mcmc_model.py
"""
import ast
import builtins
import sys

BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
SCOPED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _children_skipping_scopes(node):
    """Descendants in THIS scope: yields nested defs (so their names count
    as bindings, including defs inside an if/try block) but never descends
    into their bodies."""
    for child in ast.iter_child_nodes(node):
        yield child
        if not isinstance(child, SCOPED):
            yield from _children_skipping_scopes(child)


def bound_in(node):
    """Names this scope binds: params, assignments, imports, nested defs."""
    out = set()
    args = getattr(node, "args", None)
    if isinstance(args, ast.arguments):
        for grp in (getattr(args, "posonlyargs", []), args.args, args.kwonlyargs):
            out.update(a.arg for a in grp)
        for a in (args.vararg, args.kwarg):
            if a:
                out.add(a.arg)
    for child in ast.iter_child_nodes(node):
        if isinstance(child, SCOPED):
            out.add(getattr(child, "name", ""))     # the def's own name
            continue
        for n in [child, *_children_skipping_scopes(child)]:
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                out.add(n.id)
            elif isinstance(n, ast.alias):
                out.add((n.asname or n.name).split(".")[0])
            elif isinstance(n, ast.ExceptHandler) and n.name:
                out.add(n.name)
            elif isinstance(n, (ast.Global, ast.Nonlocal)):
                out.update(n.names)
            elif isinstance(n, SCOPED):
                out.add(getattr(n, "name", ""))
    out.discard("")
    return out


def direct_scopes(node):
    """Nested scopes reachable without crossing another scope boundary."""
    body = node.body if isinstance(node.body, list) else [node.body]
    for stmt in body:
        if isinstance(stmt, SCOPED):
            yield stmt
            continue
        for n in _children_skipping_scopes(stmt):
            if isinstance(n, SCOPED):
                yield n


def loads_in(node):
    """Name loads occurring directly in this scope (not nested scopes)."""
    body = node.body if isinstance(node.body, list) else [node.body]
    for stmt in body:
        if isinstance(stmt, SCOPED):
            continue          # a nested def is checked in its own scope
        for n in [stmt, *_children_skipping_scopes(stmt)]:
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                yield n.id, n.lineno
    # default values and decorators evaluate in the ENCLOSING scope, but
    # attributing them here only risks silence, not false alarms.


def scope_name(node, parent):
    base = getattr(node, "name", "<lambda>")
    return f"{parent}.{base}" if parent else base


def check(node, enclosing, parent, found):
    scope = enclosing | bound_in(node)
    here = scope_name(node, parent)
    for name, line in loads_in(node):
        if name not in scope and name not in BUILTINS:
            found.append((line, here, name))
    for child in direct_scopes(node):
        check(child, scope, here, found)


def main(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    module_scope = bound_in(tree) | BUILTINS
    found = []
    for node in tree.body:
        if isinstance(node, SCOPED):
            check(node, module_scope, "", found)
    seen, rows = set(), []
    for line, where, name in sorted(found):
        if (where, name) in seen:
            continue
        seen.add((where, name))
        rows.append("  line %-6d %-26s loads undefined name: %s" % (line, where + "()", name))
    print("\n".join(rows) if rows else "  none")
    print("%d distinct undefined name(s) in %s" % (len(seen), path))
    return 1 if seen else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
