"""cad_agent.advanced — low-level building blocks.

For most users `CADAgent.generate(...)` is enough. This module exposes
everything else: the reasoning loop primitives, the operation catalog,
session management with checkpoint/rollback, and the
build123d-reconstruction stack (for working backward from a STEP file).

All names below are re-exported from the underlying implementation. Use
when you need to build a custom workflow rather than one-shot generation.

Example — multi-step design with rollback:

    from cad_agent.advanced import DesignSession

    sess = DesignSession(name="bracket")
    sess.apply("create_box", dims=(50, 30, 10))
    cp = sess.checkpoint()
    sess.apply("hole", diameter=5, location=(10, 10, 5))
    if not sess.validate():
        sess.rollback(cp)
"""

from __future__ import annotations

# Lazy import — only loads when something is accessed.
import importlib
import sys


def _load():
    return importlib.import_module("cad_agent._vendored.cad_agent3")


# Eagerly import once on first attribute access so type checkers see
# the surface. This proxies any name to the underlying backend.
class _LazyProxy:
    def __init__(self):
        self._mod = None

    def __getattr__(self, name):
        if self._mod is None:
            self._mod = _load()
        return getattr(self._mod, name)

    def __dir__(self):
        if self._mod is None:
            self._mod = _load()
        return dir(self._mod)


sys.modules[__name__] = _LazyProxy()
