"""CADAgent — the public facade.

A single object that holds configuration + provides the main API:

    from cad_agent import CADAgent

    agent = CADAgent()                       # uses env credentials
    result = agent.generate(
        "A 6m x 6m SIP residential pod with a south-facing door, "
        "one east window, and a 2% sloped roof."
    )
    print(result.summary())
    # OK: 'A 6m x 6m SIP residential pod with...'
    #   volume: 28.13 m³ (28128440000 mm³)
    #   STEP:   ./cad_output/pod_001.step
    #   STL:    ./cad_output/pod_001.stl

Lower-level entry points (the reasoning loop, the orchestrator, the
operation catalog) are still available via `agent.reasoning_session(...)`
and `cad_agent.advanced.*`.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import traceback
from pathlib import Path
from typing import Any, Optional

from .config import CADAgentConfig
from .results import CADResult


# Vendored implementation backend — imported lazily so that `import
# cad_agent` doesn't pay the build123d import cost until you actually
# call .generate().

def _lazy_import_backend():
    """Import the vendored cad_agent3 package on demand."""
    from ._vendored import cad_agent3
    return cad_agent3


class CADAgent:
    """Generate CAD models from natural-language descriptions.

    Typical use:
        agent = CADAgent()                        # from env credentials
        result = agent.generate("a hex nut M6, 5mm thick")
        result.part.volume                        # → ~470 mm³
        result.step_path                          # → Path('cad_output/...')

    Pass a CADAgentConfig for finer control:
        cfg = CADAgentConfig(
            backend="gemini",
            api_key="...",
            output_dir=Path("/tmp/cad"),
            max_revisions=5,
        )
        agent = CADAgent(cfg)
    """

    def __init__(self, config: Optional[CADAgentConfig] = None):
        self.config = config or CADAgentConfig.from_env()
        # Validate credential early so users find out at construction,
        # not at first .generate() call.
        if self.config.execute_generated_code:
            self.config.resolve_api_key()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # PRIMARY API
    # ---------------------------------------------------------------
    def generate(
        self,
        spec: str,
        name: Optional[str] = None,
        extra_constraints: str = "",
    ) -> CADResult:
        """Generate a CAD model from a natural-language description.

        Args:
            spec: free-form description of the part. Be specific about
                  dimensions (mm or m), features (holes, fillets,
                  patterns), and standards (NEMA17, M3, etc.).
            name: optional artifact name. If omitted, a short hash is
                  used. The output files will be `<name>.step`,
                  `<name>.stl`, `<name>.py`.
            extra_constraints: additional context appended to the LLM
                  prompt. Use for things like "must be 3D-printable
                  without supports" or "drilled holes only, no thread
                  cuts".

        Returns:
            CADResult — see results.py.
        """
        if not name:
            name = "part_" + uuid.uuid4().hex[:8]

        if self.config.verbose:
            print(f"[cad_agent] generate('{spec[:60]}...', name={name!r})", file=sys.stderr)

        t0 = time.time()
        result = CADResult(
            spec=spec,
            success=False,
            output_dir=self.config.output_dir,
        )
        result.metadata["backend"] = self.config.backend
        result.metadata["model"] = self.config.resolve_model()
        result.metadata["name"] = name

        try:
            backend = _lazy_import_backend()

            # Set env vars expected by the vendored backend
            api_key = self.config.resolve_api_key()
            if self.config.backend == "gemini":
                os.environ.setdefault("GEMINI_API_KEY", api_key)
            else:
                os.environ.setdefault("ANTHROPIC_API_KEY", api_key)

            model = self.config.resolve_model()
            if self.config.backend == "gemini":
                os.environ.setdefault("GEMINI_CODEGEN_MODEL", model)

            # --- Optionally route through the reasoning/critic loop ---
            if self.config.use_reasoning_loop and self.config.max_revisions > 0:
                rs_result = self._run_reasoning_loop(backend, spec, extra_constraints)
                if rs_result is not None:
                    final_spec, log = rs_result
                    result.reasoning_log = log
                else:
                    final_spec = spec
            else:
                final_spec = spec

            # --- Generate build123d code ---
            gen = backend.generate_shape(
                final_spec,
                extra_constraints=extra_constraints,
                execute=self.config.execute_generated_code,
            )

            result.script = gen.code or ""
            if gen.part is not None:
                result.part = gen.part
                try:
                    result.volume_mm3 = float(gen.part.volume)
                except Exception:
                    pass

            if not gen.code:
                result.error = (gen.error or "no code generated")
                return result

            # --- Write artifacts ---
            outdir = self.config.output_dir
            if self.config.write_script:
                script_path = outdir / f"{name}.py"
                script_path.write_text(result.script)
                result.script_path = script_path

            if gen.part is not None:
                if self.config.write_step:
                    step_path = outdir / f"{name}.step"
                    gen.save_step(str(step_path))
                    result.step_path = step_path
                if self.config.write_stl:
                    stl_path = outdir / f"{name}.stl"
                    gen.save_stl(str(stl_path))
                    result.stl_path = stl_path

            # Success criterion: we got either a runnable Part (with
            # nonzero volume) OR, if execution was disabled, we got code.
            if self.config.execute_generated_code:
                result.success = (gen.part is not None
                                   and result.volume_mm3 is not None
                                   and result.volume_mm3 > 0)
                if not result.success and not result.error:
                    result.error = "generated part had zero volume or failed to execute"
            else:
                result.success = bool(gen.code)

        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            if self.config.verbose:
                traceback.print_exc(file=sys.stderr)

        result.metadata["elapsed_s"] = time.time() - t0
        return result

    # ---------------------------------------------------------------
    # LOWER-LEVEL ACCESSORS
    # ---------------------------------------------------------------
    def reasoning_session(self):
        """Return a ReasoningSession instance for multi-step design."""
        backend = _lazy_import_backend()
        return backend.ReasoningSession()

    def orchestrator(self):
        """Return an Orchestrator that resolves named standards (NEMA17, M3, …)."""
        backend = _lazy_import_backend()
        return backend.Orchestrator()

    def design_session(self, name: str = "session"):
        """Return a DesignSession with full history, checkpoints, rollback."""
        backend = _lazy_import_backend()
        return backend.DesignSession(name=name)

    def list_known_parts(self) -> dict:
        """List of standard parts in the bundled config library
        (bearings, motors, extrusions, boards). Useful for prompting."""
        backend = _lazy_import_backend()
        return backend.list_available()

    # ---------------------------------------------------------------
    # INTERNALS
    # ---------------------------------------------------------------
    def _run_reasoning_loop(self, backend, spec, extra_constraints):
        """Run the critic-feedback loop to refine the spec before codegen.

        Returns (refined_spec, log) or None on failure.
        """
        try:
            session = backend.ReasoningSession()
            out = session.run(spec, max_iterations=self.config.max_revisions)
            log = []
            for it in getattr(out, "iterations", []) or []:
                log.append(getattr(it, "summary", str(it)))
            refined = getattr(out, "final_spec", None) or spec
            return refined, log
        except Exception:
            # If reasoning fails for any reason, fall back to direct spec.
            return None


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def generate(spec: str, **kwargs) -> CADResult:
    """One-shot helper. Equivalent to `CADAgent().generate(spec, **kwargs)`.

    Use this for quick scripts; instantiate `CADAgent` directly when
    you want to reuse config across multiple calls.
    """
    return CADAgent().generate(spec, **kwargs)
