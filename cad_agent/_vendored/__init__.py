"""Vendored backend implementations.

`cad_agent3` is the original research codebase; we ship it inside
`cad_agent._vendored` so the public API surface stays small and
versioned independently. Don't import from here in user code — use
`cad_agent.advanced` instead.
"""
