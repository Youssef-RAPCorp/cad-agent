"""cad_agent3.operations — fine-grained CAD operations.

Importing this package eagerly imports all operation modules across
features/, selectors/, analysis/, repair/, transforms/, booleans/.
Each module calls @register at import time, populating the catalog.

Public API:
    catalog.register, catalog.get, catalog.all_names, catalog.summarize
    Operation, OperationDecl, OperationResult, OperationCheck
"""
from .operation_base import (
    Operation, OperationDecl, OperationResult, OperationCheck)
from . import catalog

# Trigger registration of every operation
from . import features
from . import selectors
from . import analysis
from . import repair
from . import transforms
from . import booleans

__all__ = [
    "Operation", "OperationDecl", "OperationResult", "OperationCheck",
    "catalog",
]
