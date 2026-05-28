"""Auto-load all operations in this subpackage so @register triggers."""
import os, importlib
for f in os.listdir(os.path.dirname(__file__)):
    if f.endswith('.py') and f != '__init__.py':
        importlib.import_module(f'.{f[:-3]}', package=__name__)
