"""Site/city analysis subpackage.

Modules are imported on demand rather than eagerly to avoid conflicts
with ``python3 -m src.site.site_analysis`` (runpy expects the module
not to be in sys.modules before execution).
"""
