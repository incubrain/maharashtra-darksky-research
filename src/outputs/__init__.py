"""Outputs subpackage — reports, maps, visualizations, and downloads.

Modules are imported on demand rather than eagerly to avoid conflicts
with ``python3 -m src.outputs.download_viirs`` and similar entry points
(runpy expects the module not to be in sys.modules before execution).
"""
