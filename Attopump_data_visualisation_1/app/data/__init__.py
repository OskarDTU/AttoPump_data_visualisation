"""Data loading, processing, configuration, and persistence.

Modules
-------
- ``config``         — constants, regex patterns, column heuristics, SweepSpec.
- ``io_local``       — local OneDrive folder discovery and CSV reading.
- ``experiment_log`` — shared experiment-log lookup and parsing.
- ``data_processor`` — cleaning, transforming, test-type detection, binning.
- ``test_catalog``   — resolved per-test records for overview/search/defaults.
- ``bar_groups``     — Bar / Shipment persistence (CRUD + JSON storage).
"""
