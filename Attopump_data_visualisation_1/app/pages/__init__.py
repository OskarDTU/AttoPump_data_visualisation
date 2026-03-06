"""Streamlit page entry points.

Each module exposes a ``main()`` function that is invoked lazily by the
root ``streamlit_app.py`` via ``st.navigation``.

Pages
-----
- ``test_overview`` — repository-wide overview of resolved classifications.
- ``explorer`` — Single Test Explorer (default landing page).
- ``analysis`` — Comprehensive Analysis (multi-test and multi-pump
  comparison).
- ``manage_groups`` — Manage Groups (CRUD for pumps, shipments, and
  test groups).
- ``report_builder`` — Report Builder (compose & export report packages).
"""
