"""The M10 study catalogue (SPEC §10.1, owner's M10 Part 2): one runner module per study
(``s1_forward_vol`` … ``s7_hedging``), each paired with a YAML config under
``configs/studies/catalogue/`` (``<id>.yaml`` and the CI-fast ``<id>_fast.yaml``) and run by
``volsto-study run configs/studies/catalogue/<id>.yaml`` (:mod:`volsto.studies.runner`, whose
docstring states the module protocol and the config schema; ``configs/studies/catalogue/README.md``
restates the schema).  The package itself holds no code.
"""
