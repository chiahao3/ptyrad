# Refactor plans

## Notes

- Split `ptycho_model.py` into smaller modules for probe.py, obj.py, pos.py, propagator.py
- Split `initializer.py` into smaller modules, it's almost 2000 lines now
- Rebuild `logger` logic and improve `vprint`
  - Don't use vprint on everything
  - Peripheral functions should be silent, do the logging only inside the solver context
  - Would need to canocalize the logger usage, if possible, split the rank logic with logger, and only decorate the logger if needed
- Thoroughly check the import chain, especially the `__init__.py` promotion, and `__all__` specification for API summary.

## Architecture plan
  - cli
  - core
    - models (**TODO**)
      - ptycho_model
      - base (blue print)
      - probe
      - object
      - position
      - propagator
  - init
    - initializer (**TODO**)
  - io
  - optics
    - aberrations
    - constants
    - probe
    - propagator
  - params
  - runtime
    - device
    - diagnostics
    - logging (**TODO**)
    - seed
  - solver
    - ptyrad_solver
    - hypertune
    - reconstruction
    - grouping
  - starter (demo workspace folder)
  - utils
    - dev_tools
    - image_proc
    - affine
    - time
  - plotting
    - basic
    - model