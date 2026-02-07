# Refactor plans

## Notes

- Split `ptycho_model.py` into smaller modules for probe.py, obj.py, pos.py, propagator.py
- Split `initializer.py` into smaller modules, it's almost 2000 lines now
- Clean up `io/`, might make more sense to split in half like `visualization/` to basic files and heavy libraries, or do it by file type.
- More cleaning on the `utils/`, especially math, physics, image_proc that's direclty related to reconstruciton
  - since they're mostly individual modules, it might make sense to distribute some of them outside of `utils/`
- Consider absorbing `set_device` and `set_accelerator` into `PtyRADSolver` if possible
- Thoroughly check the import chain, especially the `__init__.py` promotion, and `__all__` specification for API summary.
- Rebuild logger logic and improve vprint
  - Don't use vprint on everything
  - Peripheral functions should be silent, do the logging only inside the solver context
  - Would need to canocalize the logger usage, if possible, split the rank logic with logger, and only decorate the logger if needed

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
  - io (**TODO**)
  - optics (**TODO**, especially need to coordinate with math, physics, image_proc on their levels)
  - params
  - solver
    - ptyrad_solver (**TODO**)
    - hypertune
    - reconstruction
  - starter (demo workspace folder)
  - utils (**TODO**)
    - common
    - dev_tools
    - grouping
    - image_proc
    - logging (**TODO**)
    - math_ops
    - physics
    - provenance
    - timing
  - visualization
    - basic
    - model