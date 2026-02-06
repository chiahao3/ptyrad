# Refactor plans

## Notes

### 20260206
- Rebuild logger logic and improve vprint
- consider absorbing set_device and set_accelerator into PtyRAD solver if possible
- More cleaning on the utils, especially math, physics, image_proc
- clean up io/
- check the import chain again, now that no utils/ module is promoted, everyone is going through absolute address
- eventually initialization may need its own subpackage, it's almost 2000 lines now
- modularize models.py and promote it to subpackage
- Conceptually it might make sense to promote solver into a package as well, and then we can do
  - cli
  - solver
    - base (PtyRADSolver)
    - initialization
    - hypertune
    - reconstruction
    - visualization
  - core
    - models
      - base
      - probe
      - object
      - position
      - propagator
    - losses
    - constraints
    - forward
  - params
  - io
  - optics (or maybe physics)
  - utils