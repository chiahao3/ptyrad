# Python Interface

*PtyRAD* provides an easy Python interface for programmatically control of  `ptyrad` in your Python environment.

## Run reconstructions

A simple python script for launching *PtyRAD* in **"reconstruction mode"**, which is fully configured by the params file.

```python
from ptyrad.params import load_params
from ptyrad.solver import PtyRADSolver
from ptyrad.utils import print_system_info, set_gpu_device, CustomLogger

logger = CustomLogger(log_file='ptyrad_log.txt', log_dir='auto', prefix_time='datetime', show_timestamp=True)

params_path = "params/tBL_WSe2_reconstruct.yml"

print_system_info()
params = load_params(params_path)
device = set_gpu_device(gpuid=0)

ptycho_solver = PtyRADSolver(params, device=device, logger=logger)

ptycho_solver.run()
```

> 💡 This is the same example as `ptyrad/demo/scripts/run_ptyrad_quick_example.ipynb`.

If you want to see more internal working mechanism of *PtyRAD*, the `ptyrad/demo/scripts/run_ptyrad_detailed_walkthrough.ipynb` would be a good example.