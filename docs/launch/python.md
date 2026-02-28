# Python Interface

*PtyRAD* provides an easy Python interface for programmatically control of  `ptyrad` in your Python environment.

## Run reconstructions

A simple python script for launching *PtyRAD* in **"reconstruction mode"**, which is fully configured by the params file.

```python
from ptyrad.params import load_params
from ptyrad.runtime.device import set_gpu_device
from ptyrad.runtime.diagnostics import print_system_info
from ptyrad.runtime.logging import LoggingManager
from ptyrad.solver import PtyRADSolver

LoggingManager(log_file='ptyrad_log.txt', log_dir='auto', prefix_time='datetime', show_timestamp=True)

params_path = "params/examples/tBL_WSe2.yaml"

print_system_info()

params = load_params(params_path, validate=True)
device = set_gpu_device(gpuid=0) # Pass in `gpuid = None` if you don't have access to a CUDA-compatible GPU. Note that running PtyRAD with CPU would be much slower than on GPU.

ptycho_solver = PtyRADSolver(params, device=device)
ptycho_solver.run()
```

> 💡 This is the same example as `ptyrad/notebooks/run_ptyrad.ipynb`.