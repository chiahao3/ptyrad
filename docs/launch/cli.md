# Command-line Interface (CLI)

*PtyRAD* also provides a command-line interface to execute common commands from your terminal once you installed `ptyrad` in your Python environment.

**Run reconstructions**
```bash
# This is used to quickly launch a reconstruction / hypertune task
ptyrad run --params_path params/tBL_WSe2_reconstruct.yaml
```

**Check GPU compatibility and PyTorch build**

```bash
ptyrad check-gpu
```

**Full list of hardware and package version information**
```bash
ptyrad print-system-info
```

**Export PtyRAD preprocessed measurements data**
```bash
# Exporting measurements data for easy visualization and analysis
ptyrad export-meas --params_path params/tBL_WSe2_reconstruct.yaml --output data/ptyrad_init_meas.hdf5 --reshape 128 128 128 128 --append
```