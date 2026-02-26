# Command-line Interface (CLI)

*PtyRAD* also provides a command-line interface to execute common commands from your terminal once you installed `ptyrad` in your Python environment.

**Initialize a PtyRAD working directory**
```bash
# This will create a `ptyrad/` directory,
# which contains `data/`, `notebooks/`, `output/`, `params/`, and `scripts/`
ptyrad init
```

**Run reconstructions**
```bash
# This is used to quickly launch a reconstruction / hypertune task
ptyrad run "params/examples/tBL_WSe2.yaml"
```

**Check GPU compatibility and PyTorch build**
```bash
ptyrad check-gpu
```

**Full list of hardware and package version information**
```bash
ptyrad print-system-info
```

**Validate the PtyRAD params file**
```bash
ptyrad validate-params "params/examples/tBL_WSe2.yaml"
```

**Export PtyRAD preprocessed measurements data**
```bash
# Exporting measurements data for easy visualization and analysis
ptyrad export-meas "params/examples/tBL_WSe2.yaml" --output data/ptyrad_init_meas.hdf5 --reshape 128 128 128 128 --append
```