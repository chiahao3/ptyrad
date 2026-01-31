(reference:params)=

# Params files

*PtyRAD* uses a **single parameters file** to fully configure the reconstruction task. 

These files are called *"params files"* and contains 6 nested dictionaries with a total of more than 100 fields, making *PtyRAD* extremely customizable and flexible for each reconstruciton task.

For example, a complete *PtyRAD* params file includes: 
1. `init_params`       (required)
2. `hypertune_params`  (only needed in hypertune mode)
3. `model_params`      (optional)
4. `loss_params`       (optional)
5. `constraint_params` (optional)
6. `recon_params`      (optional)

These nested dictionaries can be provided by a range of common file formats, including the native `.py`, `.json`, `.toml`, or `.yaml`.

The recommended file format for *PtyRAD* is `.yaml` (**YAML**) for its excellent readability and compatibility.

Although *PtyRAD* contains many customizable options, usually for each reconstruction task a user only needs to specify ~ 15 fields.

For example, a minimal `params.yaml` might look like this:

```yaml
init_params
    probe_kv            : 80
    probe_conv_angle    : 24.9
    probe_defocus       : 0
    meas_Npix           : 128
    pos_N_scan_slow     : 128
    pos_N_scan_fast     : 128
    pos_scan_step_size  : 0.4290
    probe_pmode_max     : 6
    obj_Nlayer          : 6
    obj_slice_thickness : 2
    meas_params         : {'path': 'data/tBL_WSe2/Panel_g-h_Themis/scan_x128_y128.raw'}
recon_params
    NITER               : 200
    SAVE_ITERS          : 10
    output_dir          : 'output/tBL_WSe2/'
```

```{toctree}
:maxdepth: 2
:hidden:

init_p
hypertune_p
model_p
loss_p
constraint_p
recon_p
```