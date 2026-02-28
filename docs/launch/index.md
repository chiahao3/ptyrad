(reference:launch)=

# Launch PtyRAD

<iframe width="560" height="315" src="https://www.youtube.com/embed/DjshvSWDReg?si=tYdxs2Xu-tuGhnCW" title="PtyRAD YouTube Installation" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen style="margin-bottom: 20px; border-radius: 8px;"></iframe>

> 💡 This video was recorded for **Ptychography Summer School at Cornell 2025**, which has additional access to SciServer, and a slightly different demo folder structure.

You can launch *PtyRAD* with many different approaches once it's installed in your Python environment.

For example, you can:
1. Run the tutorial Jupyter notebooks under `ptyrad/notebooks/run_ptyrad.ipynb`
2. Launch with CLI tools from your terminal as simple as `ptyrad run <PARAMS_FILE_PATH>`
3. Use a Slurm job script like `ptyrad/scripts/slurm_run_ptyrad.sub`

For all the launching method, *PtyRAD* supports 2 operation modes:
- reconstruction
- hypertune (hyperparameter tuning)

These operation modes can be executed on CPU, GPU, or even distributed on multiple GPUs for better performance.

**Routine workflow**
1. Acquire experimental or get simulated data
2. Prepare params files
3. Launch *PtyRAD* task
4. Analyze and refine the params file

```{toctree}
:maxdepth: 2
:hidden:

python
cli
slurm
hypertune
multiGPU
```