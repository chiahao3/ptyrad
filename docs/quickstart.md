# Quickstart Demo
   
Here, we provide a quick walkthough of how to get started with *PtyRAD*.

Before running the demo reconstruction, please check the following:

**Checklist**
1. Install a modern code editor (e.g. [VS Code](https://code.visualstudio.com/))
2. Install a Python environment / package manager (e.g. [miniforge](https://conda-forge.org/download/))
3. Create a dedicated python environemnt `(ptyrad)` for *PtyRAD* and activate it in the terminal
4. Install [*PtyRAD*](https://ptyrad.readthedocs.io/en/latest/installation.html) into the `(ptyrad)` environment
5. Run `ptyrad init` to create a starter kit folder structure
6. Run `python ./scripts/download_demo_data.py` to automatically download the demo data to the correct location `data/`

Once you complete the checklist, you're now ready to run a quick demo using one of two interfaces:

**1. Interactive Jupyter interface (Recommended)**

Use `ptyrad/tutorials/run_ptyrad.ipynb` to quickly reconstruct the demo dataset in a Jupyter notebook

**2. Command-line interface**

```bash
# Here we assume "params/" is under the working directory
ptyrad run "params/examples/tBL_WSe2.yaml" --gpuid 0
```