(reference:quickstart)=

# Quickstart
   
Here, we provide a quick walkthough of how to get started with *PtyRAD*.

## Recommended Tools
We recommend using [*Miniforge*](https://github.com/conda-forge/miniforge) for Python environment management, and  
[*Visual Studio Code*](https://code.visualstudio.com/Download) for code editing and execution.

## Step-by-step guide

### 1. Create and Activate a clean Python Environment

First, create and activate a new conda environment **(ptyrad)** with Python > 3.10:

```sh
conda create -n ptyrad python=3.12
conda activate ptyrad
```

> 💡 **Note:** After activating the environment, your terminal prompt should show **(ptyrad)** at the beginning, indicating that the environment is active.

### 2. Install *PtyRAD* into the `(ptyrad)` Environment

Then install *PtyRAD* in the activated `(ptyrad)` environment using:

```sh
pip install ptyrad
``` 

If you're using Windows with NVIDIA CUDA GPU, you will also need to install the GPU version of PyTorch with:

```sh
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
```

*PtyRAD* can also be installed via `conda`. For detailed instructions on installing *PtyRAD* on different machines or pinning specific CUDA versions, see [the installation guide](https://ptyrad.readthedocs.io/en/latest/installation.html).

### 3. Create a *Starter Kit* Folder Structure

This starter kit folder `ptyrad/` contains examples and templates params files, useful scripts and notebooks.

```sh
ptyrad init
```


```text
# Folder structure

ptyrad/
├── data/             # Default directory for storing your 4D-STEM datasets
├── notebooks/        # Jupyter notebooks for common workflows and interactive analyses
├── output/           # Default directory where reconstruction results are saved
├── params/
│   ├── examples/     # Ready-to-run parameter files for included demo datasets (e.g., tBL_WSe2, PSO)
│   ├── templates/    # Templates ranging from minimal setups to full API reference
│   └── walkthrough/  # Tutorial-driven parameter files designed to guide you through specific features (e.g., multislice, advanced constraints, and hyperparameter tuning)
└── scripts/          # Utility scripts for fetching demo data and submitting batch jobs on computing clusters
```

### 4. Download the Demo Data

We provide a helper script to automatically fetch the example datasets, and place it in the correct `ptyrad/data/` folder:

```sh
cd ptyrad
python ./scripts/download_demo_data.py
```

After downloading and unzipping, the folder structure should look like this:
```text
# Folder structure

ptyrad/
├── data/ 
│   ├── PSO/
│   └── tBL_WSe2/
├── notebooks/
├── output/   
├── params/
└── scripts/  
```

### 5. Run the Demo Reconstructions

Please check the following before running the demo:
1. Demo datasets are downloaded and placed to the correct location under `ptyrad/data/`
2. `(ptyrad)` environment is created and activated (in VS Code it's the "Select Kernel")

Now you're ready to run a quick demo using one of two interfaces: 
- **Interactive Jupyter interface (Recommended)**
  
    Run the `ptyrad/notebooks/run_ptyrad.ipynb` in VS code, or run the following command in terminal:

    ```bash
    jupyter notebook ./tutorials/run_ptyrad.ipynb # Or direcly open it in VS code
    ``` 

- **Command-line interface** (like your *Miniforge Prompt* terminal)
    ```bash
    # Assume working directory is at `ptyrad/` and (ptyrad) environment is activated
    ptyrad run "params/examples/tBL_WSe2.yaml"
    ```