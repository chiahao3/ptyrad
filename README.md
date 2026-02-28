# PtyRAD: Ptychographic Reconstruction with Automatic Differentiation
![PyPI - Version](https://img.shields.io/pypi/v/ptyrad)
[![PyPI Downloads](https://static.pepy.tech/badge/ptyrad)](https://pepy.tech/projects/ptyrad)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/ptyrad/badges/version.svg)](https://anaconda.org/conda-forge/ptyrad)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/ptyrad/badges/latest_release_date.svg)](https://anaconda.org/conda-forge/ptyrad)
[![arXiv](https://img.shields.io/badge/arXiv-2505.07814-b31b1b.svg)](https://arxiv.org/abs/2505.07814)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15392805.svg)](https://doi.org/10.5281/zenodo.15392805)

[**Docs**](https://ptyrad.readthedocs.io/en/latest/index.html)
| [**Install Guide**](https://ptyrad.readthedocs.io/en/latest/installation.html)
| [**Quickstart**](https://ptyrad.readthedocs.io/en/latest/quickstart.html)
| [**Paper**](https://arxiv.org/abs/2505.07814)
| [**Youtube**](https://www.youtube.com/@ptyrad_official)

*PtyRAD* performs ptychographic reconstruction using an [automatic differention](https://en.wikipedia.org/wiki/Automatic_differentiation) (AD) framework powered by [*PyTorch*](https://pytorch.org/), enabling flexible and efficient implementation of gradient descent optimization. See our [Microscopy and Microanalysis paper](https://academic.oup.com/mam/article/doi/10.1093/mam/ozaf070/8222545?utm_source=authortollfreelink&utm_campaign=mam&utm_medium=email&guestAccessKey=e9e13516-273a-4e46-bec4-7488e9001d7d) and the [Zenodo record](https://doi.org/10.5281/zenodo.15273176) for more information and demo datasets.

## Features
- Automatic Differentiation (AD) based optimization
- Gradient descent algorithms (Adam, SGD, LBFGS, etc.)
- Mixed-state probe and object
- Position correction
- Position-dependent object tilt correction
- Interoperability with *PtychoShelves (fold_slice)* and *py4DSTEM*
- Streamlined preprocessing of cropping, padding, resampling, adding noises, and many more
- Hyperparameter tuning
- Multi-GPU reconstructions
- JIT compilation with `torch.compile`

## Recommended Tools
We recommend using [*Miniforge*](https://github.com/conda-forge/miniforge) for Python environment management, and  
[*Visual Studio Code*](https://code.visualstudio.com/Download) for code editing and execution.

## Major dependencies

* Python 3.10 or above
* PyTorch 2.0 or above
* While *PtyRAD* can run on CPU, GPU is strongly suggested for high-speed ptychographic reconstructions. 
    - *PtyRAD* supports both NVIDIA GPUs with CUDA and Apple Silicon (MPS)
* *PtyRAD* was tested on Windows, MacOS, and Linux

## Installation



We recommend installing *PtyRAD* using `pip` inside a fresh conda environment.

### 1. Create and Activate the Python Environment

First, create and activate a new conda environment **(ptyrad)** with Python > 3.10:
```sh
conda create -n ptyrad python=3.12
conda activate ptyrad
```
> 💡 **Note:** After activating the environment, your terminal prompt should show **(ptyrad)** at the beginning, indicating that the environment is active.

### 2. Install PtyRAD in the Python Environment

Then install *PtyRAD* in the activated `(ptyrad)` environment using:
```sh
pip install ptyrad
```

If you're using Windows with NVIDIA CUDA GPU, you will also need to install the GPU version of PyTorch with:
```sh
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
```

*PtyRAD* can also be installed via `conda`. For detailed instructions on installing *PtyRAD* on different machines or pinning specific CUDA versions, see [the installation guide](https://ptyrad.readthedocs.io/en/latest/installation.html).



### How do I check if my installed PtyRAD has the GPU support?
CUDA version, GPU support, and PyTorch build across platforms can be extremely confusing, so *PtyRAD* provides handy CLI tools to help check these information for you!

Once you activated `(ptyrad)` environment and installed *PtyRAD* via `pip install ptyrad`, you'll have access to the following command:

```bash
# You can run this command anywhere, as long as (ptyrad) environment is activated
ptyrad check-gpu
```

This command will print your CUDA information and GPU availability if available.

### How do I update my existing PtyRAD installation to a newer release?
Assuming you've activated the `(ptyrad)` environment, and you've installed *PtyRAD* via pip, you can simply update your PtyRAD installation with:

```bash
pip install -U ptyrad
```

## Get Started with the Demo

> 💡 **Note:** *PtyRAD* now includes a **starter kit** that sets up the folder structure, tutorial notebooks, scripts, and example params files for you, with just one line of code!

### 1. Initialize a Workspace
Run the following command to create a new folder (e.g., `ptyrad/`) containing all necessary templates and scripts:

```bash
# Activate your (ptyrad) python environment
conda activate ptyrad

# This creates a workspace folder 'ptyrad/' in your current location
ptyrad init # or `ptyrad init <FOLDER_NAME> to use custom folder name

# Enter the directory
cd ptyrad/
```

The initialize workspace folder structure will look like this:

```text
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

### 2. Download the Demo Data
We provide a helper script to automatically fetch the example datasets, and place it in the correct `ptyrad/data/` folder:
```bash
# Download and extract zip files (tBL-WSe2 and PSO, 1.3 GB), should be done in 1-2 mins.
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

### 3. Run the Demo Reconstructions
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

## Documentation
*PtyRAD* documentation is available at https://ptyrad.readthedocs.io/en/latest/index.html.

## Author

Chia-Hao Lee (cl2696@cornell.edu)

Developed at the Muller Group, Cornell University.

## Acknowledgments

Besides great support from the entire Muller group, this package gets inspiration from lots of community efforts, and specifically from the following packages. Some of the functions in *PtyRAD* are directly translated or modified from these packages as noted in their docstrings/comments to give explicit acknowledgment.
* [PtychoShelves](https://journals.iucr.org/j/issues/2020/02/00/zy5001/index.html)
* [fold_slice](https://github.com/yijiang1/fold_slice)
* [py4dstem](https://github.com/py4dstem/py4DSTEM)
* [adorym](https://github.com/mdw771/adorym)
* [SciComPty](https://www.mdpi.com/2410-3896/6/4/36)

## Other resources

* [ptycho-packages](https://github.com/chiahao3/ptycho-packages) lists many available ptychography packages
* [Cornell Box folder](https://cornell.box.com/s/n5balzf88jixescp9l15ojx7di4xn1uo) compiled by myself that keeps demo data, tutorial recordings, and slides for PtyRAD
* [Argonne Box folder](https://anl.box.com/s/f7lk410lf62rnia70fztd5l7n567btyv) compiled by Dr. Yi Jiang that holds tutorial slides of `fold_slice`
* [Blog post](https://chiahao-blog.super.site/posts/theory-algorithm-and-code-structure-of-ptychoshelves) written by myself that details the algorithms and code structure of `PtychoShelves` / `fold_slice`