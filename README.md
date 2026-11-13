# PtyRAD: Ptychographic Reconstruction with Automatic Differentiation

## 本分支实验功能 · 约束 Crop 模式的外圈散射

实测衍射为 128×128、前向模拟为 256×256 时，中心窗口继续用原有振幅、Poisson 或 PACBED 损失拟合。可选的 `loss_outside` 给模拟外圈一个环平均强度上限，抑制明显偏高的高角散射。它复用已有的前向模拟，默认关闭。

### 曲线怎么定

一阶 Born 近似下，散射振幅与静电势的傅里叶变换成正比。取屏蔽库仑势 $V(R)\propto e^{-\kappa R}/R$，其傅里叶变换与 $(q^2+\kappa^2)^{-1}$ 成正比。强度是振幅模平方，弹性项在大 $q$ 处趋于 $q^{-4}$。实现用以下相对半径曲线，并容许实测数据里的背景底。 [电子散射因子推导](https://journals.iucr.org/a/issues/2019/06/00/lk5052/)

$$I(r)=B+A\left(\frac{1+s^2}{r^2+s^2}\right)^2$$

$r=q/q_{\mathrm{edge}}$，$q_{\mathrm{edge}}$ 是从盘中心到方形探测器边缘的空间频率，$s$ 是相对屏蔽尺度。代码先平均扫描位置的实测衍射，再在对角扇区 $0.78\le r<1$ 鲁棒拟合 $B,A,s$。方形探测器的角落还有 $1\le r<1.4$ 的实测像素。留出这些环，取实测与拟合强度比的第 90 百分位，最低取 1，再乘 `margin` 作为上限的放大系数。224 和 232 两块真实 ROI 的角落留出 MAE 分别为全图平均衍射强度的 **0.85%** 和 **2.94%**。这两个数字衡量探测器角落的外推误差。多切片衍射、背景和晶体方向性都会改变实际散射，探测器以外的强度分布仍待验证。

### 怎么接入训练

把下面的键合并进现有 YAML，保留其他参数。示例值是首轮对照的起点。

```yaml
init_params:
  simu_Npix: 256
  simu_match_mode: crop
  meas_pad: null

loss_params:
  loss_outside:
    state: true
    weight: 0.05
    transition_px: 4.0
    margin: 1.25
```

每个小批次对 Crop 外的模拟强度按环加权平均。像素距裁切边界越远，权重按 $1-\exp[-(d/d_0)^2]$ 平滑升高，$d_0$ 是 `transition_px`。某环的预测强度 $\bar P$ 超过上限 $\bar U$ 时，加入平方振幅惩罚 $[\sqrt{\bar P}-\sqrt{\bar U}]_+^2$，低于上限时惩罚为零。这保留了单个衍射斑点的强度自由度。日志在第一轮记录拟合参数和角落误差，每轮记录加权后的 `loss_outside`。

首轮建议在同一界面 ROI、同一初始化和迭代数下比较关闭与开启外圈损失。一起看中心实测窗口的前向拟合、模拟外圈强度比例，以及同色标切片里的石墨条纹和金刚石哑铃。外圈比例下降只能说明约束起了作用，结构质量还要看这些独立指标。

---

![PyPI - Version](https://img.shields.io/pypi/v/ptyrad)
[![PyPI Downloads](https://static.pepy.tech/badge/ptyrad)](https://pepy.tech/projects/ptyrad)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/ptyrad/badges/version.svg)](https://anaconda.org/conda-forge/ptyrad)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/ptyrad/badges/latest_release_date.svg)](https://anaconda.org/conda-forge/ptyrad)
[![Paper](https://img.shields.io/badge/Paper-10.1093/mam/ozaf070-blue)](https://academic.oup.com/mam/article/doi/10.1093/mam/ozaf070/8222545?utm_source=authortollfreelink&utm_campaign=mam&utm_medium=email&guestAccessKey=e9e13516-273a-4e46-bec4-7488e9001d7d)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.15392805.svg)](https://doi.org/10.5281/zenodo.15392805)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/chiahao3/ptyrad)

[**Docs**](https://ptyrad.readthedocs.io/en/latest/index.html)
| [**Install Guide**](https://ptyrad.readthedocs.io/en/latest/installation.html)
| [**Quickstart**](https://ptyrad.readthedocs.io/en/latest/quickstart.html)
| [**Paper**](https://academic.oup.com/mam/article/doi/10.1093/mam/ozaf070/8222545?utm_source=authortollfreelink&utm_campaign=mam&utm_medium=email&guestAccessKey=e9e13516-273a-4e46-bec4-7488e9001d7d)
| [**Youtube**](https://www.youtube.com/@ptyrad_official)

<img src="docs/_static/imgs/exp_examples.png" alt="PtyRAD Examples" width="800">

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
We recommend using [*Miniforge*](https://conda-forge.org/download/) for Python environment management, and  
[*Visual Studio Code*](https://code.visualstudio.com/Download) for code editing and execution.

## Major dependencies

* Recommend Python 3.12 or above
* PyTorch 2.4 or above
* While *PtyRAD* can run on CPU, GPU is strongly suggested for high-speed ptychographic reconstructions. 
    - *PtyRAD* supports both NVIDIA GPUs with CUDA and Apple Silicon (MPS)
* *PtyRAD* was tested on Windows, MacOS, and Linux

## Installation



We recommend installing *PtyRAD* using `pip` inside a fresh conda environment.

### 1. Create and Activate the Python Environment

First, create and activate a new conda environment **(ptyrad)** with Python >= 3.12:
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
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
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
    jupyter notebook ./notebooks/run_ptyrad.ipynb # Or direcly open it in VS code
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

## Citing PtyRAD

If you use *PtyRAD* in your research, we kindly ask that you cite our [main paper](https://academic.oup.com/mam/article/31/4/ozaf070/8222545):

> Lee, C. H., Zeltmann, S. E., Yoon, D., Ma, D., & Muller, D. A. (2025). PtyRAD: A high-performance and flexible ptychographic reconstruction framework with automatic differentiation. Microscopy and Microanalysis, 31(4), ozaf070.

You can also use the following .bib for BibTex.

```bibtex
@article{lee2025ptyrad,
  title={PtyRAD: A high-performance and flexible ptychographic reconstruction framework with automatic differentiation},
  author={Lee, Chia-Hao and Zeltmann, Steven E and Yoon, Dasol and Ma, Desheng and Muller, David A},
  journal={Microscopy and Microanalysis},
  volume={31},
  number={4},
  pages={ozaf070},
  year={2025},
  publisher={Oxford University Press US}
}
```

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
* [Blog post](https://chiahao3.notion.site/Theory-Algorithm-and-Code-structure-of-PtychoShelves-c7bf28a1068c4a4f90aa77272602ab19) written by myself that details the algorithms and code structure of `PtychoShelves` / `fold_slice`
* [py4D-browser-transform](https://github.com/chiahao3/py4D-browser-transform): A plugin for [py4D-browser](https://github.com/sezelt/py4D-browser) that provides utility functions for transforming the datacube, currently including flipping, transposing, permuting axes.
    ![Demo GIF](https://github.com/chiahao3/py4D-browser-transform/raw/main/assets/demo.gif)