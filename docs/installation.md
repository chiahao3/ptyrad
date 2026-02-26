(reference:installation)=

# Installation

*PtyRAD* package is available on both `PyPI` and `conda-forge`, while the source codes are hosted on `GitHub` with additional demo params files and demo scripts.

## Platform and hardware compatibility

*PtyRAD* uses PyTorch as the computation backend, which supports a wide range of machines as shown below.

Assuming you have created and activated a fresh Python environment for *PtyRAD*, click on the corresponding {fas}`check text-success` to jump to your options:

|                  | {fab}`windows` Windows                        | {fab}`apple` macOS                           | {fab}`linux` Linux                           |
|:----------------:|:---------------------------------------------:|:--------------------------------------------:|:--------------------------------------------:|
| CPU              | [{fas}`check text-success`](#install-cpu)     | [{fas}`check text-success`](#install-macos)  | [{fas}`check text-success`](#install-cpu)    |
| NVIDIA GPU       | [{fas}`check text-success`](#install-nvidia)  | n/a                                          | [{fas}`check text-success`](#install-nvidia) |
| AMD GPU          |  {fas}`times text-danger`                     | n/a                                          | [{fas}`check text-success`](#install-amd)    |
| Apple GPU        | n/a                                           | [{fas}`check text-success`](#install-macos)  |   n/a                                        |

## Installation options

:::::::{tab-set}

::::::{tab-item} NVidia

**Windows and Linux system with a CUDA-supported GPU (NVidia)**

:::::{tab-set}

::::{tab-item} pip

```bash
pip install ptyrad
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
```
:::{admonition} Specify CUDA version
:class: dropdown note
PyTorch ships with the specified CUDA runtime (the `/cu118` in the url) so please make sure your machine has a compatible GPU driver. Run `nvidia-smi` in your terminal to see the maximum supported CUDA runtime version. You can modify the url if your machine supports newer CUDA runtime.
:::

::::

::::{tab-item} conda

```bash
conda install ptyrad pytorch-gpu -c conda-forge
```

:::{admonition} Specify CUDA version
:class: dropdown note
Although `pytorch-gpu -c conda-forge` should automatically detect a compatible CUDA runtime version for your machine, if you want explicit control of the CUDA runtime version via `conda`, you should do:
```bash
conda install ptyrad pytorch pytorch-cuda=11.8 -c nvidia -c pytorch -c conda-forge
```
The `pytorch-cuda=11.8` specifies the CUDA runtime version and you can modify the `pytorch-cuda=X.Y` if your machine supports newer CUDA runtime. Run `nvidia-smi` in your terminal to see the maximum supported CUDA runtime version. 

However, note that **PyTorch's official Anaconda channel (`-c pytorch`) is deprecated since PyTorch 2.6** according to the [PyTorch team announcement](https://github.com/pytorch/pytorch/issues/138506), so the maximum version of PyTorch you can get will be 2.5.1 via this command. You have to use `-c conda-forge` to get newer PyTorch if you still want to use `conda`.

:::

::::

:::::

::::::


::::::{tab-item} AMD

**Linux with an ROCm-supported GPU (AMD)**

:::{important}
*PtyRAD* is explictly tested on Windows and Linux with NVidia GPUs and macOS with an Apple Silicon GPU. If you're using AMD GPUs on Linux, some functionality may not work as expected. Please report your issues [here](https://github.com/chiahao3/ptyrad/issues).
:::

:::::{tab-set}

::::{tab-item} pip

```bash
pip install ptyrad
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.3 --force-reinstall
```
:::{admonition} Specify ROCm version
:class: dropdown note
PyTorch ships with the specified ROCm runtime (the `/rocm6.3` in the url) so please make sure your machine has a compatible GPU driver. Run `amd-smi` in your terminal to see the maximum supported ROCm runtime version. You can modify the url if your machine supports newer ROCm runtime.
:::

::::

::::{tab-item} conda

`conda-forge/pytorch` team is still working on making compatible PyTorch build with ROCm (AMD) GPUs. It may or may not happen, please use `pip` for now, or follow progress [here](https://github.com/conda-forge/pytorch-cpu-feedstock/issues/198).

::::

:::::

::::::

::::::{tab-item} CPU

**Windows / Linux without a GPU**

:::::{tab-set}

::::{tab-item} pip

```bash
pip install ptyrad
```

::::

::::{tab-item} conda

```bash
conda install ptyrad -c conda-forge
```

::::

:::::

::::::

::::::{tab-item} MacOS

**MacOS with or without Apple Silicon GPU**

:::::{tab-set}

::::{tab-item} pip

```bash
pip install ptyrad
```

:::{admonition} macOS with Apple Silicon GPU
:class: dropdown note
PyTorch's macOS pip wheels contain the MPS support by default, but it will only be used if supported Apple Silicon hardware is detected.
:::

::::

::::{tab-item} conda

```bash
conda install ptyrad -c conda-forge
```

:::{admonition} macOS with Apple Silicon GPU
:class: dropdown note
`conda-forge/pytorch` provides a universal macOS build with MPS support included, but it will only be used if supported Apple Silicon hardware is detected.
:::

::::

:::::

::::::

:::::::

<!-- Invisible references just to suppress Sphinx warning, we have used JavaScript -->
(install-cpu)=
(install-macos)=
(install-nvidia)=
(install-amd)=