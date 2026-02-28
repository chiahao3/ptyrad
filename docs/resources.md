# Resources

Welcome to the PtyRAD resources hub!

Here you will find official publications, educational materials, and related community tools to help you master electron ptychography and the PtyRAD software.

## Official Publications & Documentation
* [**PtyRAD GitHub Repo**](https://github.com/chiahao3/ptyrad): The official GitHub repo of PtyRAD.
* [**PtyRAD Documentation**](https://ptyrad.readthedocs.io/en/latest/index.html): The complete, up-to-date guide to installing, configuring, and using the software.
* [**PtyRAD Journal Paper**](https://academic.oup.com/mam/article/doi/10.1093/mam/ozaf070/8222545?utm_source=authortollfreelink&utm_campaign=mam&utm_medium=email&guestAccessKey=e9e13516-273a-4e46-bec4-7488e9001d7d): The official peer-reviewed publication (Microscopy and Microanalysis).
* [**PtyRAD arXiv Preprint**](https://arxiv.org/abs/2505.07814): The open-access preprint of the core paper.
* [**Paper Zenodo Archive**](https://doi.org/10.5281/zenodo.15392805): Preserved datasets, scripts, and reproducible code releases associated with the publication (29.5 GB).
* [**PtyRAD Paper Repo**](https://github.com/chiahao3/ptyrad_paper): Same with **Paper Zenodo Archive**, but strips off the datasets and files so it's only the code and notebooks.

## Tutorials & Educational Materials
* [**PtyRAD Official YouTube Channel**](https://www.youtube.com/@ptyrad_official): Video tutorials, workshop recordings, and code walkthroughs.
  <iframe width="560" height="315" src="https://www.youtube.com/embed/XQ6wsMe9DZ0?si=uOmpR33H-InvpOD9" title="PtyRAD YouTube Channel" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen style="margin-bottom: 20px; border-radius: 8px;"></iframe>

* [**PtyRAD Beta Test Materials (Cornell Box)**](https://cornell.box.com/s/n5balzf88jixescp9l15ojx7di4xn1uo): A curated collection containing raw demo data, tutorial recordings, and presentation slides.
* [**Algorithms & Code Structure of PtychoShelves (Blog)**](https://chiahao3.notion.site/Theory-Algorithm-and-Code-structure-of-PtychoShelves-c7bf28a1068c4a4f90aa77272602ab19): A deep dive written by myself detailing the algorithms and code structure of `PtychoShelves` and `fold_slice`.
* [**`fold_slice` Tutorial Slides (Argonne Box)**](https://anl.box.com/s/f7lk410lf62rnia70fztd5l7n567btyv): Educational slides compiled by Dr. Yi Jiang covering the mechanics of the `fold_slice` algorithm.

## Community & Ecosystem
* [**py4D-browser-transform**](https://github.com/chiahao3/py4D-browser-transform): A plugin for [py4D-browser](https://github.com/sezelt/py4D-browser) that provides utility functions for transforming the datacube, currently including flipping, transposing, permuting axes.
  ```{image} https://github.com/chiahao3/py4D-browser-transform/raw/main/assets/demo.gif
  :width: 600px
  :alt: py4D-browser-transform demo
* [**ptycho-packages**](https://github.com/chiahao3/ptycho-packages): A GitHub repository tracking and listing other available ptychography software packages across the scientific community.

  Selected list:
  ---
  | **Year** | **Reference**                                                                                                                                           | **Supported Algorithms**       | **Language**          | **Notes**                                                                                                     |
  | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------| --------------------- | --------------------------------------------------------------------------------------------------------------|
  | 2025     | [PtyRAD](https://github.com/chiahao3/ptyrad) (Lee 2025)                                                                                                 | AD                             | Python                | PyTorch-based AD package for electron and x-ray ptychography.                                                 |
  | 2025     | [phaser](https://github.com/hexane360/phaser) (Gilgenbach 2025)                                                                                         | ePIE, LSQML, AD                | Python                | Jax-based package with AD and conventional algos with different backends like cupy and pytorch (experimental) |
  | 2025     | [pty-chi](https://github.com/AdvancedPhotonSource/pty-chi) (Du 2025)                                                                                    | ePIE family, DM, LSQML, AD, BH | Python                | PyTorch-based ptychography engine (use Ptychodus for preprocessing)                                           |
  | 2025     | [ptyrodactyl](https://github.com/debangshu-mukherjee/ptyrodactyl)                                                                                       | AD                             | Python                | Jax-based AD for electron and optical ptychography                                                            |
  | 2025     | [quantem](https://github.com/electronmicroscopy/quantem)                                                                                                | AD, direct ptycho              | Python                | Developing full suite of electron microscopy methods                                                          |
  | 2023     | [PtyLab](https://github.com/PtyLab) (Loetgering 2023)                                                                                                   | ePIE family                    | MATLAB, Python, Julia | Fourier & conventional ptychography; mPIE, zPIE, aPIE, pcPIE, e3PIE.                                          |
  | 2021     | [fold_slice](https://github.com/yijiang1/fold_slice) (Chen 2021)                                                                                        | ePIE, DM, LSQML                | MATLAB                | Modified PtychoShelves for electron ptychography.                                                             |
  | 2021     | [py4DSTEM](https://github.com/py4dstem/py4DSTEM) (Savitzky 2021, Varvanides 2023)                                                                       | SSB, WDD, DM, RAAR, GD         | Python                | Full 4D-STEM toolbox beyond ptychography.                                                                     |
  | 2021     | [Adorym](https://github.com/mdw771/adorym) (Du 2021)                                                                                                    | AD                             | Python                | HPC AD framework for 2D/3D ptychography, CDI, holography, tomography.                                         |
  | 2020     | [PtychoShelves](https://www.psi.ch/en/sls/csaxs/software) (Wakonig 2020)                                                                                | ePIE, DM, LSQML                | MATLAB                | MATLAB-based GPU-accelerated engines for mixed state multislice. X-ray focused.                               |
  | 2016     | [PtyPy](https://ptycho.github.io/ptypy/) (Enders 2016)                                                                                                  | DM, RAAR, ePIE, ML             | Python                | Supports on-the-fly reconstructions; mixed probe and object.                                                  |