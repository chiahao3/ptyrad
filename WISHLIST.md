# New Features and Enhancements


Last Update: 2026.02.13

---

### User Interface

- Consider a GUI for easier operation

### Logging

- Rebuild `logger` logic and improve `vprint`
  - Don't use vprint on everything
  - Peripheral functions should be silent, do the logging only inside the solver context
  - Would need to canocalize the logger usage, if possible, split the rank logic with logger, and only decorate the logger if needed

### Simulator

- Maybe add a namespace sub-package for torch-based 4D-STEM simulation

### Overall workflows

- Add better error messages.
- Add pyramidal approaches
    - Similar to `asize.presolve` from PtyShv to initialize from rough object with coarse sampling
    - Appending different reconstruciton engines
    - Could start from notebooks, but ultimately a wrapper class would be nice

### Initialization

- Split `initializer.py` into smaller modules, it's almost 2000 lines now
- Add padding and resampling to loaded object, probe, and position altogether as `recon_resampling`
- Add object initialization methods
    - [autocorrelation](https://doi.org/10.1364/OPTICA.522380)
    - Wirtinger Flow spectral method
    - tcBF?

### Forward

- Test propagator with higher order terms to handle larger convergence angle
    - [A multislice approach to the spherical aberration theory in electron optics](https://www.tandfonline.com/doi/abs/10.1080/09500349608232892)

### Models

- Split `ptycho_model.py` into smaller modules for probe.py, obj.py, pos.py, propagator.py
- Revisit the `Npix-simu` branch about allowing forward model to generate diffraction patterns with larger kMax than the actual data. We can either center-crop the forward diffraction pattern before calculating the loss, or add a mask to exclude gradients from the additional region.
    - This would require the object and probe to be initialized at higher real space sampling than the data kMax, but it will allow the forward model to properly scatter outside of the collected region on the detector, hence reduce the edge artifact. This is a better approach than padding the experimental diffraction pattern.
    - Although early attempts seem to suggest the extra k region were not constraint by data at all so it creates weird artifact in the k-space probe. Would need to regularize probe to supress that.
- Add optimizable param of forward model CBED shift correction for k-space misalignment with data
    - Should probably just do that right before applying detector blur inside the model forward method
    - We can make it similar to crystal tilt that has options of “fixed”, “global”, or “each”
- Add optimizable params of pos-dependent defocus and pos-dependent thickness
    - https://arxiv.org/abs/2504.17501
    - Note that this optimization would likely be just a refinement from typical MEP
    - A postprocessing of object is needed to map it back to Cartessian space
- Might be an interesting approach to optimize probe in k-space or completely with aberration coefficients, optimize object in real space, and optimizer tilt / thickness with PACBED because we can have multiple optimizers for different parameters.

### Reconstruction

- Decouple the reconstruction objective with data error so that we can reconstruct with whatever target loss, while having an independent data error metric that can be used as a standard value for comparison

### Loss Functions

- Add new error metrics based in image space (spatial correlation-aware loss).
    - Might be more useful for low dose as it captures spatial correlations. This can be implemented as new loss, just make some virtual image in the batch, although it'll be spatially nonsense because of the mini batch, but hopefully it'll capture the spatial correlation. It will only look like an image if we do compact grouping or large batch. We can do a vBF loss and vDF loss.
    - vBF loss (virtual bright field).
    - vDF loss (virtual dark field).
- Add PACBED loss for better thickness and tilt optimization.
    - Do we even need the crystal to estimate the tilt? Can't we directly estimate it with PACBED?
    - do I use total summed CBED or 1 unit cell for more features?
    - Fix probe, object, position, and just optimize tilt and thickness
    - We can also add that directly to BO
        - Might want to save the PACBED at each batch and normalize it by the end of each iteration
        - Need to take the on-the-fly resampling into account so the file shape matches

### Constraints

- Probe
    - Use a vacuum probe (either real or reciprocal space) to regularize the probe intensity and spatial distributions
    - Develop Orthogonal Probe Relaxation (i.e., allow variable probe modes).
        - Seems important for X-rays
        - Don’t need to give up on the total intensity constraint
    - Fit aberration to k-space probe to enforce smoothness of the aberration chi function.
        - Need to think about how to incorporate this with mixed state probe. It might be better to leave some probe modes un-constrained just like `probe_mask_k` use the accumulated power
    - Fix the probe corner intensity artifact. Feel like some intrinsic phase instability of complex probe
    - Add an active decoupling between probe and object to avoid probe absorbing too much object structure. Could be a deconvolution in either space. Should look into how PtyShv update the probe closer, and maybe implement an illumination-normalized constraint, or just a full option of conventional analytical grad update for probe
- Position
    - Try [iCGD](https://github.com/ningustc/iCGD) for their position constrain

### Visualization

- Refine plotting and saving routines.
    - Plotting with matplotlib feels very slow
- Plot radially accumulated intensity for k-space probe
- Add a `plot_obj_tilts_interp` for interpolated version of tilt_x, tilt_y for cleaner visualization could be nice
- Plot obj FFT figure, use the FOV region and PS FFT. Maybe to `plot_summary` and `save_results` as well
- Add convergence monitoring:
    - Convergence line plot for object, probe, and pos in either relative or absolute scale. Should monitor how this 1 value change. Probably the size of the gradient? Print it out like PtyShv or maybe plot it as figure? Should I do gradient or actual update? Or should I just track the actual difference?
    - Line plots tracking object, probe, and position updates.
    - Track gradients or update norms per iteration.
        - results will seem to converge with an adaptive/decreasing update rate, bit that doesn't really mean it converges, or even converges to the correct results
        - The update step would be damped by adaptive learning rate, so we should probably track original grad

### Output

- Write modeled CBED as an output for py4DGUI examination
- Finish the weighted sum of `omode_occu` in `save_results` when `omode_occu != 'uniform’`

### User-friendly scripts and utils

- Try [4DSTEM-calibration] (https://github.com/ningustc/4DSTEM-Calibration) for position correction initialization with better affine transformation values
- Add a scan rotation fitting routine from the curl of gradCoM of CBEDs similar to the py4DSTEM's `solve_for_center_of_mass_relative_rotation` could be very handy
- Data orientation checking script (permuting 8 configurations)

### Code clarity

- Improve the dict usage with .get with default value to prevent version mismatch between PtyRAD and the params files
- Unify meshgrid usage, naming, and unit would be nice
- Improve function signatures with type hints, clearer positional/keyword arguments with default values
- Add proper doc strings (Google style) for major functions and classes
- Run through Ruff formatter for PEP8 code style (for certain part of the PtyRAD codebase)

### Test

- Make a list of tests that can help ensure the robustness of future development