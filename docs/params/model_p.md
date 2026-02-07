# model_params

`model_params` determines the forward model behavior, the optimizer configuration, and the learning of the PyTorch model (PtychoModel)
optimizer configurations are specified in 'optimizer_params', see https://pytorch.org/docs/stable/optim.html for detailed information of available optimizers and configs.
update behaviors of optimizable variables (tensors) are specified in 'update_params'.
'start_iter' specifies the iteration at which the variables (tensors) can start being updated by automatic differentiation (AD)
'lr' specifies the learning rate for the variables (tensors)
Usually slower learning rate leads to better convergence/results, but is also updating slower. 
The variable optimization has 2 steps, (1) calculate gradient and (2) apply update based on learning rate * gradient
'start_iter: null' will disable grad calculation and would not update the variable regardless the learning rate through out the whole reconstruction
'start_iter: N(int)' would only calculate the grad when iteration >= N, so no grad will be calculated when iteration < N
Therefore, only the variable with non-zero learning rate would be optimized when iteration > start_iter.
If you don't want/need to optimize certain parameters, set their start_iter to null AND learning rate to 0 for faster computation. 
Typical learning rate is 1e-3 to 1e-4.

```yaml
model_params : {
    'obj_preblur_std'     : null, # type: null or float, unit: px (real space). This applies Gaussian blur to the object before simulating diffraction patterns. Since the gradient would flow to the original "object" before blurring, it's essentially deconvolving the object with a Gaussian kernel of specified std. This sort of deconvolution can generate sharp features, but the usage is not easily justifiable so treat it carefully as a visualization exploration
    'detector_blur_std'   : null, # type: null or float, unit: px (k-space). This applies Gaussian blur to the forward model simulated diffraction patterns to emulate the PSF of high-energy electrons on detector for experimental data. Typical value is 0-1 px (std) based on the acceleration voltage 
    'optimizer_params'    : {'name': 'Adam', 'configs': {}, 'load_state': null}, # Support all PyTorch optimizer including LBFGS because LBFGS can't set separate learning rates for different tensors. The suggested optimizer is 'Adam' with default configs (null). You can load the previous optimizer state by passing the path of `model.hdf5` to `load_state`, this way you can continue previous reconstruciton smoothly without abrupt gradients. (Because lots of the optimizers are adaptive and have history-dependent learning rate manipulation, so loading the optimizer state is necessary if you want to continue the previous optimization trajectory). However, the optimizer state must be coming from previous reconstructions with the same set of optimization variables with identical size of the dimensions otherwise it won't run.
    'update_params':{
        'obja'            : {'start_iter': 1, 'lr': 5.0e-4}, # object amplitude
        'objp'            : {'start_iter': 1, 'lr': 5.0e-4}, # object phase
        'obj_tilts'       : {'start_iter': null, 'lr': 0}, # object tilts
        'slice_thickness' : {'start_iter': null, 'lr': 0}, # object dz slice thickness 
        'probe'           : {'start_iter': 1, 'lr': 1.0e-4},  # probe
        'probe_pos_shifts': {'start_iter': 1, 'lr': 1.0e-4}, # sub-px probe positions
    }
}
```