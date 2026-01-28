"""
Utility functions for logging, timing, image processing, etc.

"""

# ┌────────────────────────────────────────────────────────────────────────────┐
# │ PROMOTING COMMONLY USED FUNCTIONS FROM SUBMODULES                          │
# └────────────────────────────────────────────────────────────────────────────┘
# These relative imports make selected functions/classes available directly via:
#     from ptyrad.utils import vprint, get_time, fftshift2, ...
#
# This creates a *clean and stable public interface* for utils, while allowing
# internal code organization (e.g., separating utils functions for image_proc, 
# math_ops, and physics).
#
# Relative imports are used here because:
#   - This file is part of the same package (utils/)
#   - It avoids hardcoding the parent package name ("ptyrad"), which increases
#     flexibility for testing, sub-packaging, or renaming the root.

from .common import (  # noqa: F401
    CustomLogger,
    expand_presets,
    get_nested,
    get_time,
    handle_hdf5_types,
    list_nested_keys,
    ndarrays_to_tensors,
    parse_hypertune_params_to_str,
    parse_sec_to_time_str,
    print_gpu_info,
    print_system_info,
    resolve_seed_priority,
    safe_filename,
    set_accelerator,
    set_gpu_device,
    set_random_seed,
    tensors_to_ndarrays,
    time_sync,
    vprint,
    vprint_nested_dict,
)
from .grouping import (  # noqa: F401
    get_hilbert_key,
    remap_batches_to_global,
    sparse_sampler_fps,
    sparse_sampler_hilbert,
)
from .image_proc import (  # noqa: F401
    create_one_hot_mask,
    fit_background,
    fit_cbed_pattern,
    gaussian_blur_1d,
    get_blob_size,
    guess_radius_of_bright_field_disk,
    imshift_batch,
    normalize_by_bit_depth,
    normalize_from_zero_to_one,
)
from .math_ops import (  # noqa: F401
    compose_affine_matrix,
    dct_2d,
    exponential_decay,
    fftshift2,
    idct_2d,
    ifftshift2,
    make_gaussian_mask,
    make_sigmoid_mask,
    power_law,
    torch_phasor,
)
from .physics import (  # noqa: F401
    complex_object_z_resample_torch,
    get_EM_constants,
    infer_dx_from_params,
    make_fzp_probe,
    make_mixed_probe,
    make_stem_probe,
    near_field_evolution,
    near_field_evolution_torch,
    orthogonalize_modes_vec_np,
    sort_by_mode_int_np,
)
from .provenance import (  # noqa: F401
    collect_provenance,
    generate_provenance_json,
    save_provenance_to_hdf5,
)
