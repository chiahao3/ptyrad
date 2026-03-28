"""
Developer tools for logging, testing, checking sizes and types, etc.

"""

import ast
import os
from collections import defaultdict

import numpy as np


def print_package_tree(package_path):
    """
    `print_package_tree` prints the package structure with module, class, method, and function definitions for a concise view of the entire package structure

    Args:
        package_path (str): package_path (str): Path to the target package
    """
    def parse_defs(file_path, rel_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read(), filename=file_path)
            except SyntaxError:
                return []

        defs = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                defs.append((rel_path, "Function", node.name))
            elif isinstance(node, ast.ClassDef):
                defs.append((rel_path, "Class", node.name))
                for sub_node in node.body:
                    if isinstance(sub_node, ast.FunctionDef):
                        defs.append((rel_path, "Method", sub_node.name))
        return defs

    # Gather all defs from the package
    collected_defs = []
    for root, _, files in os.walk(package_path):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, package_path)
                collected_defs.extend(parse_defs(full_path, rel_path))

    # Organize into tree structure: {module -> {class_or_func_name -> [methods]}}
    tree = defaultdict(lambda: defaultdict(list))
    for module, kind, name in collected_defs:
        if kind == "Class":
            tree[module][name] = []
        elif kind == "Function":
            tree[module][name] = None
        elif kind == "Method":
            # Add to the last class added (assuming no nested classes)
            last_class = next(reversed(tree[module]))
            tree[module][last_class].append(name)

    # Print formatted output with connectors
    for module in sorted(tree):
        print(f"📄 {module}")
        items = list(tree[module].items())
        for i, (name, methods) in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└──" if is_last else "├──"
            if methods is None:
                print(f"  {connector} def {name}()")
            else:
                print(f"  {connector} class {name}:")
                for j, method in enumerate(methods):
                    sub_connector = "└──" if j == len(methods) - 1 else "├──"
                    print(f"      {sub_connector} def {method}()")
        print()

def has_nan_or_inf(tensor):
    """
    Check if a torch.Tensor contains any NaN or Inf values.

    Parameters:
        tensor (torch.Tensor): Input tensor to check.

    Returns:
        bool: True if the tensor contains any NaN or Inf values, False otherwise.
    """
    import torch
    
    # Check for NaN values
    has_nan = torch.isnan(tensor).any()

    # Check for Inf values
    has_inf = torch.isinf(tensor).any()

    return has_nan or has_inf

def get_size_bytes(x):
    import torch
    
    print(f"Input tensor has shape {x.shape}, dtype {x.dtype}, and live on {x.device}")
    size_bytes = torch.numel(x) * x.element_size()
    size_mib = size_bytes / (1024 * 1024)
    size_gib = size_bytes / (1024 * 1024 * 1024)
    
    if size_bytes < 128 * 1024 * 1024:
        print(f"The size of the tensor is {size_mib:.2f} MiB")
    else:
        print(f"The size of the tensor is {size_gib:.2f} GiB")
    return size_bytes

def check_modes_ortho(tensor, rtol=1e-3):
    ''' Check if the modes in tensor (Nmodes, []) is orthogonal to each other'''
    # The easiest way to check orthogonality is to calculate the dot product of their 1D vector views
    # Orthogonal vectors would have dot product equals to 0 (Note that `orthonormal` also requires they have unit length)
    # Note that due to the floating point precision, we should set a reasonable tolerance w.r.t 0.
    # Also note that Matlab's dot(p2,p1) for complex input would implictly apply with the complex conjugate, 
    # so Matlab's dot() != torch.dot because torch.dot doesn't automatically apply the complex conjugate.
    # This is pointed out by @dong-zehao in issue #11.
    # Therefore, instead of torch.dot(a,a), which would output un-intended result when a is complex,
    # use torch.dot(a, a.conj()) for the correct inner product.
    import torch
    import numpy as np

    # Automatically convert numpy array to torch tensor
    if isinstance(tensor, np.ndarray):
        print("Casting input tensor from 'np.ndarray' to 'torch.tensor'")
        tensor = torch.tensor(tensor)

    tensor = tensor.to(dtype=torch.complex128)

    print(f"Input tensor has shape {tensor.shape} and dtype {tensor.dtype}")

    # Initialize the master flag
    is_orthogonal = True 

    for i in range(tensor.shape[0]):
        ai = tensor[i].view(-1)
        norm_i = torch.linalg.norm(ai)

        for j in range(i + 1, tensor.shape[0]):
            aj = tensor[j].view(-1)
            norm_j = torch.linalg.norm(aj)

            dot = torch.dot(ai, aj.conj()) # Note that torch.dot only takes 1D tensor

            # Relative overlap (dimensionless!)
            rel_overlap = dot.abs() / (norm_i * norm_j + 1e-16)

            if rel_overlap < rtol:
                print(f"Modes {i}, {j} orthogonal: rel_overlap = {rel_overlap.item():.3e}")
            else:
                print(f"Modes {i}, {j} NOT orthogonal: rel_overlap = {rel_overlap.item():.3e}")
                # Flip the flag if any pair fails the tolerance check
                is_orthogonal = False 

    return is_orthogonal

# Testing functions
def test_loss_fn(model, indices, loss_fn):
    """ Print loss values for each term for convenient weight tuning """
    # model: PtychoModel model
    # indices: array-like indices indicating which probe position to evaluate
    # measurements: 4D-STEM data that's already passed to DEVICE
    # loss_fn: loss function object created from CombinedLoss
    import torch
    
    with torch.no_grad():
        model_CBEDs, objp_patches = model(indices)
        measured_CBEDs = model.get_measurements(indices)
        _, losses = loss_fn(model_CBEDs, measured_CBEDs, objp_patches, model.omode_occu)

        # Print loss_name and loss_value with padding
        for loss_name, loss_value in zip(loss_fn.loss_params.keys(), losses):
            print(f"{loss_name.ljust(11)}: {loss_value.detach().cpu().numpy():.8f}")
    return

def test_constraint_fn(test_model, constraint_fn, plot_forward_pass):
    """ Test run of the constraint_fn """
    # Note that this would directly modify the model so we need to make a test one

    indices = np.random.randint(0,len(test_model.measurements),2)
    
    constraint_fn(test_model, niter=1) 
    if plot_forward_pass is not None:
        plot_forward_pass(test_model, indices, 0.5)
    del test_model
    return