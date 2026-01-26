"""
Custom DataLoader class to load batched measurements either from GPU device memory or host RAM.

"""

import torch
import numpy as np
from typing import Optional, Union, List
from torch.utils.data import Dataset

class MeasDataLoader:
    """
    Data loader for PtyRAD experimental measurements with on-the-fly processing.
    
    Handles indexed slicing of experimental pattern arrays with flexible
    device placement (CPU/GPU), on-demand or pre-loaded options, and optional
    on-the-fly padding/resampling.
    
    Args:
        meas_arr: np.ndarray of experimental diffraction patterns [N, H, W]
        preload_data: If True, load all data into device memory at initialization.
                       If False, load on-demand per access. Default: True
        device: torch.device or str ('cpu', 'cuda', etc.). Default: 'cuda'
        dtype: torch data type for output tensors. Default: torch.float32
        meas_padded: Optional np.ndarray for on-the-fly padding. Padded pattern template.
        meas_padded_idx: Optional tuple (pad_h1, pad_h2, pad_w1, pad_w2) for padding regions.
        meas_scale_factors: Optional tuple (scale_h, scale_w) for on-the-fly resampling.
    """
    
    def __init__(
        self,
        meas_arr: np.ndarray,
        preload_data: bool = True,
        device: Union[str, torch.device] = 'cuda',
        dtype: torch.dtype = torch.float32,
        meas_padded: Optional[np.ndarray] = None,
        meas_padded_idx: Optional[tuple] = None,
        meas_scale_factors: Optional[tuple] = None,
    ):
        self.meas_arr = meas_arr
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.preload_data = preload_data
        self.N_scans = len(meas_arr)
        
        # On-the-fly processing parameters
        self.meas_padded     = torch.tensor(meas_padded, dtype=torch.float32, device=device) if meas_padded is not None else None
        self.meas_padded_idx = torch.tensor(meas_padded_idx, dtype=torch.int32, device=device) if meas_padded_idx is not None else None
        self.meas_scale_factors = meas_scale_factors
        
        if self.preload_data:
            # Load everything into device memory at init
            if not meas_arr.flags['C_CONTIGUOUS']:                    
                meas_arr = np.ascontiguousarray(meas_arr) # PyTorch can't create tensor from numpy array with negative strides, so a contiguous RAM copy is temporarily needed
            self.data = torch.from_numpy(meas_arr).to(device=self.device, dtype=self.dtype)
        else:
            # Keep as numpy array, load on-demand
            self.data = meas_arr
    
    def __len__(self) -> int:
        """Return the total number of diffraction patterns."""
        return self.N_scans
    
    def __getitem__(self, idx: Union[int, List, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Get measurement data by index or indices with optional on-the-fly processing.
        
        Args:
            idx: Single index (int), array of indices (list, np.ndarray), or tensor indices
        
        Returns:
            Tensor of experimental patterns on the specified device, with optional
            padding/resampling applied.
        """
        
        # Convert tensor indices to numpy on CPU for slicing. This should only happen when using Accelerator and DDP with multiGPU
        if isinstance(idx, torch.Tensor):
            idx = idx.cpu().numpy()
            
        if self.preload_data:
            # Data already on device
            measurements = self.data[idx]
        else:
            # Load from numpy and convert to tensor
            sliced_data = np.asarray(self.data[idx])
            if not sliced_data.flags['C_CONTIGUOUS']:                    
                sliced_data = np.ascontiguousarray(sliced_data) # PyTorch can't create tensor from numpy array with negative strides, so a contiguous RAM copy is needed
            measurements = torch.from_numpy(sliced_data).to(device=self.device, dtype=self.dtype)
        
        # Apply on-the-fly padding if configured
        if self.meas_padded is not None:
            pad_h1, pad_h2, pad_w1, pad_w2 = self.meas_padded_idx
            canvas = torch.zeros(
                (measurements.shape[0], *self.meas_padded.shape[-2:]),
                dtype=self.dtype, device=self.device
            )
            canvas += self.meas_padded
            canvas[..., pad_h1:pad_h2, pad_w1:pad_w2] = measurements
            measurements = canvas
        
        # Apply on-the-fly resampling if configured
        if self.meas_scale_factors is not None:
            scale_h, scale_w = self.meas_scale_factors
            if scale_h != 1 or scale_w != 1:
                # 2D interpolate requires 4D input (N, C, H, W)
                measurements = torch.nn.functional.interpolate(
                    measurements.unsqueeze(1),
                    scale_factor=(scale_h, scale_w),
                    mode='bilinear'
                ).squeeze(1)
                # Normalize to preserve intensity scale
                measurements = measurements / (scale_h * scale_w)
        
        return measurements

class IndicesDataset(Dataset):
    """
    The Dataset class used specifically for the multiGPU mode for DDP
    """
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx]

