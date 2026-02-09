"""
Grouping functions for scan positions

"""

import numpy as np

def sparse_sampler_fps(points, n_groups, seed=None):
    """
    Splits 2D points into G groups using Farthest Point Sampling (FPS) to ensure 
    maximum spatial separation (hyperuniformity) within each group.

    Mechanism:
      1. Maintains a distance cache `dist_caches` of shape (G, N), where entry [g, i] 
         is the distance from point i to the nearest existing member of group g.
      2. Iteratively selects the point with the maximum distance value for the current 
         target group (greedy maximin strategy).
      3. Marks selected points as "visited" globally by setting their distance to -1.0 
         in the cache, ensuring sampling without replacement.

    Args:
        points (np.ndarray): (N, 2) array of coordinates.
        n_groups (int): Number of desired groups (mini-batches).

    Returns:
        list[np.ndarray[int]]: A list of G arrays, where each inner array contains indices.
    """
    N = len(points)
    points = points.astype(np.float32) # Cast to float32 to reduce memory footprint and increase cache locality
    
    groups = [[] for _ in range(n_groups)]
    
    # Initialize distance matrix with Infinity. 
    dist_caches = np.full((n_groups, N), np.inf, dtype=np.float32)
    
    # Seed Group 0 with a random starting point
    np.random.seed(seed)
    first_idx = np.random.randint(0, N)
    groups[0].append(first_idx)
    
    # Update Group 0's distance cache based on this seed
    dists = np.linalg.norm(points - points[first_idx], axis=1)
    dist_caches[0] = np.minimum(dist_caches[0], dists)
    
    # Mark the seed as visited globally (for ALL groups) so no other group picks it
    # We use -1.0 as a flag for "visited" points to avoid managing a separate boolean mask.
    dist_caches[:, first_idx] = -1.0 

    # Initialize Groups 1..G by picking points farthest from Group 0's set.
    # This heuristic separates the starting seeds of different groups.
    for g in range(1, n_groups):
        # Find the point farthest from Group 0
        # (argmax ignores -1.0 as long as positive distances exist)
        farthest_idx = np.argmax(dist_caches[0])
        groups[g].append(farthest_idx)
        
        # Update distances for the current group
        d_new = np.linalg.norm(points - points[farthest_idx], axis=1)
        dist_caches[g] = np.minimum(dist_caches[g], d_new)
        
        # Mark as visited globally
        dist_caches[:, farthest_idx] = -1.0

    # Iteratively assign remaining points to groups in a round-robin fashion.
    count = n_groups
    
    while count < N:
        g = count % n_groups
        
        # Select the available point farthest from the current members of group g
        next_idx = np.argmax(dist_caches[g])
        groups[g].append(next_idx)
        
        # Calculate distances from the new point to all other points
        d_new = np.linalg.norm(points - points[next_idx], axis=1)
        
        # Update this group's minimal distance cache
        # Note: If a point 'k' is already visited, dist_caches[g, k] is -1.0.
        # np.minimum(-1.0, d_new) remains -1.0, preserving the visited state.
        dist_caches[g] = np.minimum(dist_caches[g], d_new)
        
        # Mark the new point as visited for ALL groups
        dist_caches[:, next_idx] = -1.0
        
        count += 1
    
    # Final type cleaning: Turn it into list of array of int
    groups = [np.int64(group) for group in groups]
    
    return groups

def sparse_sampler_hilbert(points, n_groups, resolution=16):
    """
    Splits 2D points into G groups using Hilbert Curve sorting to ensure 
    spatial stratification (hyperuniformity) within each group.

    Mechanism:
      1. Map continuous (x, y) coordinates to a discrete 1D Hilbert integer index.
      2. Sort points by this 1D index to group spatially local points together.
      3. Use strided indexing (modulo arithmetic) to peel off layers, ensuring 
         that consecutive points in a group are distant in 2D space.

    Args:
        points (np.ndarray): (N, 2) array of coordinates.
        n_groups (int): Number of desired groups (mini-batches).
        resolution (int): Grid resolution order. n=16 creates a 65536x65536 grid, 
                          sufficient for most float precision needs.

    Returns:
        list[np.ndarray[int]]: A list of G arrays, where each inner array contains indices.
    """

    # Normalization to [0,1]
    p_min = points.min(0)
    p_max = points.max(0)
    norm_p = (points - p_min) / (p_max - p_min + 1e-9) # prevents index out-of-bounds for the exact max value.

    # Calculate the 1D Hilbert index (key) for every point
    keys = [get_hilbert_key(p, resolution) for p in norm_p]
    
    # Get indices that sort the points along the curve
    sorted_idx = np.argsort(keys)

    # Picking every G-th point guarantees they are far apart in the group.
    groups = [[] for _ in range(n_groups)]
    
    for i, original_idx in enumerate(sorted_idx):
        group_id = i % n_groups
        groups[group_id].append(original_idx)
    
    # Final type cleaning: Turn it into list of array of int
    groups = [np.int64(group) for group in groups]
    
    return groups

def get_hilbert_key(p, resolution=16):
    """
    Calculates the Hilbert curve integer index for a single normalized point.
    Uses bitwise operations to traverse the recursive quadrants.
    """
    # 1. Discretize: Scale continuous [0,1] coords to integer grid [0, 2^n - 1]
    limit = 2**resolution - 1
    x = int(p[0] * limit)
    y = int(p[1] * limit)
    
    d = 0 # The running 1D distance (Hilbert index)
    s = 2**(resolution - 1) # Start with the largest quadrant size
    
    # 2. Iteratively determine which quadrant the point is in
    while s > 0:
        # Check if x or y is in the upper half of the current quadrant size 's'
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        
        # Add the distance for the quadrant we are in.
        # (3 * rx) ^ ry maps the (rx, ry) coords to the order (0, 1, 2, 3)
        d += s * s * ((3 * rx) ^ ry)
        
        # 3. Rotate/Flip: Hilbert curves require rotation when entering specific quadrants
        # If we are in the 'bottom-left' or 'bottom-right' relative to parent...
        if ry == 0:
            if rx == 1:
                # Flip coordinates for the symmetric quadrant
                x = limit - x
                y = limit - y
            
            # Swap x and y (Rotation)
            x, y = y, x
        
        s //= 2 # Descend to next level of detail
        
    return d

def remap_batches_to_global(local_batches, global_lookup):
    """
    Maps batch indices from a local subset coordinate system back to the global coordinate system.
    
    Args:
        local_batches (list of arrays): The output from the sampler (indices 0..M-1).
        global_lookup (np.ndarray): The actual global indices corresponding to the subset (values 0..N-1).
                                    This is your original 'indices' array.
    
    Returns:
        list of np.ndarrays: Batches containing the global indices.
    """
    # Ensure global_lookup is a numpy array for advanced indexing
    if not isinstance(global_lookup, np.ndarray):
        global_lookup = np.array(global_lookup)

    global_batches = []
    for batch in local_batches:
        # NumPy magic: Using a list/array of integers to index an array 
        # returns the values at those positions.
        # e.g. if global_lookup = [10, 20, 30] and batch = [0, 2], result is [10, 30]
        mapped_batch = global_lookup[batch]
        global_batches.append(mapped_batch)
        
    return global_batches