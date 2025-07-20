from tqdm import tqdm
import dask.array as da
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from omnialigner.dtypes import Dask_image_NCHW, Tensor_grid_M, Tensor_image_NCHW


def worker(args):
    """Process a chunk of the input tensor with grid sampling.

    Args:
        args (tuple): A tuple containing:
            - input_tensor (torch.Tensor): Input tensor of shape (N, C, H_in, W_in)
            - grid (torch.Tensor): Sampling grid of shape (N, H_out, W_out, 2)
            - n_range (tuple): Range of batch indices (n_start, n_end)
            - h_range (tuple): Range of height indices (h_start, h_end)
            - w_range (tuple): Range of width indices (w_start, w_end)
            - idx (int): Index of the current chunk
            - mode (str): Interpolation mode for grid sampling
            - padding_mode (str): Padding mode for grid sampling
            - align_corners (bool): Whether to align corners in grid sampling

    Returns:
        tuple: A tuple containing:
            - idx (int): Index of the processed chunk
            - result (numpy.ndarray): Processed chunk after grid sampling
    """
    input_tensor, grid, n_range, h_range, w_range, idx, mode, padding_mode, align_corners = args
    n_start, n_end = n_range
    h_start, h_end = h_range
    w_start, w_end = w_range

    input_block = input_tensor[n_start:n_end, :, :, :]
    grid_block = grid[n_start:n_end, h_start:h_end, w_start:w_end, :]
    with torch.no_grad():
        result = F.grid_sample(input_block, grid_block, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    
    return (idx, result.detach().cpu().numpy())

def dask_grid_sample(
        da_NCHW: Dask_image_NCHW, 
        grid: Tensor_grid_M, 
        mode='bilinear', 
        padding_mode='zeros',
        align_corners=True, 
        max_workers=4
    ):
    """Apply grid sampling to a dask array layer by layer.

    This function processes each channel of the input dask array separately using grid sampling,
    which is useful for large images that need to be processed in a memory-efficient way.

    Args:
        da_NCHW (Dask_image_NCHW): Input dask array in NCHW format
        grid (Tensor_grid_M): Sampling grid tensor
        mode (str, optional): Interpolation mode. Defaults to 'bilinear'.
        padding_mode (str, optional): Padding mode for sampling. Defaults to 'zeros'.
        align_corners (bool, optional): Whether to align corners in grid sampling. Defaults to True.
        max_workers (int, optional): Maximum number of worker processes. Defaults to 4.

    Returns:
        da.Array: Processed dask array after grid sampling
    """
    l_dasks_out = []
    for i_layer in range(da_NCHW.shape[1]):
        da_layer = da_NCHW[:, i_layer:i_layer+1, :, :]
        tensor_layer = torch.from_numpy(da_layer.compute()).float()
        output_NCHW_layer = mp_array_grid_sample(tensor_layer, grid.detach(), mode=mode, padding_mode=padding_mode, align_corners=align_corners, max_workers=max_workers)
        l_dasks_out.append(output_NCHW_layer)
    
    result = da.concatenate(l_dasks_out, axis=1)
    return result

# @profile
def mp_array_grid_sample(
        input_tensor: Tensor_image_NCHW, 
        grid: Tensor_grid_M, 
        chunks=(1, 'auto', 1024, 1024), 
        mode='bilinear', 
        padding_mode='zeros',
        align_corners=True, 
        max_workers=4
    ) -> da.Array:
    """Apply grid sampling to a large tensor using multiprocessing and Dask.

    This function implements the same functionality as torch.nn.functional.grid_sample but uses
    multiprocessing and Dask arrays to handle large inputs efficiently. The input tensor is
    processed in chunks to reduce memory usage.

    Args:
        input_tensor (Tensor_image_NCHW): Input tensor in NCHW format
        grid (Tensor_grid_M): Sampling grid tensor of shape (N, H_out, W_out, 2)
        chunks (tuple, optional): Chunk sizes for processing (N, C, H, W). Defaults to (1, 'auto', 1024, 1024).
        mode (str, optional): Interpolation mode ('bilinear' or 'nearest'). Defaults to 'bilinear'.
        padding_mode (str, optional): Padding mode ('zeros', 'border', or 'reflection'). Defaults to 'zeros'.
        align_corners (bool, optional): Whether to align corners in grid sampling. Defaults to True.
        max_workers (int, optional): Maximum number of concurrent processes. Defaults to 4.

    Returns:
        da.Array: Processed dask array after grid sampling, with shape matching the input grid
    """
    N, C, H_in, W_in = input_tensor.shape
    _, H_out, W_out, _ = grid.shape

    chunks_N = chunks[0] if chunks[0] != -1 else N
    chunks_H = chunks[2] if chunks[2] != -1 else H_out
    chunks_W = chunks[3] if chunks[3] != -1 else W_out

    N_blocks = (N + chunks_N - 1) // chunks_N
    H_blocks = (H_out + chunks_H - 1) // chunks_H
    W_blocks = (W_out + chunks_W - 1) // chunks_W

    total_tasks = N_blocks * H_blocks * W_blocks

    tasks = []
    idx = 0

    for n in range(N_blocks):
        n_start = n * chunks_N
        n_end = min((n + 1) * chunks_N, N)
        for h in range(H_blocks):
            h_start = h * chunks_H
            h_end = min((h + 1) * chunks_H, H_out)
            for w in range(W_blocks):
                w_start = w * chunks_W
                w_end = min((w + 1) * chunks_W, W_out)

                n_range = (n_start, n_end)
                h_range = (h_start, h_end)
                w_range = (w_start, w_end)

                tasks.append((input_tensor, grid, n_range, h_range, w_range, idx, mode, padding_mode, align_corners))
                idx += 1

    output_shape = (N, C, H_out, W_out)
    output_chunks = (chunks_N, C, chunks_H, chunks_W)
    output_dtype = np.float32

    output = da.zeros(shape=output_shape, chunks=output_chunks, dtype=output_dtype)

    with tqdm(total=total_tasks, desc="Processing") as pbar:
        with mp.Pool(processes=max_workers) as pool:
            results = pool.imap_unordered(worker, tasks)

            for idx_received, result_block in results:
                n_idx = idx_received // (H_blocks * W_blocks)
                hw_idx = idx_received % (H_blocks * W_blocks)
                h_idx = hw_idx // W_blocks
                w_idx = hw_idx % W_blocks

                n_start = n_idx * chunks_N
                n_end = min((n_idx + 1) * chunks_N, N)
                h_start = h_idx * chunks_H
                h_end = min((h_idx + 1) * chunks_H, H_out)
                w_start = w_idx * chunks_W
                w_end = min((w_idx + 1) * chunks_W, W_out)

                output_block_slices = (slice(n_start, n_end), slice(None), slice(h_start, h_end), slice(w_start, w_end))
                output[output_block_slices] = da.from_array(result_block, chunks=result_block.shape)
                pbar.update(1)


    return output


if __name__ == '__main__':
    mp.set_start_method('spawn')

    input_tensor = torch.randn(1, 1, 51200, 51200)
    grid = torch.randn(1, 51200, 51200, 2)

    n_cpus = mp.cpu_count()
    start_time = time.time()
    result = mp_array_grid_sample(input_tensor, grid, chunks=(1, 'auto', 1024, 1024), max_workers=n_cpus)
    end_time = time.time()
    print('mp_array_grid_sample time:', end_time - start_time)
    result = torch.from_numpy(result.compute())
    print('shape:', result.shape)

    start_time = time.time()
    result_direct = F.grid_sample(input_tensor, grid, align_corners=True)
    end_time = time.time()
    print('F.grid_sample time:', end_time - start_time)

    
    print('difference:', (result - result_direct).abs().max().item())
    