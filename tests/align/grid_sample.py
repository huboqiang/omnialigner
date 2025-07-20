import time
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm import tqdm
# from memory_profiler import profile
import torch
import torch.nn.functional as F
import multiprocessing as mp
from tqdm import tqdm
import dask.array as da
import numpy as np

def worker(args):
    input_tensor, grid, n_range, h_range, w_range, idx, align_corners = args
    n_start, n_end = n_range
    h_start, h_end = h_range
    w_start, w_end = w_range

    input_block = input_tensor[n_start:n_end, :, :, :]
    grid_block = grid[n_start:n_end, h_start:h_end, w_start:w_end, :]

    result = F.grid_sample(input_block, grid_block, align_corners=align_corners)
    return (idx, result.cpu().numpy())  # 将结果转换为 NumPy 数组

# @profile
def mp_array_grid_sample(input_tensor, grid, chunks=(1, 'auto', 1024, 1024), align_corners=True, max_workers=4):
    """
    使用 multiprocessing 实现的并行版本，并利用 Dask 将输出数据写入磁盘以减少内存使用

    Parameters:
    - input_tensor: 输入张量，形状为 (N, C, H_in, W_in)
    - grid: 采样网格，形状为 (N, H_out, W_out, 2)
    - chunks: 定义如何分块数据的元组，(chunks_N, 'auto', chunks_H, chunks_W)
    - align_corners: 与 F.grid_sample 的参数相同
    - max_workers: 最大同时运行的进程数
    """
    N, C, H_in, W_in = input_tensor.shape
    _, H_out, W_out, _ = grid.shape

    # 解析 chunks 参数
    chunks_N = chunks[0] if chunks[0] != -1 else N
    chunks_H = chunks[2] if chunks[2] != -1 else H_out
    chunks_W = chunks[3] if chunks[3] != -1 else W_out

    # 计算每个维度的块数
    N_blocks = (N + chunks_N - 1) // chunks_N
    H_blocks = (H_out + chunks_H - 1) // chunks_H
    W_blocks = (W_out + chunks_W - 1) // chunks_W

    total_tasks = N_blocks * H_blocks * W_blocks  # 总的任务数量

    # 准备任务列表
    tasks = []
    idx = 0  # 用于标记块的索引

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

                tasks.append((input_tensor, grid, n_range, h_range, w_range, idx, align_corners))
                idx += 1

    # 创建 Dask 数组，用于将结果存储到磁盘
    output_shape = (N, C, H_out, W_out)
    output_chunks = (chunks_N, C, chunks_H, chunks_W)
    output_dtype = np.float32

    # 创建一个空的 Dask 数组，使用 Zarr 存储到磁盘
    output = da.zeros(shape=output_shape, chunks=output_chunks, dtype=output_dtype)
    output_zarr_path = 'output.zarr'  # 输出文件路径

    # 创建进度条
    with tqdm(total=total_tasks, desc="Processing") as pbar:
        with mp.Pool(processes=max_workers) as pool:
            results = pool.imap_unordered(worker, tasks)

            for idx_received, result_block in results:
                # 计算块的位置
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

                # 将结果写入到 Dask 数组对应的位置
                output_block_slices = (slice(n_start, n_end), slice(None), slice(h_start, h_end), slice(w_start, w_end))
                output[output_block_slices] = da.from_array(result_block, chunks=result_block.shape)

                # 更新进度条
                pbar.update(1)

    # 将 Dask 数组写入磁盘
    da.to_zarr(output, output_zarr_path, overwrite=True)

    # 如果需要返回结果，可以从磁盘读取
    # result = da.from_zarr(output_zarr_path)
    # return torch.from_numpy(result.compute())

    # 或者，如果不需要返回，可以直接返回输出文件路径
    return output


if __name__ == '__main__':
    mp.set_start_method('spawn')

    # 示例输入
    input_tensor = torch.randn(1, 1, 51200, 51200)
    grid = torch.randn(1, 51200, 51200, 2)

    n_cpus = mp.cpu_count()
    start_time = time.time()
    result = mp_array_grid_sample(input_tensor, grid, chunks=(1, 'auto', 1024, 1024), max_workers=n_cpus)
    end_time = time.time()
    print('mp_array_grid_sample time:', end_time - start_time)
    result = torch.from_numpy(result.compute())
    print('结果形状：', result.shape)

    start_time = time.time()
    result_direct = F.grid_sample(input_tensor, grid, align_corners=True)
    end_time = time.time()
    print('F.grid_sample time:', end_time - start_time)

    
    print('结果与直接调用grid_sample的差异：', (result - result_direct).abs().max().item())
    