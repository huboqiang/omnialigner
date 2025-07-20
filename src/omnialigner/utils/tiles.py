from typing import List, Tuple

import torch
import dask.array as da        # noqa:  F401  (needed for isinstance check)
import numpy as np

from omnialigner.dtypes import Tuple_bbox

def _torch_merge(
    l_tiles: List[Tuple_bbox],
    l_disp:  List[torch.Tensor],
    H: int,
    W: int,
) -> torch.Tensor:
    """internal – PyTorch backend"""
    N, C = l_disp[0].shape[0], l_disp[0].shape[-1]
    dtype  = l_disp[0].dtype
    device = l_disp[0].device

    disp_sum = torch.zeros((N, H, W, C), dtype=dtype, device=device)
    cnt_sum  = torch.zeros((1, H, W, 1),   dtype=torch.int32, device=device)

    for (x_beg, y_beg, x_end, y_end), d in zip(l_tiles, l_disp):
        # allow (Ht, Wt, C) by auto-broadcasting batch dim
        if d.ndim == 3:
            d = d.unsqueeze(0)

        d = d * torch.tensor([x_end-x_beg, y_end-y_beg], dtype=dtype, device=device).reshape(1, 1, 1, 2)
        disp_sum[:, y_beg:y_end, x_beg:x_end] += d
        cnt_sum [:, y_beg:y_end, x_beg:x_end] += 1

    disp_raw = disp_sum / cnt_sum.float().clamp_min_(1)   # ÷0 → 0
    disp_raw = disp_raw / torch.tensor([W, H], dtype=dtype, device=device).reshape(1, 1, 1, 2)
    return disp_raw


def _dask_merge(
    l_tiles: List[Tuple_bbox],
    l_disp:  List["da.Array"],
    H: int,
    W: int,
    hw_chunk: int = 512,               # 目标画布高/宽方向 chunk 大小
) -> "da.Array":
    if not l_disp:
        raise ValueError("`l_disp` 不能为空")

    # ----- 基本 meta ---------------------------------------------------------
    x0 = l_disp[0]
    if x0.ndim == 3:                   # (Ht,Wt,C) → (1,…) for consistency
        x0 = x0[None, ...]
    N, _, _, C = x0.shape

    disp_sum = da.zeros((N, H, W, C),
                        chunks=(1, hw_chunk, hw_chunk, C),
                        dtype=x0.dtype)
    cnt_sum  = da.zeros((1, H, W, 1),
                        chunks=(1, hw_chunk, hw_chunk, 1),
                        dtype=np.int32)

    # ----- 单块注入 ----------------------------------------------------------
    def _inject(chunk, tile, x_beg, y_beg, tile_w, tile_h, *, block_info=None):
        """把 tile 加到对应 output-chunk；无交集则直接返回 chunk。"""
        gh0, gh1 = block_info[None]["array-location"][1]   # global H [beg,end)
        gw0, gw1 = block_info[None]["array-location"][2]   # global W [beg,end)

        i0 = y_beg
        j0 = x_beg
        th0 = max(gh0, i0)
        tw0 = max(gw0, j0)
        th1 = min(gh1, i0 + tile.shape[1])
        tw1 = min(gw1, j0 + tile.shape[2])

        if th0 >= th1 or tw0 >= tw1:                       # 无交集
            return chunk

        if tile.dtype != chunk.dtype:                      # dtype 对齐
            tile = tile.astype(chunk.dtype)

        out = chunk.copy()

        # chunk 内局部坐标
        ch0 = th0 - gh0
        cw0 = tw0 - gw0
        ch1 = ch0 + (th1 - th0)
        cw1 = cw0 + (tw1 - tw0)

        # tile 内局部坐标
        t_h0 = th0 - i0
        t_w0 = tw0 - j0

        # 关键修改：将位移按照 tile 尺寸缩放到像素值
        tile_scaled = tile * np.array([tile_w, tile_h], dtype=tile.dtype).reshape(1, 1, 1, 2)
        
        out[:, ch0:ch1, cw0:cw1, :] += tile_scaled[:, t_h0:t_h0 + (ch1 - ch0),
                                                   t_w0:t_w0 + (cw1 - cw0), :]
        return out

    # ----- 逐 tile 合并 ------------------------------------------------------
    for (x_beg, y_beg, x_end, y_end), t in zip(l_tiles, l_disp):
        if t.ndim == 3:
            t = t[None, ...]                                # (1,Ht,Wt,C)

        # 计算 tile 尺寸
        tile_w = x_end - x_beg
        tile_h = y_end - y_beg

        ones = da.ones_like(t[..., :1], dtype=np.int32)     # (N,Ht,Wt,1)

        disp_sum = da.map_blocks(
            _inject, disp_sum, t, x_beg, y_beg, tile_w, tile_h,
            dtype=disp_sum.dtype, chunks=disp_sum.chunks,
        )
        cnt_sum = da.map_blocks(
            _inject, cnt_sum, ones, x_beg, y_beg, tile_w, tile_h,
            dtype=cnt_sum.dtype,  chunks=cnt_sum.chunks,
        )

    # 平均后归一化回比例值
    disp_averaged = disp_sum / da.maximum(cnt_sum, 1)
    disp_normalized = disp_averaged / da.array([W, H], dtype=disp_averaged.dtype).reshape(1, 1, 1, 2)
    
    return disp_normalized
# -----------------------------------------------------------------------------


def merge_from_overlapped_tiles(
    l_tiles_extend: List[Tuple_bbox],
    l_disp:        List[torch.Tensor] | List["da.Array"],
    H: int = -1,
    W: int = -1,
):
    """
    Merge overlapped displacement tiles (PyTorch **or** Dask) into one map.

    Parameters
    ----------
    l_tiles_extend : list[[x_beg, y_beg, x_end, y_end]]
        End-exclusive tile coordinates in the full image.
    l_disp : list[Tensor] | list[dask.array.Array]
        Displacement tiles – shape (N, Ht, Wt, [y, x]) **or** (Ht, Wt, [y, x]).
    H, W : int
        Full height / width; -1 → infer from *l_tiles_extend*.

    Returns
    -------
    disp_raw : Tensor | dask.array
        (N, H, W, C) averaged displacement; untouched pixels are **0**.
    """
    if not l_disp:
        raise ValueError("`l_disp` must contain at least one tile")

    # -------------------------------------------------------------------------
    #   determine canvas size
    # -------------------------------------------------------------------------
    H = max(t[2] for t in l_tiles_extend) if H == -1 else H
    W = max(t[3] for t in l_tiles_extend) if W == -1 else W

    first = l_disp[0]
    is_dask = isinstance(first, da.Array)

    if is_dask:
        return _dask_merge(l_tiles_extend, l_disp, H, W)

    return _torch_merge(l_tiles_extend, l_disp, H, W)
