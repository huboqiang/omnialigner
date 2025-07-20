from typing import List, Union

import cv2
from dask import delayed
import dask.array as da
import numpy as np
import torch
import torch.nn.functional as F
from tifffile import TiffFile, TiffWriter
import tifffile
from omnialigner.logging import logger as logging


def read_ome_tiff(file_name: str, i_page: int=0, i_level: int=-1, l_channels: List[int]=None, **kwargs) -> da.Array:
    """Read an OME-TIFF file and return it as a dask array.

    Args:
        file_name: Path to the OME-TIFF file.
        i_page: Page index to read from. Defaults to 0.
        i_level: Resolution level to read. Defaults to -1 (highest resolution).
        l_channels: List of channel indices to read. If None, reads all channels.
        **kwargs: Additional arguments:
            - axes: List specifying axis order for output array.

    Returns:
        da.Array: Dask array containing the image data with shape (H, W, C).

    Note:
        The function handles multi-resolution and multi-channel OME-TIFF files.
        Output array dimensions depend on the input file structure and kwargs.
    """
    with TiffFile(file_name) as handle:
        l_dasks = []
        logging.info(f"reading {file_name}, handle.series[{i_page}].levels[{i_level}]")
        store = handle.series[i_page].levels[i_level]
        if l_channels is None:
            l_channels = range(len(store))

        l_channels_used = range(len(store))
        for i_channel in l_channels_used:
            da_arr = da.from_zarr(store[i_channel].aszarr())
            l_dasks.append(da_arr)

        da_out = da.stack(l_dasks, axis=2)
        if len(da_out.shape) == 4 and da_out.shape[2] ==1:
            kwargs["axes"] = [0, 1, 3]

        if "axes" in kwargs:
            da_out = select_axes(da_out, kwargs["axes"])

        da_out = da_out[:, :, np.array(l_channels)]
        return da_out

def select_axes(array: da.Array, axes: list) -> da.Array:
    """Select and reorder axes in a dask array.

    Args:
        array: Input dask array.
        axes: List of axis indices to select and their new order.

    Returns:
        da.Array: Reordered dask array containing only the selected axes.
    """
    moved = da.moveaxis(array, source=axes, destination=range(len(axes)))
    slices = tuple(slice(None) if i < len(axes) else 0 
                  for i in range(array.ndim))
    return moved[slices]


def read_qptiff_dask(file_name, key=0, component="0") -> da.Array:
    """Read a QPTIFF file into a dask array.

    Args:
        file_name: Path to the QPTIFF file.
        key: Series key to read. Defaults to 0.
        component: Component identifier. Defaults to "0".

    Returns:
        da.Array: Dask array containing the image data.
    """
    handle = TiffFile(file_name)
    kwargs = {}
    if key is not None:
        kwargs["key"] = key

    store = handle.aszarr(**kwargs)
    kwargs = {}
    if component is not None:
        kwargs["component"] = component

    da_arr = da.from_zarr(store, **kwargs)
    return da_arr


def write_NCHW_dask_ometiff(
        da_array: Union[da.Array, np.array], 
        file_name: str, 
        chunks: tuple=(512, 512), 
        compression: str="lzw", 
        photometric: str="rgb"
    ):
    """Write a dask array to an OME-TIFF file in NCHW format.

    Args:
        da_array: Input array in NCHW format (batch, channels, height, width).
        file_name: Output file path.
        chunks: Tile size for chunked writing. Defaults to (512, 512).
        compression: Compression method. Defaults to "lzw".
        photometric: Color interpretation. Use "rgb" for HE, "minisblack" for IHC.

    Note:
        The function includes PerkinElmer-QPI specific metadata in the output file.
    """
    with TiffWriter(file_name, bigtiff=True, ome=True) as tif:
        options = dict(
            tile=chunks,
            compression=compression,
            photometric=photometric,
        )
        metadata={
            'axes': 'ZCYX',
            "ImageWidth" : da_array.shape[2:][::-1][0],
            "ImageLength" : da_array.shape[2:][::-1][1],
            "XResolution": 107598 / 4294967295,
            "YResolution": 107598 / 4294967295,
            "ResolutionUnit": "CENTIMETER",
            "Software" : "PerkinElmer-QPI"
        }
        tif.write(da_array, **options, metadata=metadata)

def write_qptiff_2d(
        file_name: str, 
        np_arr: Union[np.array, da.Array], 
        sizes: List[int]=None, 
        photometric="rgb"
    ):
    """Write a 2D image to QPTIFF format with multiple resolution levels.

    Args:
        file_name: Output file path.
        np_arr: Input image in HWC format.
        sizes: List of downsampling factors for resolution pyramid. 
            Defaults to [1, 4, 8, 40].
        photometric: Color interpretation. Use "rgb" for HE, "minisblack" for IHC.

    Note:
        Creates a multi-resolution pyramid with specified downsampling factors.
        Includes PerkinElmer-QPI specific metadata.
    """
    if sizes is None:
        print("Using default sizes [1, 5, 10, 40]")
        # sizes = [1, 5, 10, 40]
        sizes = [1, 4, 8, 40]
    
    print("sizes:", sizes)
    if isinstance(np_arr, da.Array):
        np_arr = np_arr.compute()

    with TiffWriter(file_name, bigtiff=True, ome=True) as tif:
        for level, size in enumerate(sizes):
            new_size = (np_arr.shape[1] // size, np_arr.shape[0] // size)
            options = dict(
                tile=(512, 512),
                compression='lzw',
                photometric=photometric,
            )
            metadata = {
                'axes': 'YXC',
                "ImageWidth": new_size[0],
                "ImageLength": new_size[1],
                "XResolution": 107598 / 4294967295,
                "YResolution": 107598 / 4294967295,
                "ResolutionUnit": "CENTIMETER",
                "Software": "PerkinElmer-QPI"
            }
            logging.info(f"writing level {level} with size {new_size}")
            if level == 0:
                tif.write(np_arr, **options, metadata=metadata)
                continue
            
            image = cv2.resize(np_arr, new_size)
            tif.write(image, **options, metadata=metadata, subfiletype=1)


def chunked_grid_sample_dask(dask_input, grid, N_chunk_size, C_chunk_size, align_corners=True):
    """Apply grid_sample to a dask array in chunks to optimize memory usage.

    Processes the input array in chunks along batch (N) and channel (C) dimensions
    to handle large arrays efficiently.

    Args:
        dask_input: Input dask array of shape (N, C, H, W).
        grid: Sampling grid tensor of shape (N, H_out, W_out, 2).
        N_chunk_size: Number of samples to process per batch chunk.
        C_chunk_size: Number of channels to process per channel chunk.
        align_corners: Whether to align corners in grid sampling. Defaults to True.

    Returns:
        torch.Tensor: Resampled tensor of shape (N, C, H_out, W_out).

    Note:
        Input array should be uint8 or float32 dtype.
        Grid coordinates should be in the range [-1, 1].
    """
    N, C, H, W = dask_input.shape
    N_out, H_out, W_out, _ = grid.shape

    delayed_arrays = []
    chunks_N = []
    chunks_C = []

    for n_start in range(0, N, N_chunk_size):
        n_end = min(n_start + N_chunk_size, N)
        N_chunk = n_end - n_start
        chunks_N.append(N_chunk)
        delayed_arrays_C = []
        for c_start in range(0, C, C_chunk_size):
            c_end = min(c_start + C_chunk_size, C)
            C_chunk = c_end - c_start
            if n_start == 0:
                chunks_C.append(C_chunk)

            def compute_chunk(n_start=n_start, n_end=n_end, c_start=c_start, c_end=c_end):
                dask_input_chunk = dask_input[n_start:n_end, c_start:c_end, :, :]
                input_chunk_np = dask_input_chunk.compute()
                input_chunk = torch.from_numpy(input_chunk_np).float()
                grid_chunk = grid[n_start:n_end]
                output_chunk = F.grid_sample(input_chunk, grid_chunk, align_corners=align_corners)
                output_chunk_np = output_chunk.cpu().detach().numpy()
                return output_chunk_np

            delayed_compute = delayed(compute_chunk)()
            shape = (N_chunk, C_chunk, H_out, W_out)
            dtype = np.float32
            chunk_array = da.from_delayed(delayed_compute, shape=shape, dtype=dtype)

            delayed_arrays_C.append(chunk_array)

        delayed_arrays_N = da.concatenate(delayed_arrays_C, axis=1)
        delayed_arrays.append(delayed_arrays_N)

    output_dask_array = da.concatenate(delayed_arrays, axis=0)

    output_dask_array = output_dask_array.rechunk((tuple(chunks_N), tuple(chunks_C), H_out, W_out))

    return output_dask_array

def chunked_grid_sample_core(input_tensor, grid, N_chunk_size, C_chunk_size, align_corners=True, input_getter=None):
    """Core function to apply grid_sample to a tensor input in chunks.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (N, C, H, W).
        grid (torch.Tensor): Grid tensor of shape (N, H_out, W_out, 2).
        N_chunk_size (int): Chunk size along the N dimension.
        C_chunk_size (int): Chunk size along the C dimension.
        align_corners (bool, optional): Boolean flag for grid_sample. Defaults to True.
        input_getter (callable, optional): Optional function to retrieve input chunks.

    Returns:
        torch.Tensor: Output tensor of shape (N, C, H_out, W_out).
    """
    N, C, H, W = input_tensor.shape
    N_out, H_out, W_out, _ = grid.shape

    assert N == N_out, "Input tensor and grid must have the same batch size"

    output = torch.zeros(N, C, H_out, W_out)

    for n_start in range(0, N, N_chunk_size):
        n_end = min(n_start + N_chunk_size, N)

        grid_chunk = grid[n_start:n_end]

        for c_start in range(0, C, C_chunk_size):
            c_end = min(c_start + C_chunk_size, C)

            if input_getter is not None:
                input_chunk = input_getter(n_start, n_end, c_start, c_end)
            else:
                input_chunk = input_tensor[n_start:n_end, c_start:c_end, :, :]

            input_chunk = input_chunk

            output_chunk = F.grid_sample(input_chunk, grid_chunk, align_corners=align_corners)

            output_chunk = output_chunk.cpu()

            output[n_start:n_end, c_start:c_end, :, :] = output_chunk

    return output

def chunked_grid_sample(input_tensor, grid, N_chunk_size, C_chunk_size, align_corners=True):
    """Wrapper function to apply grid_sample in chunks without a custom input getter.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (N, C, H, W).
        grid (torch.Tensor): Grid tensor of shape (N, H_out, W_out, 2).
        N_chunk_size (int): Chunk size along the N dimension.
        C_chunk_size (int): Chunk size along the C dimension.
        align_corners (bool, optional): Boolean flag for grid_sample. Defaults to True.

    Returns:
        torch.Tensor: Output tensor of shape (N, C, H_out, W_out).
    """
    return chunked_grid_sample_core(
        input_tensor=input_tensor,
        grid=grid,
        N_chunk_size=N_chunk_size,
        C_chunk_size=C_chunk_size,
        align_corners=align_corners,
        input_getter=None
    )


def write_spatialdata(file_name, key=0, component="0"):
    pass


def write_NCHW_dask_ometiff_V2(
    da_array: Union[da.Array, np.ndarray],
    file_name: str,
    chunks: tuple = (512, 512),
    compression: str = "lzw",
    sizes: List[int] = None,
    photometric: str = "minisblack",
    channel_order: List[int] = None
):
    """Write a dask array as a pyramidal OME-TIFF with multiple resolution levels.

    Args:
        da_array: Input array of shape (1, N_channels, H, W).
        file_name: Output file path (recommended to end with .ome.tif).
        chunks: Tile size for chunked writing. Defaults to (512, 512).
        compression: TIFF compression algorithm. Defaults to "lzw".
        sizes: List of downsampling factors. Defaults to [1, 4, 8, 40].
        photometric: Color interpretation. Use "rgb" for HE, "minisblack" for IHC.
        channel_order: Optional list specifying channel order in output.

    Note:
        Creates a multi-resolution pyramid with specified downsampling factors.
        Includes PerkinElmer-QPI specific metadata.
    """
    from tqdm import tqdm

    if sizes is None:
        sizes = [1, 4, 8, 40]

    _, C, H, W = da_array.shape
    # levels = [da_array[0, i].compute() for i in range(da_array.shape[1])]

    
    # OME-XML metadata（这里只示例基本字段，按需补充 Channel/Plane 等信息）
    metadata = {
        "axes": "YX",  # Z 方向是金字塔级别
        "XResolution": 107598 / 4294967295,
        "YResolution": 107598 / 4294967295,
        "ResolutionUnit": "CENTIMETER",
        "Software": "PerkinElmer-QPI",
    }
    options = dict(
        tile=chunks,
        compression=compression,
        photometric=photometric,
        metadata=metadata
    )
    if channel_order is None:
        channel_order = list(range(C))

    with TiffWriter(file_name, bigtiff=True, ome=True) as tif:
        for i_size, size in tqdm(enumerate(sizes)):
            for i_channel in channel_order:
                if i_size > 0 or i_channel > 0:
                    options['subfiletype'] = 1

                img = da_array[0, i_channel]
                if isinstance(img, da.Array):
                    img = img.compute()

                if i_size > 0:
                    new_size = (img.shape[1] // size, img.shape[0] // size)
                    img = cv2.resize(img, new_size)

                tif.write(img, **options)



def write_NCHW_dask_ometiff_V3(
    da_array: Union[da.Array, np.ndarray],
    file_name: str,
    chunks: tuple = (512, 512),
    compression: str = "lzw",
    sizes: List[int] = None,
    photometric: str = "minisblack",
    channel_order: List[int] = None
):
    """Write a dask array as a pyramidal OME-TIFF with interleaved channels.

    Similar to V2, but writes channels in interleaved format (NHWC) instead of
    planar format (NCHW).

    Args:
        da_array: Input array of shape (1, N_channels, H, W).
        file_name: Output file path (recommended to end with .ome.tif).
        chunks: Tile size for chunked writing. Defaults to (512, 512).
        compression: TIFF compression algorithm. Defaults to "lzw".
        sizes: List of downsampling factors. Defaults to [1, 4, 8, 40].
        photometric: Color interpretation. Use "rgb" for HE, "minisblack" for IHC.
        channel_order: Optional list specifying channel order in output.

    Note:
        Creates a multi-resolution pyramid with specified downsampling factors.
        Includes PerkinElmer-QPI specific metadata.
        Uses interleaved channel format for better compatibility with some readers.
    """
    from tqdm import tqdm

    if sizes is None:
        sizes = [1, 4, 8, 40]

    _, C, H, W = da_array.shape
    # levels = [da_array[0, i].compute() for i in range(da_array.shape[1])]

    
    # OME-XML metadata（这里只示例基本字段，按需补充 Channel/Plane 等信息）
    metadata = {
        "axes": "YXC",
        "XResolution": 107598 / 4294967295,
        "YResolution": 107598 / 4294967295,
        "ResolutionUnit": "CENTIMETER",
        "Software": "PerkinElmer-QPI",
    }
    options = dict(
        tile=chunks,
        compression=compression,
        photometric=photometric,
        metadata=metadata,
        planarconfig="contig"
    )
    if channel_order is None:
        channel_order = list(range(C))

    with TiffWriter(file_name, bigtiff=True, ome=True) as tif:
        for i_size, size in tqdm(enumerate(sizes)):
            if i_size > 0:
                options['subfiletype'] = 1

            img = np.moveaxis(da_array[0, :], 0, 2)[:, :, np.array(channel_order)]
            if isinstance(img, da.Array):
                img = img.compute()

            if i_size > 0:
                new_size = (img.shape[1] // size, img.shape[0] // size)
                img = cv2.resize(img, new_size)

            tif.write(img, **options)