import os
from typing import Dict
import functools

import numpy as np
import dask.array as da
import torch
import scanpy as sc
import torch.nn.functional as F
from tqdm import tqdm

from omnialigner.configs import load_config
from omnialigner.cache_files import ProjInfo, StageTag, StageSampleTag
from omnialigner.datasets.datasets import read_file
from omnialigner.dtypes import ConfigFile, Tensor_tfrs, Tensor_image_NCHW, Dask_image_NCHW, Dask_image_HWC, DataType
from omnialigner.logging import logger #as logging
from omnialigner.utils.image_viz import rgb_he_to_gray
from omnialigner.utils.field_transform import get_sampling_grid

class Omni3D(object):
    """A comprehensive class for managing and processing 3D image data in the OmniAligner pipeline.

    The Omni3D class serves as the central data management system for the OmniAligner
    project, handling everything from raw image loading to multi-stage processing and
    caching. It maintains the state of image processing across different stages and
    provides utilities for data transformation and alignment.

    Key Features:
        - Manages project configuration and cache states
        - Handles raw tiff image loading with zoom level support
        - Provides access to processed data at various pipeline stages
        - Supports both grayscale and color image processing
        - Implements efficient caching mechanisms for intermediate results
        - Handles special cases for different image types (e.g., H&E staining)

    Attributes:
        config (Dict): Project configuration dictionary containing:
            - Dataset parameters
            - Processing settings
            - Cache configurations
        proj_info (ProjInfo): Project information object
        CACHE_TAG (StageTag): Current processing stage tag
        tag (DataType): Data type identifier (RAW/GRAY)
        padded_tensor (Tensor_image_NCHW): Cached padded image tensor
        method (str): Processing method identifier
        transform_engine (str): Transformation engine type
        zoom_level (int): Current zoom level for image loading
        max_size (int): Maximum size constraint for images
        l_scales (List[float]): List of scale factors
        sizes (List[float]): Computed sizes based on scales
    """
    def __init__(self,
                 config_info: ConfigFile|Dict=None,
                 **kwargs):
        """Initialize an Omni3D instance with configuration and optional parameters.

        This constructor sets up the Omni3D object by loading configuration,
        initializing processing parameters, and determining the current cache state
        of the project.

        Args:
            config_info (ConfigFile|Dict, optional): Configuration information either as
                a config file path or dictionary. If None, uses default settings.
            **kwargs: Additional configuration parameters that override config_info:
                - project (str): Project identifier
                - group (str): Group identifier
                - version (str): Version identifier
                - nonrigid (bool): Whether to use nonrigid transformation
                - zoom_level (int): Initial zoom level
                - tag (str): Data type tag
                - method (str): Processing method
                - transform_engine (str): Transformation engine
                - min_disp (float): Minimum displacement
                - max_size (int): Maximum image size
                - l_scales (List[float]): Scale factors
                - file_data (str): Data file path
                - raw_img_prefix (str): Raw image file prefix
                - raw_is_tiff (bool): Whether raw images are TIFF
                - plt_row_col (List[int]): Plot dimensions
                - plt_figsize (Tuple[int, int]): Plot figure size

        Note:
            - The constructor automatically detects and loads cached data
            - Cache stages are checked in sequence (PAD -> STACK -> AFFINE)
            - Configuration parameters can be overridden by kwargs
            - Project paths are automatically expanded
        """
        if config_info is not None:
            self.config = load_config(config_info)
        
        self._load_dataset_params(**kwargs)        
        for current_stage in StageTag:
            FILE_CACHE = current_stage.get_file_name(projInfo=self.proj_info, check_exist=True)
            if FILE_CACHE is None:
                break
            
            self.CACHE_TAG = current_stage
            logger.info(f"project {self.project}, group {self.group}, version {self.version}, all cached files found in stage: {self.CACHE_TAG}")

        
        if self.CACHE_TAG >= StageTag.PAD:
            self._load_pad()
        
        if self.CACHE_TAG >= StageTag.STACK:
            self._load_stack()
        
        if self.CACHE_TAG >= StageTag.AFFINE:
            self._load_align()
        
    
    def load_tiff(self,
            i_layer:int,
            zoom_level:int=0,
            is_raw:bool=True,
            resize_to_20x:bool=False) -> Dask_image_HWC:
        """Load and process a raw TIFF image for a specific layer.

        This method handles the loading of raw TIFF images with support for
        different zoom levels and optional preprocessing. It's equivalent to
        using `om.pp.read_image(..., is_tiff=True, crop_image=False)` and
        applying `rgb_he_to_gray` if the data type is set to grayscale.

        Args:
            i_layer (int): Layer index to load.
            zoom_level (int, optional): Zoom level for image loading.
                Higher values mean more zoomed out. Defaults to 0.
            is_raw (bool, optional): Whether to load raw unprocessed image.
                Defaults to True.
            resize_to_20x (bool, optional): Whether to resize image to 20x
                magnification. Defaults to False.

        Returns:
            da.Array: Loaded image as a dask array with shape [H, W, C].

        Raises:
            FileNotFoundError: If the raw image file is not found at the
                expected location.

        Note:
            - For H&E images in grayscale mode, special color conversion is applied
            - Uses dask arrays for memory-efficient processing
            - Supports various zoom levels through OME-TIFF format
            - Additional processing parameters can be specified in config["datasets"]
            - Results are returned as uint8 type arrays
        """
        sample = self.proj_info.get_sample_name(i_layer=i_layer)
        dict_file_name = StageSampleTag.RAW.get_file_name(i_layer=i_layer, projInfo=self.proj_info)
        if dict_file_name is None:
            dict_file_name = StageSampleTag.RAW.get_file_name(i_layer=i_layer, projInfo=self.proj_info, check_exist=False)
            out_file = dict_file_name["raw"]
            err_msg = f"""Raw image for layer {i_layer}, {sample} of {self.project}/{self.group}/{self.version} not found
            You can generate it by running:
             ```
            da_img, np_coords = om.pp.read_image(RAW_FILE, is_tiff=False, crop_image=False)
            om.tl.write_qptiff_2d({out_file}, da_img.compute())
            ```
            """
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        file = dict_file_name["raw"]
        kwargs_data = self.config["datasets"]
        kwargs_data["zoom_level"] = zoom_level
        kwargs_data["is_raw"] = is_raw
        kwargs_data["resize_to_20x"] = resize_to_20x
        kwargs_read_ome_tiff = kwargs_data.get("read_ome_tiff", {})
        for k,v in kwargs_read_ome_tiff.items():
            if k == "i_page":
                v = zoom_level

            kwargs_read_ome_tiff[k] = v

        da_image = read_file(file_name=file, is_tiff=True, **kwargs_data)
        if self.tag == DataType.GRAY:
            is_HE = self.proj_info.get_dtype(i_layer=i_layer) == "HE"
            da_image = rgb_he_to_gray(da_image, is_HE=is_HE, **self.config["datasets"].get("to_gray", {}))

        return da_image.astype(np.uint8)



    def load_3d_NCHW(self, aligned_tag: str="AFFINE", l_layers:list[int]=None) -> Tensor_image_NCHW| Dask_image_NCHW:
        """Load aligned 3D image data in NCHW format for specified alignment stage.

        This method provides access to the image data at different stages of the
        alignment pipeline. It supports both memory-efficient dask arrays for
        high-definition stages and regular PyTorch tensors for other stages.

        Args:
            aligned_tag (str, optional): Alignment stage to load data from.
                Valid options:
                - 'AFFINE_HD': Returns dask array for high-def affine aligned data
                - 'RAW': Raw unaligned data
                - 'PAD': Padded unaligned data
                - 'STACK': Initially aligned stack
                - 'AFFINE': Affine aligned data
                - 'NONRIGID': Non-rigidly aligned data
                Defaults to "AFFINE".
            l_layers (list[int], optional): List of layer indices to load.
                If None, loads all layers. Defaults to None.

        Returns:
            Union[torch.Tensor, da.Array]: Loaded image data in NCHW format:
                - For AFFINE_HD/NONRIGID_HD: dask array [N, C, H, W]
                - For other tags: PyTorch tensor [N, C, H, W]
                where:
                    N: number of layers
                    C: number of channels
                    H: height
                    W: width

        Note:
            - HD (high-definition) versions return dask arrays for memory efficiency
            - Regular versions return PyTorch tensors
            - Automatically handles color to grayscale conversion if needed
            - Caches results for efficient reuse
            - Returns None if requested stage data is not found
        """
        # Convert string to Dataset3DTag enum
        if l_layers is None:
            l_layers = list(range(len(self)))
        
        if aligned_tag == "AFFINE_HD":
            l_da_out = []
            for i_layer in tqdm(l_layers, desc="Loading AFFINE_HD"):
                dict_file_name = StageSampleTag.AFFINE_HD.get_file_name(i_layer=i_layer, projInfo=self.proj_info)
                if dict_file_name is None:
                    logger.error(f"No file found for {aligned_tag} for {self.proj_info.get_sample_name(i_layer=i_layer)} in {self.project}/{self.group}/{self.version}, please run `om.align.affine()` and `om.align.apply_affine_HD()` first.")
                    return None
                zarr_path = dict_file_name["zarr"]
                da_HWC = da.from_zarr(zarr_path)
                if self.tag == "gray":
                    is_HE = self.proj_info.get_dtype(i_layer=i_layer) == "HE"
                    da_HWC = rgb_he_to_gray(da_HWC, is_HE=is_HE, **self.config["datasets"].get("to_gray", {}))
                
                l_da_out.append(da.moveaxis(da_HWC, 2, 0))

            da_out = da.stack(l_da_out, axis=0)
            return da_out
        

        if aligned_tag == "NONRIGID_HD":
            l_da_out = []
            for i_layer in tqdm(l_layers, desc=f"Loading NONRIGID_HD"):
                dict_file_name = StageSampleTag.NONRIGID_HD.get_file_name(i_layer=i_layer, projInfo=self.proj_info)
                if dict_file_name is None:
                    logger.error(f"No file found for {aligned_tag} for {self.proj_info.get_sample_name(i_layer=i_layer)} in {self.project}/{self.group}/{self.version}, please run `om.align.affine()` and `om.align.apply_affine_HD()` first.")
                    return None
                zarr_path = dict_file_name["zarr"]
                da_HWC = da.from_zarr(zarr_path)
                if self.tag == "gray":
                    is_HE = self.proj_info.get_dtype(i_layer=i_layer) == "HE"
                    da_HWC = rgb_he_to_gray(da_HWC, is_HE=is_HE, **self.config["datasets"].get("to_gray", {}))
                
                l_da_out.append(da.moveaxis(da_HWC, 2, 0))

            da_out = da.stack(l_da_out, axis=0)
            return da_out
        
        try:
            FILE_NCHW = StageTag[aligned_tag]
        except KeyError:
            raise ValueError(f"Invalid aligned_tag: {aligned_tag}. Must be one of: {[tag.name for tag in StageTag]}")

        dict_file_name = FILE_NCHW.get_file_name(projInfo=self.proj_info)
        if dict_file_name is None:
            logger.warning(f"No file found for {aligned_tag} in {self.project}/{self.group}/{self.version}")
            return None
        
        logger.info(f"loading {dict_file_name['padded_tensor']}")
        self.padded_tensor = torch.load(dict_file_name['padded_tensor'])[l_layers]
        return self.padded_tensor

    def _load_dataset_params(self, **kwargs):
        set_kwargs_valid_keys = set(["project", "group", "version", "nonrigid", "zoom_level","tag", "method", "transform_engine", "min_disp", "max_size", "l_scales", "file_data", "raw_img_prefix", "raw_is_tiff", "read_ome_tiff", "plt_row_col", "plt_figsize"])

        self.project = None
        self.group = None
        self.version = None
        self.nonrigid = False
        self.zoom_level = 0
        self.root_dir = "../"
        self.tag = DataType.RAW
        
        self.method = "hipt"
        self.transform_engine = "torch"
        self.min_disp = 0.0
        self.max_size = 1600
        self.l_scales = [40, 10, 5, 1]
        self.file_data = ""
        self.zoom_key = "i_page"
        self.plt_row_col = [30, 30]
        self.plt_figsize = (30, 30)
        self.read_ome_tiff = None
        self.raw_img_prefix = ".tiff"
        self.raw_is_tiff = False
        dict_config = self.config["datasets"]
        for key, value in dict_config.items():
            logger.info(f"config.datasets: {key}: {value}")
            if key in set_kwargs_valid_keys:
                setattr(self, key, value)

        for key, value in kwargs.items():
            if key in set_kwargs_valid_keys and key in dict_config:
                logger.info(f"overwrite {key}: ({dict_config[key]} -> {value}) defined in kwargs")
                setattr(self, key, value)
        
        self.sizes = [s / self.l_scales[0] for s in self.l_scales]
        self.file_data = os.path.expanduser(self.file_data)
        self.root_dir = os.path.expanduser(self.root_dir)
        self.proj_info = ProjInfo(dict_config)    
        self.set_tag(self.proj_info.tag)
    
    def set_cache_tag(self, cache_tag:str):
        """Set the current cache stage tag for the dataset.

        Updates the processing stage indicator for the dataset, which determines
        what level of processed data is available and being used.

        Args:
            cache_tag (str): Cache stage identifier. Must be one of:
                - 'RAW': Raw unprocessed data
                - 'STACK': Stacked and initially aligned
                - 'AFFINE': Affine aligned
                - 'NONRIGID': Non-rigidly aligned
                - 'STALIGNER_MODEL': Model cache

        Raises:
            ValueError: If cache_tag is not one of the valid stage tags.

        Note:
            - Affects which cached data is accessed by other methods
            - Used to track processing progress
            - Validates tag against StageTag enum
        """
        try:
            _ = StageTag[cache_tag]
        except KeyError:
            raise ValueError(f"Invalid cache_tag: {cache_tag}. Must be one of: {[tag.name for tag in StageTag]}")
        self.CACHE_TAG = cache_tag


    def _load_TFRS_params(self, i_layer: int) -> Tensor_tfrs:
        dict_file_name_stack = StageTag.STACK.get_file_name(projInfo=self.proj_info)
        if dict_file_name_stack is None:
            logger.error("Stacked tensor not found. Please run `om.stack()` first.")
            return None, None
        
        dir_stack = os.path.dirname(dict_file_name_stack["padded_tensor"])
        if os.path.isfile(f"{dir_stack}/omni_stack/tfrs_to_stacked.pt"):
            l_tfrs = torch.load(f"{dir_stack}/omni_stack/tfrs_to_stacked.pt")
            tensor_tfrs = l_tfrs[i_layer]
            return tensor_tfrs

        angle, flip, _ = torch.load(StageTag.STACK.get_file_name(self.proj_info)["flip_angle"])[i_layer]
        fx = 1 if flip[0] == 0 else -1
        fy = 1 if flip[1] == 0 else -1
        sx, sy = 0, 0
        
        tensor_tfrs = torch.FloatTensor([np.deg2rad(angle), 0, 0, sx, sy, fx, fy])
        return tensor_tfrs

    def _get_adata(self, i_layer, tag="sub", method="hipt"):
        """Load annotated data for a specific layer.

        Internal method to load processed and annotated data from h5ad files,
        typically containing feature embeddings or analysis results.

        Args:
            i_layer (int): Layer index to load data for.
            tag (str, optional): Data subset tag. Defaults to "sub".
            method (str, optional): Processing method identifier. Defaults to "hipt".

        Returns:
            anndata.AnnData: Annotated data object containing processed results.

        Note:
            - Loads data from standardized project directory structure
            - File path format: {root_dir}/analysis/{project}/09.omics_alignment/{group}/{tag}/{sample}_{method}_{tag}.h5ad
            - Used primarily for accessing processed embeddings and analysis results
        """
        sample = self.proj_info.get_sample_name(i_layer=i_layer)
        adata = sc.read_h5ad(f"{self.root_dir}/analysis/{self.project}/09.omics_alignment/{self.group}/{tag}/{sample}_{method}_{tag}.h5ad")
        return adata

    def set_zoom_level(self, zoom_level):
        """Set the current zoom level for image loading.

        Args:
            zoom_level (int): New zoom level value. Higher values mean more
                zoomed out view of the images.

        Note:
            - Affects subsequent image loading operations
            - Zoom levels typically correspond to pyramid levels in OME-TIFF
            - Higher zoom levels result in smaller images
        """
        self.zoom_level = zoom_level

    def set_tag(self, tag: str| DataType):
        """Set the data type tag for the dataset.

        Updates the data type identifier which determines how images are processed
        and handled throughout the pipeline.

        Args:
            tag (Union[str, DataType]): New data type tag, either as string or
                DataType enum. Valid string values must match DataType enum names.

        Raises:
            ValueError: If tag string doesn't match any DataType enum value.

        Note:
            - Affects image processing behavior (e.g., color vs grayscale)
            - Updates both instance and project info tag values
            - Case-insensitive when provided as string
        """
        if isinstance(tag, str):
            tag = tag.upper()
            if tag in DataType.__members__:
                self.tag = DataType[tag]
                self.proj_info.tag = self.tag
                return
            else:
                raise ValueError(f"Invalid tag: {tag}. Must be one of: {[tag.name for tag in DataType]}")
        
        self.tag = tag
        self.proj_info.tag = self.tag

    def set_method(self, method):
        """Set the processing method identifier.

        Args:
            method (str): New method identifier.

        Note:
            - Updates both instance and project info method values
            - Affects how subsequent processing steps are performed
        """
        self.method = method
        self.proj_info.method = method

    def set_transform_engine(self, transform_engine):
        """Set the transformation engine type.

        Args:
            transform_engine (str): New transform engine identifier.

        Note:
            - Determines which backend is used for image transformations
            - Common values include 'torch' and 'cv2'
        """
        self.transform_engine = transform_engine

    def _resize_padding(self, i_layer, da_image, zoom_level=0, max_size=1600):
        """Calculate padding parameters for image resizing.

        Internal method to compute padding values needed to properly resize
        and align images while maintaining consistent dimensions.

        Args:
            i_layer (int): Layer index for padding calculation.
            da_image (da.Array): Input dask array image.
            zoom_level (int, optional): Current zoom level. Defaults to 0.
            max_size (int, optional): Maximum size constraint. Defaults to 1600.

        Returns:
            tuple: Padding parameters ((pad_h_before, pad_h_after),
                                    (pad_w_before, pad_w_after),
                                    (0, 0))

        Note:
            - Handles special cases for different image types (e.g., IHC)
            - Accounts for zoom level in padding calculations
            - Ensures output size doesn't exceed max_size
            - Returns padding for height, width, and channels
        """
        dtype = self.proj_info.get_dtype(i_layer=i_layer)

        scale_input = self.l_scales[zoom_level]
        if self.project == "CRC" and dtype == "IHC":
            scale_input = scale_input // 2
        
        vec_pad = self.padded_sizes[i_layer]
        beg_h = int((vec_pad[2]) * scale_input)
        end_h = int((max_size - vec_pad[3]) * scale_input)
        beg_w = int((vec_pad[0]) * scale_input)
        end_w = int((max_size - vec_pad[1]) * scale_input)
        
        img_h, img_w = da_image.shape[0], da_image.shape[1]
        end_h = min(end_h, beg_h + img_h)
        end_w = min(end_w, beg_w + img_w)

        pad_h_before = beg_h
        pad_h_after = max(0, (max_size * scale_input) - (beg_h + img_h))
        pad_w_before = beg_w
        pad_w_after = max(0, (max_size * scale_input) - (beg_w + img_w))
        return ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after), (0, 0))

    def _load_transform(self, tensor_tfrs=None, init_disp=None, zoom_level=0, max_size=1600):
        """Initialize transformation sampling grid.

        Internal method to create a sampling grid for image transformation
        based on provided parameters.

        Args:
            tensor_tfrs (Tensor_tfrs, optional): Transformation parameters.
                Defaults to identity transform if None.
            init_disp (torch.Tensor, optional): Initial displacement field.
                Defaults to zero displacement if None.
            zoom_level (int, optional): Current zoom level. Defaults to 0.
            max_size (int, optional): Maximum size constraint. Defaults to 1600.

        Returns:
            torch.Tensor: Sampling grid for image transformation.

        Note:
            - Creates grid on CPU by default
            - Handles scaling based on zoom level
            - Supports both affine and displacement-based transformations
            - Grid size is adjusted based on max_size and scale
        """
        dev = torch.device("cpu")
        scale = self.l_scales[zoom_level]
        if tensor_tfrs is None:
            tensor_tfrs = torch.FloatTensor([0,0,0,0,0, 1,1]).to(dev)
        if init_disp is None:
            init_disp = torch.zeros([1, int(max_size*scale), int(max_size*scale), 2]).to(dev)
        
        tensor_size = [int(max_size*scale), int(max_size*scale)]
        sampling_grid = get_sampling_grid(tensor_tfrs, tensor_size, init_disp, dev)
        return sampling_grid
    
    def _load_pad(self):
        """Load padding information from cache.

        Internal method to retrieve cached padding parameters required for
        maintaining consistent image dimensions.

        Note:
            - Loads from PAD stage cache
            - Updates instance padded_sizes attribute
            - Required for proper image alignment
        """
        FILE_cache_padding = StageTag.PAD
        dict_file_name = FILE_cache_padding.get_file_name(projInfo=self.proj_info)
        self.padded_sizes = torch.load(dict_file_name["merged_padded_sizes"])
    
    def _load_stack(self):
        """Load stacking parameters from cache.

        Internal method to retrieve cached parameters related to image
        stacking and initial alignment.

        Note:
            - Loads flip and angle parameters
            - Adds default parameters for first layer
            - Required for stack reconstruction
        """
        FILE_cache_stack = StageTag.STACK
        dict_file_name = FILE_cache_stack.get_file_name(projInfo=self.proj_info)
        l_flip_angle = torch.load(dict_file_name["flip_angle"])
        self.flip_angle =  [[0., [0, 0], 0.]] + l_flip_angle
        
    def _load_align(self):
        """Load alignment model parameters from cache.

        Internal method to retrieve cached alignment model parameters for
        both affine and optional nonrigid transformations.

        Note:
            - Loads affine model parameters by default
            - Optionally loads nonrigid parameters if enabled
            - Parameters are loaded to CPU
            - Updates instance model and parameter attributes
        """
        FILE_cache_affine = StageTag.AFFINE
        dict_file_name = FILE_cache_affine.get_file_name(projInfo=self.proj_info)
        self.model_affine = dict_file_name["affine_model"]
        self.affine_params = torch.load(self.model_affine, map_location="cpu")
        self.nonrigid_params = None
        if self.nonrigid:
            FILE_cache_nonrigid = StageTag.NONRIGID
            dict_file_name = FILE_cache_nonrigid.get_file_name(projInfo=self.proj_info)
            self.model_nonrigid = dict_file_name["nonrigid_model"]
            self.nonrigid_params = torch.load(self.model_nonrigid, map_location="cpu")

    def _load_embed(self, project, group, sample, tag="sub", method="hipt", hipt_size=None):
        if method != "hipt" and hipt_size is None:
            hipt_size = self.init_hipt_size(project, group, sample)

        pt_file = f"{self.root_dir}/analysis/{project}/02.1.dino_feats/{group}/{sample}_{method}.pt"
        
        dict_ref = torch.load(pt_file)
        tensor = dict_ref[tag].unsqueeze(0)
        if tag == "cls":
            tensor = tensor.permute(0, 3, 1, 2)

        if method == "hipt":
            return tensor

        h, w = hipt_size[sample][tag][0], hipt_size[sample][tag][1]
        return F.interpolate(tensor, size=[h, w], mode="bilinear", align_corners=True)

    def init_hipt_size(self, project, group, sample, align_to="hipt"):
        """Initialize HiPT size parameters for samples.

        Computes and returns size information for HiPT (Hierarchical Pre-trained
        Transformer) processing of samples.

        Args:
            project (str): Project identifier.
            group (str): Group name.
            sample (str): Sample identifier.
            align_to (str, optional): Alignment method. Defaults to "hipt".

        Returns:
            dict: Nested dictionary containing size information:
                {sample: {'cls': [height, width], 'sub': [height, width]}}

        Note:
            - Computes sizes for both 'cls' and 'sub' embeddings
            - Uses cached embeddings if available
            - Required for proper feature extraction and alignment
        """
        l_samples = [sample]
        hipt_size = {}
        for sample in l_samples:
            hipt_size[sample] = {}
            for tag in ["cls", "sub"]:
                tensor = self._load_embed(project, group, sample, tag=tag, method=align_to) 
                hipt_size[sample][tag] = [ tensor.shape[2], tensor.shape[3] ]

        return hipt_size

    def __len__(self):
        """Get the number of layers in the dataset.

        Returns:
            int: Number of layers in the project.

        Note:
            - Delegates to proj_info length
            - Useful for iteration over layers
        """
        return len(self.proj_info)



def tag_decorator(func):
    @functools.wraps(func)
    def wrapper(obj, **kwargs):
        original_tag = obj.tag
        
        if 'tag' in kwargs:
            obj.set_tag(kwargs['tag'])
            logger.debug(f"Changed tag to: {kwargs['tag']} in object {obj}, function {func.__name__}")
        
        result = func(obj, **kwargs)
        obj.set_tag(original_tag)
        logger.debug(f"Restored tag to: {original_tag} in object {obj}, function {func.__name__}")
        
        return result
    return wrapper

def zoom_level_decorator(func):
    @functools.wraps(func)
    def wrapper(obj, **kwargs):
        original_zoom_level = obj.zoom_level
        
        if 'zoom_level' in kwargs:
            obj.zoom_level = kwargs['zoom_level']
            logger.debug(f"Changed zoom_level to: {kwargs['zoom_level']} in object {obj}, function {func.__name__}")

        result = func(obj, **kwargs)
        
        obj.zoom_level = original_zoom_level
        logger.debug(f"Restored zoom_level to: {original_zoom_level} in object {obj}, function {func.__name__}")
        
        return result
    return wrapper