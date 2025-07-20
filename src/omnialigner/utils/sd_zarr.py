import os
import copy
import logging as logging_

import h5py
import pandas as pd
import numpy as np
import dask.array as da
import spatialdata as sd
import geopandas as gpd
from spatialdata.models import TableModel, PointsModel, Image2DModel
from spatialdata.transformations import set_transformation, Scale
from xarray import DataArray
import anndata as ad
from shapely.affinity import scale as _scale_geom, translate as _translate_geom

import omnialigner as om
from omnialigner.dtypes import Dask_image_HWC

logging_.getLogger('ome_zarr').propagate = False

def load_spatial_adata_from_h5(
    h5_path: str,
    features_key: str = "features",
    coords_key: str = "coords"
) -> ad.AnnData:
    """
    Load spatial transcriptomics data from an HDF5 file and return a Scanpy AnnData object.
    The AnnData object contains:
        - .X: Feature matrix.
        - .obs: Pixel coordinates.
        - .obsm['spatial']: 2D spatial coordinates.

    Args:
        h5_path (str): 
            Path to the input HDF5 file, which should contain the datasets `features_key` and `coords_key`.
        features_key (str, optional): 
            Name of the flattened feature dataset in the HDF5 file. Defaults to "features".
        coords_key (str, optional): 
            Name of the expanded coordinate dataset in the HDF5 file. Defaults to "coords".

    Returns:
        ad.AnnData: 
            An AnnData object containing:
                - .X (float32): Feature matrix.
                - .obs['x'], .obs['y'] (int): Pixel coordinates.
                - .obsm['spatial'] (float): 2D spatial coordinates.
    """
    with h5py.File(h5_path, "r") as f:
        feats = f[features_key][:]
        coords = f[coords_key][:]

    obs = pd.DataFrame({
        "x": coords[:, 1],
        "y": coords[:, 0]
    }, index=[f"cell_{i}" for i in range(coords.shape[0])])

    adata = ad.AnnData(X=feats, obs=obs)
    adata.obsm["spatial"] = coords
    return adata


def move_gdf(gdf: gpd.GeoDataFrame, scale_factor:float=0.25, xoff: float=0., yoff:float=0.) -> gpd.GeoDataFrame:
    scaled_gdf = gdf.copy()
    scaled_gdf['geometry'] = scaled_gdf.geometry.map(
        lambda geom: _translate_geom(
            _scale_geom(geom, xfact=scale_factor, yfact=scale_factor, origin=(0, 0)),
            xoff=xoff,
            yoff=yoff
        )
    )
    return scaled_gdf


def merge_spatialdata_with_scaling(sdata, sdata1, scale_factor=0.25, xoff=0, yoff=0):
    """
    Scale all global coordinates of `sdata1` by `scale_factor`, translate them by `(xoff, yoff)`, 
    and merge the result into `sdata`.

    Args:
        sdata (SpatialData): 
            The target SpatialData object from `om.tl.write_omnialigner_zarr`
        sdata1 (SpatialData): 
            The source SpatialData object from `om.external.lazy_slide.run_zs`
        scale_factor (float, optional): 
            Scaling factor. Omnialigner using zoom_level=1 for `l_scales: [40, 10, 5, 1]`, default is 10/40 = 0.25.
        xoff (float, optional): 
            Offset for translation in the X direction. (Pad - Crop). Defaults to 0.
        yoff (float, optional): 
            Offset for translation in the Y direction. (Pad - Crop). Defaults to 0.

    Returns:
        SpatialData: 
            The modified `sdata` object with the merged content.
    """
    # 1) 合并 Images
    for img_name, img_da in sdata1.images.items():
        scaled_img = img_da.copy() if hasattr(img_da, 'copy') else copy.deepcopy(img_da)
        sdata.images[img_name] = scaled_img

    # 2) 合并 Shapes
    for layer_name, gdf in sdata1.shapes.items():
        sdata.shapes[layer_name] = move_gdf(gdf, scale_factor=scale_factor, xoff=xoff, yoff=yoff)

    # 3) 合并 Tables（AnnData）
    for tbl_name, ad in sdata1.tables.items():
        scaled_ad = ad.copy()
        for coord in ('x', 'y'):
            if coord in scaled_ad.obs.columns:
                offset = xoff if coord == 'x' else yoff
                scaled_ad.obs[coord] = scaled_ad.obs[coord] * scale_factor + offset
        sdata.tables[tbl_name] = scaled_ad

    return sdata


def dask_to_multiscale_image(data: Dask_image_HWC, scale_factors=None):
    if scale_factors is None:
        scale_factors = [{'y': 2, 'x': 2}, {'y': 2, 'x': 2}, {'y': 2, 'x': 2}]
    
    image = DataArray( da.moveaxis(data, 2, 0), dims=("c", "y", "x"))
    parsed = Image2DModel.parse(image, scale_factors=scale_factors, rgb=None)
    return parsed


def create_image2d(da_raw: Dask_image_HWC):
    parsed_image = dask_to_multiscale_image(da_raw)
    Image2DModel().validate(parsed_image)
    return parsed_image

def create_table(adata: ad.AnnData):
    adata = adata.copy()
    adata.X = None
    parsed_table = TableModel.parse(adata)
    TableModel().validate(parsed_table)
    return parsed_table


def write_omnialigner_zarr(om_data: om.Omni3D, zoom_level: int=1, adata_merged: ad.AnnData=None, sample_tag: str="sample", overwrite_cache=False) -> sd.SpatialData:
    """
    Write omni-aligner results and metadata to a Zarr file.

    Args:
        om_data (Omni3D): The Omni3D object containing project configuration and data.
        zoom_level (int, optional): The zoom level to use for image alignment and output. Defaults to 1.
        adata_merged (AnnData, optional): The merged AnnData object containing omics data after MISO. 
            If None, a default file will be loaded.

    Returns:
        SpatialData: zarr already written.

    """
    if adata_merged is None:
        adata_merged = ad.read_h5ad("/cluster/home/bqhu_jh/projects/panlab/code/wujiangchao/ALLTLS/merged_adata_anno.h5ad")
    
    root_dir = os.path.expanduser(om_data.config["datasets"]["root_dir"])
    project = om_data.config["datasets"]["project"]
    group_id = om_data.config["datasets"]["group"]
    version = om_data.config["datasets"]["version"]
    out_dir = f"{root_dir}/analysis/{project}/{version}/10.merge_spatialdata/{group_id}/"
    zarr_dir = f"{out_dir}/sub_{group_id}.zarr"
    if not overwrite_cache and os.path.isdir(zarr_dir):
        sd_sub = sd.read_zarr(zarr_dir)
        return sd_sub

    dict_IHC_layer = om_data.proj_info.load_IHC_info()
    dict_results = {}
    adata_omics = adata_merged[(adata_merged.obs[sample_tag] == group_id)]
    for i_layer in range(len(om_data)):
        dtype = om_data.proj_info.get_dtype(i_layer)
        tag = "raw" if dtype == "HE" else "rmaf"
        om_data.set_zoom_level(zoom_level)
        da_img = om.align.apply_nonrigid_HD(om_data, i_layer=i_layer, overwrite_cache=False, tag=tag)
        if dtype == "HE":
            dict_results["raw_HE"] = create_image2d(da_img)
        else:
            for idx, i_channel in enumerate(dict_IHC_layer["order_label"]):
                gene_name = dict_IHC_layer["genes"][dtype][idx]
                dict_results[f"{dtype}_{gene_name}"] = create_image2d(da_img[:, :, i_channel:i_channel+1])
    
    adata_used = adata_omics
    if "pca_harmony_rgb" in adata_used.obsm:
        adata_used.obs["pca_0"] = adata_used.obsm["pca_harmony_rgb"][:, 0]
        adata_used.obs["pca_1"] = adata_used.obsm["pca_harmony_rgb"][:, 1]
        adata_used.obs["pca_2"] = adata_used.obsm["pca_harmony_rgb"][:, 2]

    adata_used.uns["spatialdata_attrs"] = {"region": "cls_labels", "region_key": "region", "instance_key" : "point_id"}
    adata_used.obs["region"] = "cls_labels"
    adata_used.obs["region"] = adata_used.obs["region"].astype('category')

    adata_used.obs["point_id"] = adata_used.obs.index
    dict_results["obs_sub"] =  TableModel.parse(adata_used)
    points = adata_used.obsm["spatial"]
    gdf = sd.to_circles(PointsModel.parse(np.array(points)), radius=4)  # gdf #shapes_model
    gdf.index = adata_used.obs.index
    # gdf.index.name = "point_id_"
    dict_results["cls_labels"] = gdf
    
    
    sd_sub = sd.SpatialData.from_elements_dict(dict_results)
    scaled = om_data.l_scales[zoom_level]
    scale = Scale([scaled, scaled], axes=("x", "y"))
    set_transformation(sd_sub.shapes["cls_labels"], scale, to_coordinate_system="global")

    os.makedirs(out_dir, exist_ok=True)
    sd_sub.write(zarr_dir, overwrite=True)
    return sd_sub
    

