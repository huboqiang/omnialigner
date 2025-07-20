import unittest
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import numpy as np

from omnialigner.preprocessing.cluster_anno_cutter import train_cut_model    
from omnialigner.configs.pdac_tree import nodes, children_left, children_right, feature_names

@dataclass
class SampleMarkerConfig:
    markers: List[str]
    low: float = 0.05
    high: float = 0.9


def example_marker1() -> Dict[str, SampleMarkerConfig]:
    marker_dict = {
        "Tumor1": SampleMarkerConfig(["MKI67", "CD274"]),
        "Tumor2": SampleMarkerConfig(["MKI67", "CD274"]),
        "Epi1": SampleMarkerConfig(["PANK1", "CD274"]),
        "CD163+Mac1": SampleMarkerConfig(["CD163", "CD68"]),
        "CD163-Mac1": SampleMarkerConfig(["CD68"]),
        "Tc1": SampleMarkerConfig(["CD8A", "CD3E"]),
        "Treg1": SampleMarkerConfig(["FOXP3", "CD4", "CD3E"]),
        "Th1": SampleMarkerConfig(["CD4", "CD3E"]),
        "Th2": SampleMarkerConfig(["CD4", "CD3E"]),
        "DNT1": SampleMarkerConfig(["CD3E"]),
        "BCell1": SampleMarkerConfig(["MS4A1", "FCER2", "CR2"]),
        "BCell2": SampleMarkerConfig(["MS4A1", "FCER2", "CR2"]),
        "BCell3": SampleMarkerConfig(["MS4A1", "FCER2", "CR2"]),
        "Monocyte1": SampleMarkerConfig(["IL10", "MPO"]),
        "Monocyte2": SampleMarkerConfig(["IL10", "MPO"]),
        "Dendritic1": SampleMarkerConfig(["ITGAX", "HLA-DRA", "LAMP3"]),
        "Vascular1": SampleMarkerConfig(["PECAM1"]),
        "Fibroblast1": SampleMarkerConfig(["ACTA2"]),
        "Fibroblast2": SampleMarkerConfig(["ACTA2"]),
        "Unclass1": SampleMarkerConfig([])  # 留空但保留默认 low/high
    }
    return marker_dict

def generate_test_matrix(
    marker_dict: Dict[str, SampleMarkerConfig],
    gene_names: list[str] = None
) -> pd.DataFrame:
    """
    Generates a test gene expression matrix with customizable per-sample low/high marker definitions.

    Parameters:
        marker_dict: dict mapping sample name to dict:
                     {
                         'markers': list of marker genes,
                         'low': float (default 0.05),
                         'high': float (default 0.9)
                     }
        gene_names: list of gene names (used as rows)

    Returns:
        pd.DataFrame: gene × sample matrix
    """
    if gene_names is None:
        gene_names = feature_names

    samples = list(marker_dict.keys())
    df = pd.DataFrame(index=feature_names, columns=samples)

    for sample, config in marker_dict.items():
        df[sample] = config.low
        valid = [m for m in config.markers if m in df.index]
        df.loc[valid, sample] = config.high
    return df.astype(np.float32).T


class TestClusterAnnoCutter(unittest.TestCase):
    def test_cut_matrix(self):
        df_avg = generate_test_matrix(example_marker1())
        df_pred = train_cut_model(df_avg, n_trials=300, children_left=children_left, children_right=children_right, nodes=nodes, feature_names=feature_names)
        self.assertIsInstance(df_pred, pd.DataFrame)
        

if __name__ == "__main__":
    unittest.main()