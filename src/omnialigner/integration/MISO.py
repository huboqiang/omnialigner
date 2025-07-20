import os
import sys

import numpy as np
import torch
import scanpy as sc

root_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '../'))
vendor_path = os.path.join(root_path, 'vendor/miso/miso')
sys.path.append(vendor_path)
print(vendor_path)
from miso.hist_features import get_features
from miso.utils import *
from miso import Miso


seed=100
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("CUDA is available. GPU:", torch.cuda.get_device_name(0))
else:
    device = 'cpu'
    print("CUDA is not available. Using CPU.")


def integrate_MISO(np_exp: np.ndarray, np_HE: np.ndarray, n_clusters=15, outdir=None):
    model = Miso([np_exp, np_HE], ind_views='all',combs='all',sparse=True, device=device)
    model.train()
    clusters_ = model.cluster(n_clusters=n_clusters)
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        torch.save([ p.state_dict() for p in model.mlps ], f"{outdir}/model_fc.pth")
        np.save(f"{outdir}/emb.npy", model.emb)
    
    return clusters_, model.emb
