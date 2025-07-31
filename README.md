# OmniAligner

OmniAligner is a tool designed for advanced image alignment and processing, particularly in the context of high-resolution microscopy and spatial data analysis.

## Features

- Keypoint detection and matching
- Image registration and transformation
- Support for multi-resolution image formats
- Integration with machine learning models for enhanced alignment

## Requirements

- Python 3.10 or higher
- Dependencies:
  - `torch`
  - `numpy`
  - `dask`
  - `opencv-python`
  - `Pillow`
  - `scanpy`

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/huboqiang/omnialigner.git
cd omnialigner
conda env create -f environment.yaml
conda activate omni
pip install -r requirements.txt

### for istar checkpoints
cd src/omnialigner/vendor/istar
bash download_checkpoints.sh
```

## Example for pdac 2d

### Change Config file
1. Download example data [10.5281/zenodo.16222047](10.5281/zenodo.16222047) and unzip
2. Modify "YOUR_DATA_DIR" into dirname of downloaded data in `config/panlab2d/config_pdac.yaml`
3. Modify "YOUR_PROJECT_DIR" into dirname of current dir name.

### run OmniAligner

```python
om.pp.pad(om_data)
om.align.stack(om_data)
om.align.affine(om_data)
## nonrigid need a server
om.align.nonrigid(om_data)
```

### OmniAligner MCP agent

```bash
## open MCP server
python servers/aligner_agent/omni_MCP/omnialigner_mcp.py

## run agent in another bash
streamlit run servers/agent.py
```

## OmniTME agent

First make sure the browser worked with the data folder.

```bash
python ./servers/omics_browser/examples/deepzoom/deepzoom_multiserver.py  -p 5020 -l 0.0.0.0 YOUR_DATA_DIR/analysis/panlab/v1/fig/
```

Then start the omniTME agent:

```bash
python ./servers/navigation_agent/voyager.py
```