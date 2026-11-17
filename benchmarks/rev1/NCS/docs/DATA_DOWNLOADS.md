# Original H5AD downloads

No H5AD files are stored in Git. Direct file URLs and section mapping notes are in [H5AD_downloads.csv](H5AD_downloads.csv). Exact evaluation filenames and SHA-256 checksums are in [evaluation_input_files.csv](evaluation_input_files.csv).

| Dataset | Original data download | Evaluation sections |
|---|---|---:|
| DLPFC | [SABench example archive](https://zenodo.org/api/records/18605715/files/SABench.zip/content) | 4 |
| MHPR | [MERFISH H5AD example](http://sdmbench.drai.cn/tcm/download/?file_path=/mnt/JINGD/data/file/sdmbench/db/MERFISH_0.04.h5ad); all 5 links in CSV | 5 |
| MMPC | [STARmap H5AD example](http://sdmbench.drai.cn/tcm/download/?file_path=/mnt/JINGD/data/file/sdmbench/db/20180417_BZ5_control.h5ad); all 3 links in CSV | 3 |
| MPC | [BaristaSeq H5AD example](http://sdmbench.drai.cn/tcm/download/?file_path=/mnt/JINGD/data/file/sdmbench/db/Slice_1.h5ad); all 3 links in CSV | 3 |
| E14–16 | [Stereo-seq H5AD](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000060/stomics/E14-16h_a_count_normal_stereoseq.h5ad) | 16 |
| MLN | [Open-ST H5AD.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE251nnn/GSE251926/suppl/GSE251926_metastatic_lymph_node_3d.h5ad.gz) | 19 |

The notebooks use the public `rev1-benchmark-data.tar.gz` release archive, which supplies the exact expression panel, labels, cell identities and raw coordinates needed by the frozen evaluation. Its SHA-256 is verified before extraction, and every section NPZ is checked against its manifest. Full H5AD downloads are optional and are not read by these notebooks.

The upstream downloads are not byte-identical to the historical per-section scoring containers. Their hashes and filenames in `evaluation_input_files.csv` document provenance; a fresh public-H5AD-to-container preprocessing replay is not claimed. The release NPZs avoid any dependence on those private containers while preserving their exact scoring inputs.

The Zenodo record was checked through its API; E14–16, MLN and one H5AD link for each of MHPR/MMPC/MPC returned HTTP 200 on 17 November 2026. The other links retain the recorded upstream download pattern and are separately listed for checking.

Sources: [SABench repository](https://github.com/Yunzhi-Yan/SABench), [benchmark article](https://doi.org/10.1038/s43588-026-00977-z), [SDMBench](http://sdmbench.drai.cn/), [Open-ST GEO record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE251926).
