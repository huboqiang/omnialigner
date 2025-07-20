from omnialigner.omni_3D import Omni3D
from omnialigner.cache_files import StageTag

omni_dataset = Omni3D(config_file="./configs/pdac/config.yaml")
omni_dataset.set_tag("gray")
omni_dataset.set_zoom_level(3)


tag_enum = StageTag["RAW"]
dict_file_name = tag_enum.get_file_name(projInfo=omni_dataset.proj_info)
dict_file_name


from src.omnialigner import StageSampleTag
kpt_tag = "KEYPOINTS"
FILE_per_sample = StageSampleTag[kpt_tag]
file_kpts = FILE_per_sample.get_file_name(projInfo=omni_dataset.proj_info, i_layer=0, check_exist=False)
file_kpts