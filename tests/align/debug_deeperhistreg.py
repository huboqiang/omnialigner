import yaml
import omnialigner as om

if __name__ == "__main__":
    # om_data = om.omni_3D.Omni3D(config_info="/cluster/home/bqhu_jh/projects/omni/config/CRC/config_CRC1.yaml")
    group_id = "COAD_01_4"
    with open("/cluster/home/bqhu_jh/projects/omni/config/ANHIR/config_ANHIR.yaml", 'r') as f:
        template_string = f.read()
        config_info = yaml.load(template_string, Loader=yaml.FullLoader)

    config_info["datasets"]["group"] = f"{group_id}"
    om_data = om.Omni3D(config_info=config_info)
    padded_tensor = om.align.nonrigid(om_data, overwrite_cache=True)
    om.pl.plot_nchw_2d(om_data, aligned_tag="NONRIGID")
