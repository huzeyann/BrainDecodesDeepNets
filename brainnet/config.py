from pathlib import Path
from yacs.config import CfgNode
from yacs.config import CfgNode as CN

# YACS overwrite these settings using YAML, all YAML variables MUST BE defined here first
# as this is the master list of ALL attributes.
_C = CN()

# importing default as a global singleton
_C.DESCRIPTION = "Default config"

_C.DATASET = CN()
_C.DATASET.DATA_DIR = "/data/download/alg23/subj01"
_C.DATASET.RESOLUTION = (224, 224)
_C.DATASET.BATCH_SIZE = 8

_C.MODEL = CN()
_C.MODEL.LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
_C.MODEL.LAYER_WIDTHS = [768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768]
_C.MODEL.BOTTLENECK_DIM = 128

_C.OPTIMIZER = CN()
_C.OPTIMIZER.LR = 1e-3
_C.OPTIMIZER.WEIGHT_DECAY = 3e-4

_C.REGULARIZATION = CN()
_C.REGULARIZATION.LAMBDA = 0.1
_C.REGULARIZATION.DECAY_TOTAL_STEPS = 2000

_C.TRAINER = CN()



def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    return _C.clone()

def save_to_yaml(cfg, path_output):
    """
    Save the current config to a YAML file.
    :param cfg: CfgNode object to be saved
    :param path_output: path to output files
    """
    path_output = Path(path_output)
    path_output.parent.mkdir(parents=True, exist_ok=True)
    with open(path_output, "w") as f:
        f.write(cfg.dump())
        
def load_from_yaml(path_cfg_data, path_cfg_override=None, list_cfg_override=None):
    """
    Load a config from a YAML file.
    :param path_cfg_data: path to path_cfg_data files
    :param path_cfg_override: path to path_cfg_override actual
    :param list_cfg_override: [key1, value1, key2, value2, ...]
    :return: cfg_base incorporating the overwrite.
    """
    cfg_base = get_cfg_defaults()
    cfg_base.merge_from_file(path_cfg_data)
    return cfg_base