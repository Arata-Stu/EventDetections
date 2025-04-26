import sys
sys.path.append('../..')

import hydra
from omegaconf import DictConfig, OmegaConf

from config.modifier import dynamically_modify_train_config
from modules.utils.fetch import fetch_data_module, fetch_model_module
from utils.vis_utils import create_video

@hydra.main(config_path="../../config", config_name="visualize", version_base="1.2")
def main(cfg: DictConfig):
    dynamically_modify_train_config(cfg)
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    print('------ Configuration ------')
    print(OmegaConf.to_yaml(cfg))
    print('---------------------------')

    output_path = cfg.output_path
    show_gt = cfg.gt
    show_pred = cfg.pred
    fps = cfg.fps
    num_sequence = cfg.num_sequence
    dataset_mode = cfg.dataset_mode
    ckpt_path = cfg.ckpt_path

    ## データセットの読み込み
    data = fetch_data_module(config=cfg)
    ## モデルの読み込み
    module = fetch_model_module(config=cfg)
    module_class = type(module)
    if ckpt_path is not None:
        model = module_class.load_from_checkpoint(module, ckpt_path=ckpt_path)
    ##ビデオの作成
    create_video(data, model, show_gt, show_pred, output_path, fps, num_sequence, dataset_mode)

if __name__ == '__main__':
    main()
