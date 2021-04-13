import os
import hydra
import torch
import logging
import torch.nn as nn
from torch import optim
from hydra import utils
from torch.utils.data import DataLoader
# self
from preprocess import preprocess
from dataset import CustomDataset, collate_fn
from trainer import TRAIN, TEST
from utils import load_pkl

logger = logging.getLogger(__name__)

@hydra.main(config_path='config/config.yaml')
def main(cfg):
    cwd = utils.get_original_cwd()
    cfg.cwd = cwd

    # device
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda', cfg.gpu_id)
    else:
        device = torch.device('cpu')
    logger.info(f'device: {device}')

    # once enough
    if cfg.preprocess:
        preprocess(cfg)

    train_data_path = os.path.join(cfg.cwd, cfg.out_path, 'train.pkl')
    valid_data_path = os.path.join(cfg.cwd, cfg.out_path, 'valid.pkl')
    test_data_path = os.path.join(cfg.cwd, cfg.out_path, 'test.pkl')
    vocab_path = os.path.join(cfg.cwd, cfg.out_path, 'vocab.pkl')

    train_dataset = CustomDataset(train_data_path)
    valid_dataset = CustomDataset(valid_data_path)
    test_dataset = CustomDataset(test_data_path)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn(cfg))
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn(cfg))
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn(cfg))

    # TRAIN("tail", train_dataloader, valid_dataloader, test_dataloader, device, cfg)
    # TRAIN("rel", train_dataloader, valid_dataloader, test_dataloader, device, cfg)

    TEST(test_dataloader, device, cfg)

    


if __name__ == '__main__':
    main()
    # python predict.py --help  # 查看参数帮助
    # python predict.py -c
    # python predict.py chinese_split=0,1 replace_entity_with_type=0,1 -m
