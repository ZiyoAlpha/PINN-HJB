from utils import load_yaml
from model.train import train
import torch
from utils import get_logger
import logging
import time
import os
from data.data_loader import calculate_parements_stock

if __name__ == "__main__":
    cfg = load_yaml('config.yaml')

    # 设置日志
    os.makedirs(cfg.log_dir, exist_ok=True)
    get_logger(cfg.log_name, log_file=f"{cfg.log_dir}/loss_{time.strftime('%d')}.log")
    logger = logging.getLogger(cfg.log_name)
    
    # 计算或加载参数
    paraments = calculate_parements_stock(cfg.csv_path, cfg.result_path, save_csv=cfg.needing_calculate)

    # 训练或加载模型
    if cfg.train:
        train(cfg, paraments)
    else:
        model_path = os.path.join(cfg.model_dir, "pinn_model.pth")
        if os.path.exists(model_path):
            logger.info(f"加载模型: {model_path}")
            # 这里需要先创建模型实例，然后加载权重
            # model = PINN(...)
            # model.load_state_dict(torch.load(model_path))
        else:
            logger.warning(f"模型文件不存在: {model_path}")

    

    