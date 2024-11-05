import logging
import os
import random
import numpy as np
import torch


##配置单线程日志，输出至控制台和文件
def setup_logging(log_file, level):
    # 创建文件处理器和流处理器
    file_handler = logging.FileHandler(filename=log_file)
    stream_handler = logging.StreamHandler()
    # 定义日志格式
    formatter_file = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',datefmt='%Y-%m-%d,%H:%M:%S')
    formatter_stream = logging.Formatter('%(asctime)s | %(message)s',datefmt='%H:%M:%S')
    # 为处理器设置格式
    file_handler.setFormatter(formatter_file)
    stream_handler.setFormatter(formatter_stream)
    # 设置日志等级
    file_handler.setLevel(level)
    stream_handler.setLevel(level)
    # 获取根记录器
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    # 设置记录器日志等级
    root_logger.setLevel(level)
# # 使用示例
# setup_logging('app.log', logging.DEBUG)
# logging.info('This is an info message.')
# logging.error('This is an error message.')

## 随机种子
def seed_torch(seed=1029):
    # 为了禁止hash随机化，使得实验可复现
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

## 转换模型为fp32精度
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def logging_params(args):
    logging.info("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        logging.info(f"  {name}: {val}")