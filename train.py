from ipdb import set_trace
import logging
import os
import time

import numpy as np
import torch
from torch import optim
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler

from data import get_data
from train_process import train, eval

from adds.params import parse_args
from adds.scheduler import cosine_lr
from adds.utils import setup_logging,seed_torch,convert_models_to_fp32,logging_params
import models.clip as clip
from models.model import convert_weights


### 总训练过程
def main():
    ##1.配置
    # 获取超参数
    args = parse_args()
    # 配置当前实验的结果文件夹
    if args.exp_name is None:
        args.exp_name = time.strftime(
                                f"epoch={args.epoch_num}_"
                                f"lr={args.lr}_"
                                f"date=%Y-%m-%d-%H-%M-%S",
                                time.localtime()
                                )
    args.result_path = os.path.join(args.train_result, args.exp_name)
    os.makedirs(args.result_path, exist_ok=True)
    # 配置日志
    args.log_path = os.path.join(args.result_path, "log.log")
    args.log_level = logging.INFO
    logging.getLogger().handlers.clear()
    setup_logging(args.log_path, args.log_level)
    # 其他配置
    seed_torch(seed=13)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)

    ##2.实例化
    # 将超参数输出日志
    logging_params(args)
    # 数据集
    context_length = args.context_length  #若为None则文本的最大长度默认为model.state_dict()["positional_embedding"].shape[0]
    if not context_length:
        context_length = model.state_dict()["positional_embedding"].shape[0]
    data = get_data(args, max_txt_length=context_length)

    # 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(args.pretrain, device=device, context_length=context_length)
    if args.precision == "amp" or args.precision == "fp32":
        convert_models_to_fp32(model)
    if args.precision == "fp16":
        convert_weights(model)
    if not torch.cuda.is_available():
        model.float()
        logging.warning("using CPU, this will be slow")

    # 优化器和学习率策略
    exclude = lambda n : "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n : not exclude(n)
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

    optimizer = optim.AdamW([
                                    {"params": gain_or_bias_params, "weight_decay": 0.},
                                    {"params": rest_params, "weight_decay": args.wd},
                                    ],
                            lr=args.lr,
                            betas=(args.beta1, args.beta2),
                            eps=args.eps,
                           )

    total_steps = data["train"].num_batches * args.epoch_num
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
    scaler = GradScaler() if args.precision == "amp" else None

    ##3.zeroshot、train、test
    train_start_time = time.time()
    cudnn.benchmark = True
    cudnn.deterministic = False
    start_epoch = 0
    val_best_rsum, val_best_epoch = 0, 0
    test_best_rsum, test_best_epoch = 0, 0
    for epoch in range(start_epoch, args.epoch_num):
        logging.info(f'Start epoch {epoch}')
        if epoch == start_epoch:
            logging.info("------------------=================="+ "zero shot val" + "=================------------------")
            val_best_rsum, val_best_epoch = eval(args,data,"val",model,optimizer,epoch,val_best_rsum,val_best_epoch)
            logging.info("------------------=================="+ "zero shot test" + "=================------------------")
            test_best_rsum, test_best_epoch = eval(args,data,"test",model,optimizer,epoch,test_best_rsum,test_best_epoch)

        train(args,data,model,optimizer,epoch,scheduler,scaler)
        logging.info("------------------=================="+ "val" + "=================------------------")
        val_best_rsum, val_best_epoch = eval(args,data,"val",model,optimizer,epoch,val_best_rsum,val_best_epoch)
        logging.info("------------------=================="+ "test" + "=================------------------")
        test_best_rsum, test_best_epoch = eval(args,data,"test",model,optimizer,epoch,test_best_rsum,test_best_epoch)

    print("本次训练总时间：", time.time()-train_start_time)

if __name__ == "__main__":
    main()
