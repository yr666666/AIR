import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # 配置参数
    parser.add_argument("--train_result",
                        type=str,
                        default="train_result/",
                        help="file of save train result",)
    parser.add_argument("--exp_name",
                        type=str,
                        default=None,
                        help="name like train_result/exp.0, otherwise use current time",)
    parser.add_argument("--precision",
                        choices=["amp", "fp16", "fp32"],
                        default="amp",
                        help="floating point precition")
    parser.add_argument("--log_frequency",
                        type=int,
                        default=100,
                        help="log of batch frequency")
    parser.add_argument("--save_ckpt",
                        type=bool,
                        default=False,
                        help="Whether to save ckpt")

    # 数据集参数
    parser.add_argument("--train_json",
                        type=str,
                        default="/data2/ly/data/igarss/RSITMD/dataset_RSITMD.json",
                        help=".json like RSITMD dataset json")

    parser.add_argument("--train_img",
                        type=str,
                        default="/data2/ly/data/igarss/RSITMD/images/",
                        help="file of train image")
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="train and test batch size")

    # 模型参数
    parser.add_argument("--pretrain",
                        default="/data2/ly/clip_pretrain/ViT-B-16.pt",
                        type=str,
                        help="ViT-B-16.pt/ViT-B-32.pt/...", )

    parser.add_argument("--context_length",
                        default=17,
                        type=int,
                        help="context max length")

    # 优化器和学习率策略参数
    parser.add_argument("--beta1",
                        type=float,
                        default=0.9,
                        help="Adam beta1 for ViTB 16/32")
    parser.add_argument("--beta2",
                        type=float,
                        default=0.98,
                        help="Adam beta2 for ViTB 16/32")
    parser.add_argument("--eps",
                        type=float,
                        default=1.0e-6,
                        help="Adam epsilon for ViTB 16/32")
    parser.add_argument("--wd",
                        type=float,
                        default=0.001,
                        help="weight decay.")
    parser.add_argument("--epoch_num",
                        type=int,
                        default=10,
                        help="train epoch num")
    parser.add_argument("--lr",
                        type=float,
                        default=1.0e-5,
                        help="learning rate")
    parser.add_argument("--warmup",
                        type=int,
                        default=100,
                        help="number of steps to warmup for")

    args = parser.parse_args()
    return args
