from ipdb import set_trace
import json as jsonmod
import logging
import os
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from models.clip import tokenize


def _convert_to_rgb(image):
    return image.convert('RGB')

def build_transform(resolution=224):
    normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    return Compose([
                    Resize(resolution, interpolation=Image.BICUBIC), #扩大分辨率
                    CenterCrop(resolution),                          #中心裁剪
                    _convert_to_rgb,                                 #转换成RGB三通道
                    ToTensor(),
                    normalize,
                    ])

class JsonDataset(Dataset):
    def __init__(self, args, json_filename, img_filename, split="val", max_txt_length=24):
        self.transform = build_transform()
        json_dir = json_filename
        self.root = img_filename
        self.dataset = jsonmod.load(open(json_dir, 'r'))['images']
        self.ids = []
        
        for i, ds in enumerate(self.dataset):
            if ds['split'] == split:
                #五个句子
                if len(ds['sentences']) != 5:
                    logging.info(split)
                    logging.info(len(ds['sentences']))
                self.ids += [(i, x) for x in range(len(ds['sentences']))]

        self.length = len(self.ids)
        self.split = split
        self.max_txt_length = max_txt_length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        root = self.root
        ann_id = self.ids[idx]
        img_id = ann_id[0]
        cap_id = ann_id[1]
        caption = self.dataset[img_id]['sentences'][cap_id]['raw']
        path = self.dataset[img_id]['filename']

        image = Image.open(os.path.join(root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        text = tokenize([str(caption)])
        # tokenize([str(caption)], context_length=self.max_txt_length)[0]
        # eos_index = text.numpy().tolist().index(_tokenizer.vocab['[SEP]'])
        # set_trce()
        return image, text#, eos_index

#定义dataset和dataload
def get_dataset(args, is_train, split=None, max_txt_length=24):
    input_json = args.train_json
    img_filename = args.train_img
    dataset = JsonDataset(args, input_json, img_filename, split=split, max_txt_length=max_txt_length)
    num_samples = len(dataset)

    logging.info(split+"_数据总量: "+str(len(dataset)))
    dataloader = DataLoader(dataset,batch_size=args.batch_size,shuffle=is_train,
                            num_workers=0,pin_memory=True,drop_last=is_train)
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return dataloader
    
#将训练的验证的数据传入
def get_data(args, max_txt_length=24):
    data = {}

    if args.train_json:
        split = "train"
        data["train"] = get_dataset(args, is_train=True, split=split, max_txt_length=max_txt_length)
    # if args.val_data:
        split = "val"
        data["val"] = get_dataset(args, is_train=False, split=split, max_txt_length=max_txt_length)
        split = "test"
        data["test"] = get_dataset(args, is_train=False, split=split, max_txt_length=max_txt_length)
        if args.train_json.split("/")[-2] == "RSITMD":
            data["val"] = data["test"]

    return data
