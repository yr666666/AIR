import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_loss(args, model, images, texts):
    # 前向传播
    image_features, text_features, logit_scale = model(images, texts)
    logit_scale = logit_scale.mean()
    # 计算相似度矩阵
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logit_scale * text_features @ image_features.t()
    # 构造标签
    ground_truth = torch.arange(len(logits_per_image)).long().cuda(non_blocking=True)  #torch.Size([32])
    # 计算损失
    CE_loss = nn.CrossEntropyLoss().cuda()
    itc_loss = (CE_loss(logits_per_image, ground_truth)+ CE_loss(logits_per_text, ground_truth)) / 2

    KL_loss1 = 30*mlce_loss(image_features, text_features, tem=1)
    return itc_loss,KL_loss1

def mlce_loss(features, embedding, tem):

    cosine_distance_l = 0.5 * (1 + torch.mm(features, features.t())) 
    cosine_distance_h = 0.5 * (1 + torch.mm(embedding, embedding.t()))
    W_h0 = torch.softmax(cosine_distance_h / tem, dim=0).t()
    W_l0 = torch.softmax(cosine_distance_l / tem, dim=0).t()

    cross_loss0 = F.kl_div(W_l0.log(), W_h0, reduction='sum')

    knowledge_loss = cross_loss0.mean()
    return knowledge_loss