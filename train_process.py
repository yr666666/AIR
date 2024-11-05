from ipdb import set_trace
import logging
import os
import time

import numpy as np
import torch
from torch.cuda.amp import autocast

from loss import get_loss

##训练
def train(args, data, model, optimizer, epoch, scheduler, scaler):
    dataloader = data['train']
    num_batches_per_epoch = dataloader.num_batches
    model.train()

    batch_start_time = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        optimizer.zero_grad()

        images, texts = batch
        texts = texts.squeeze()
        images = images.cuda(non_blocking=True)
        texts = texts.cuda(non_blocking=True)
        data_time = time.time() - batch_start_time

        if args.precision == "amp":
            with autocast():#混合精度，提升运行速度
                itc_loss, KL_loss1 = get_loss(args, model, images, texts)
                total_loss = itc_loss + KL_loss1
                scaler.scale(total_loss).backward()#提升精度，放大损失来防止梯度的下溢
                scaler.step(optimizer)
            scaler.update()
        else:
            itc_loss, KL_loss1 = get_loss(args, model, images, texts)
            total_loss = itc_loss + KL_loss1
            total_loss.backward()
            optimizer.step()
        model_time = time.time() - batch_start_time - data_time
        batch_start_time = time.time()

        if (i % args.log_frequency) == 0:   #日志输出至控制台
            num_samples = i * len(images)
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * i / num_batches_per_epoch
            logging.info(
                        f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                        f"Loss_itc: {itc_loss.item():.6f}\tLoss_kl: {KL_loss1.item():.6f}\tTime_d: {data_time:.3f}\tTime_m: {model_time:.3f}"
                        f"\tLR: {optimizer.param_groups[0]['lr']:5f}\tlogit_scale {model.logit_scale.data:.3f}"
                        )

#测试
def test(model, data, epoch, args, split=None):
    logging.info(f"Begin to eval epoch: {epoch}...")
    dataloader = data[split]
    model.eval()

    all_image_features, all_text_features = [], []
    with torch.no_grad():
        for i,batch in enumerate(dataloader):
            images, texts = batch
            texts = texts.squeeze()
            images = images.cuda(non_blocking=True)
            texts = texts.cuda(non_blocking=True)
            image_features, text_features, logit_scale = model(images, texts)
                
            all_image_features.append(image_features.data.cpu().numpy().copy())
            all_text_features.append(text_features.data.cpu().numpy().copy())
        all_image_features = np.concatenate(all_image_features,axis=0)
        all_text_features = np.concatenate(all_text_features,axis=0)
        all_image_features = np.array([all_image_features[i] for i in range(0, len(all_image_features), 5)])

    return all_image_features, all_text_features

## 评估
def eval(args, data, split, model, optimizer, epoch, best_rsum, best_epoch):
    all_image_features, all_text_features = test(model,data,epoch + 1,args,split,)
    (ir1, ir5, ir10, imedr, imeanr) = i2t5(all_image_features, all_text_features,npts=None, return_ranks=False)
    (tr1, tr5, tr10, tmedr, tmeanr) = t2i5(all_image_features, all_text_features,npts=None, return_ranks=False)
    rsum = ir1+ ir5+ ir10+ tr1+ tr5+ tr10

    logging.info(split + "   Image to text: %.2f, %.2f, %.2f, %.2f, %.2f" %(ir1, ir5, ir10, imedr, imeanr))
    logging.info(split + "   Text to image: %.2f, %.2f, %.2f, %.2f, %.2f" %(tr1, tr5, tr10, tmedr, tmeanr))

    # remember best R@ sum and save checkpoint
    is_best = rsum > best_rsum
    # is_best = False
    if is_best :
        best_rsum = max(rsum, best_rsum)
        best_epoch = epoch
        if split=="test" and args.save_ckpt==True:
            torch.save({
                            "epoch": epoch + 1,
                            "name": args.exp_name,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            },
                       os.path.join(args.result_path, f"best.pt"),
                       )
    logging.info(split + "    当前epoch的rsum: %.3f" %(rsum))
    logging.info(split + "    best_rsum: %.3f" %(best_rsum))
    logging.info(split + "    best_epoch: %.1f" %(best_epoch))

    return best_rsum, best_epoch

def i2t5(images, captions, npts=None, return_ranks=False):
    sims = np.dot(images, captions.T)
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

def t2i5(images, captions, npts=None, return_ranks=False):
    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = np.dot(captions, images.T)
    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)