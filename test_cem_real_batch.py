import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset.dataloader_video import BaseFeeder
from slr_cem_network import CEMSLRModel


def main():
    torch.manual_seed(0)
    with open("configs/cem_phoenix14.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    gloss_dict = np.load(
        "preprocess/phoenix2014/gloss_dict.npy",
        allow_pickle=True,
    ).item()
    model_args = dict(config["model_args"])
    model_args["num_classes"] = len(gloss_dict) + 1
    model = CEMSLRModel(
        **model_args,
        gloss_dict=gloss_dict,
        loss_weights=config["loss_weights"],
    ).cuda()
    model.train()

    feeder_args = dict(config["feeder_args"])
    feeder_args["mode"] = "train"
    feeder_args["transform_mode"] = True
    dataset = BaseFeeder(
        gloss_dict=gloss_dict,
        prefix="./dataset/phoenix2014/phoenix-2014-multisigner",
        **feeder_args,
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        drop_last=True,
        num_workers=2,
        collate_fn=dataset.collate_fn,
    )
    videos, video_lengths, labels, label_lengths, _ = next(iter(loader))
    videos = videos.cuda()
    video_lengths = video_lengths.cuda()
    labels = labels.cuda()
    label_lengths = label_lengths.cuda()
    print("videos={} video_lengths={} labels={}".format(
        tuple(videos.shape),
        video_lengths.tolist(),
        label_lengths.tolist(),
    ))

    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    scaler = torch.cuda.amp.GradScaler()
    started = time.time()
    with torch.cuda.amp.autocast():
        output = model(videos, video_lengths, label=labels, label_lgt=label_lengths)
        loss = model.criterion_calculation(output, labels, label_lengths)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    elapsed = time.time() - started
    print("loss={:.6f} step_seconds={:.3f}".format(loss.item(), elapsed))
    print("max_gpu_memory_mb={:.1f}".format(torch.cuda.max_memory_allocated() / 1024 ** 2))


if __name__ == "__main__":
    main()
