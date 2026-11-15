import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from cfe_slr_network import CFESLRModel
from dataset.dataloader_video_cfe import CFEFeeder


def main():
    torch.manual_seed(0)
    with open("configs/cfe_phoenix14.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    gloss_dict = np.load(
        "/var/lib/docker/wd/Sign/codes/VAC_CSLR/preprocess/phoenix2014/gloss_dict.npy",
        allow_pickle=True,
    ).item()
    model_args = dict(config["model_args"])
    model_args["num_classes"] = len(gloss_dict) + 1
    model_args["hidden_size"] = 256
    model_args["environment_size"] = 64
    model_args["pretrained_backbone"] = False
    model = CFESLRModel(
        **model_args,
        gloss_dict=gloss_dict,
        loss_weights=config["loss_weights"],
    ).cuda()
    model.train()

    feeder_args = dict(config["feeder_args"])
    feeder_args["mode"] = "train"
    feeder_args["transform_mode"] = True
    dataset = CFEFeeder(
        gloss_dict=gloss_dict,
        prefix=(
            "/var/lib/docker/wd/Sign/dataset/phoenix2014-release/"
            "phoenix-2014-multisigner"
        ),
        **feeder_args
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True,
        num_workers=2,
        collate_fn=dataset.collate_fn,
    )
    videos, video_lengths, labels, label_lengths, _, env_ids = next(iter(loader))
    videos = videos.cuda()
    video_lengths = video_lengths.cuda()
    labels = labels.cuda()
    label_lengths = label_lengths.cuda()
    env_ids = env_ids.cuda()
    print(
        "videos={} video_lengths={} labels={} env_ids={}".format(
            tuple(videos.shape),
            video_lengths.tolist(),
            label_lengths.tolist(),
            env_ids.tolist(),
        )
    )

    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    scaler = torch.cuda.amp.GradScaler()
    started = time.time()
    with torch.cuda.amp.autocast():
        output = model(
            videos,
            video_lengths,
            label=labels,
            label_lgt=label_lengths,
            env_ids=env_ids,
        )
        loss = model.criterion_calculation(output, labels, label_lengths)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    elapsed = time.time() - started
    if not torch.isfinite(loss):
        raise AssertionError("Real-batch loss is not finite")
    print("loss={:.6f} step_seconds={:.3f}".format(loss.item(), elapsed))

    model.eval()
    with torch.no_grad():
        eval_output = model(videos, video_lengths, env_ids=env_ids)
    if len(eval_output["recognized_sents"]) != videos.size(0):
        raise AssertionError("Inference did not produce one prediction per video")
    print("recognized_sents={}".format(len(eval_output["recognized_sents"])))
    print(
        "max_gpu_memory_mb={:.1f}".format(
            torch.cuda.max_memory_allocated() / 1024 ** 2
        )
    )


if __name__ == "__main__":
    main()
