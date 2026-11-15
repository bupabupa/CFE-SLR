import os

import numpy as np
import torch
import yaml

from slr_cem_network import CEMSLRModel


def main():
    torch.manual_seed(0)
    with open("configs/cem_phoenix14.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    gloss_dict = np.load(
        config["dataset_info"]["dict_path"]
        if "dataset_info" in config
        else "preprocess/phoenix2014/gloss_dict.npy",
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

    batch_size = 2
    timesteps = 48
    videos = torch.rand(batch_size, timesteps, 3, 224, 224, device="cuda")
    video_lengths = torch.tensor([48, 44], dtype=torch.long, device="cuda")
    labels = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.long, device="cuda")
    label_lengths = torch.tensor([4, 3], dtype=torch.long, device="cuda")

    output = model(
        videos,
        video_lengths,
        label=labels,
        label_lgt=label_lengths,
    )
    for key in ["feat_len", "frame_length"]:
        print("{}={} {}".format(key, tuple(output[key].shape), output[key].tolist()))
    for key in ["conv_logits", "sequence_logits", "cfa_logits"]:
        print("{}={}".format(key, tuple(output[key].shape)))
    loss = model.criterion_calculation(output, labels, label_lengths)
    loss.backward()
    torch.cuda.synchronize()

    print("loss={:.6f}".format(loss.item()))
    for key in [
        "conv_logits",
        "sequence_logits",
        "cfa_logits",
        "visual_gloss_logits",
        "fused_gloss_logits",
        "counterfactual_gloss_logits",
    ]:
        print("{}={}".format(key, tuple(output[key].shape)))
    finite_gradients = sum(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    trainable = sum(parameter.requires_grad for parameter in model.parameters())
    print("finite_gradients={}/{}".format(finite_gradients, trainable))
    print("max_gpu_memory_mb={:.1f}".format(torch.cuda.max_memory_allocated() / 1024 ** 2))


if __name__ == "__main__":
    main()
