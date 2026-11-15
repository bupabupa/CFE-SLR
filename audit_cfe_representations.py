import argparse

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from cfe_slr_network import CFESLRModel
from dataset.dataloader_video_cfe import CFEFeeder


def encode_loader(model, loader, device, max_batches):
    features = []
    environments = []
    consistency = []
    model.eval()
    with torch.no_grad():
        for batch_index, data in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            videos = data[0].to(device)
            lengths = data[1].to(device)
            env_ids = data[5].to(device)
            output = model(
                videos,
                lengths,
                env_ids=env_ids,
                return_counterfactual=True,
            )
            features.append(output["semantic_pooled"].detach().cpu())
            environments.append(env_ids.detach().cpu())
            if output["counterfactual"] is not None:
                student = output["sequence_logits"].detach()
                target = output["counterfactual"]["sequence_logits"].detach()
                teacher_probs = target.softmax(dim=-1)
                losses = (
                    teacher_probs
                    * (
                        teacher_probs.clamp(min=1.0e-8).log()
                        - student.log_softmax(dim=-1)
                    )
                ).sum(dim=-1)
                mask = torch.arange(losses.size(0), device=losses.device).unsqueeze(1)
                mask = mask < output["feat_len"].to(losses.device).unsqueeze(0)
                consistency.append(losses[mask].mean().cpu())
    return (
        torch.cat(features, dim=0),
        torch.cat(environments, dim=0),
        float(torch.stack(consistency).mean()) if consistency else float("nan"),
    )


def linear_probe_accuracy(features, targets, seed=0):
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(features.size(0), generator=generator)
    split = max(1, int(features.size(0) * 0.8))
    train_index = order[:split]
    test_index = order[split:]
    if test_index.numel() == 0:
        test_index = train_index

    features = F.normalize(features, dim=1)
    probe = torch.nn.Linear(features.size(1), int(targets.max().item()) + 1)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1.0e-2, weight_decay=1.0e-3)
    for _ in range(300):
        optimizer.zero_grad()
        loss = F.cross_entropy(probe(features[train_index]), targets[train_index])
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        predictions = probe(features[test_index]).argmax(dim=1)
        accuracy = (predictions == targets[test_index]).float().mean().item()
    counts = torch.bincount(targets[train_index]).float()
    majority = counts.max().item() / counts.sum().item()
    return accuracy, majority


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    with open("configs/cfe_phoenix14.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    gloss_dict = np.load(
        "/var/lib/docker/wd/Sign/codes/VAC_CSLR/preprocess/phoenix2014/gloss_dict.npy",
        allow_pickle=True,
    ).item()
    model_args = dict(config["model_args"])
    model_args["num_classes"] = len(gloss_dict) + 1
    model = CFESLRModel(
        **model_args,
        gloss_dict=gloss_dict,
        loss_weights=config["loss_weights"],
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(args.device)

    feeder_args = dict(config["feeder_args"])
    feeder_args["mode"] = "dev"
    feeder_args["transform_mode"] = False
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
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        collate_fn=dataset.collate_fn,
    )
    features, environments, consistency = encode_loader(
        model,
        loader,
        args.device,
        args.max_batches,
    )
    accuracy, majority = linear_probe_accuracy(features, environments)
    print("samples={}".format(features.size(0)))
    print("signer_probe_accuracy={:.4f}".format(accuracy))
    print("majority_baseline={:.4f}".format(majority))
    print("counterfactual_consistency_kl={:.6f}".format(consistency))
    print(
        "note=Lower signer probe accuracy and lower consistency KL are better; "
        "neither result alone identifies a causal effect."
    )


if __name__ == "__main__":
    main()
