import numpy as np
import torch
import yaml

from cfe_slr_network import CFESLRModel


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
    model_args["hidden_size"] = 128
    model_args["environment_size"] = 32
    model_args["pretrained_backbone"] = False
    model_args["gradient_checkpointing"] = False
    model_args["text_model_path"] = None
    loss_weights = dict(config["loss_weights"])
    loss_weights["PrivilegedTeacher"] = 0.0
    model = CFESLRModel(
        **model_args,
        gloss_dict=gloss_dict,
        loss_weights=loss_weights,
    ).cuda()
    model.train()

    batch_size = 3
    timesteps = 20
    videos = torch.empty(
        batch_size, timesteps, 3, 64, 64, device="cuda"
    ).uniform_(-1.0, 1.0)
    video_lengths = torch.tensor([20, 18, 16], dtype=torch.long, device="cuda")
    labels = torch.tensor([1, 2, 2, 3], dtype=torch.long, device="cuda")
    label_lengths = torch.tensor([2, 1, 1], dtype=torch.long, device="cuda")
    environment_ids = torch.tensor([0, 1, 2], dtype=torch.long, device="cuda")

    output = model(
        videos,
        video_lengths,
        label=labels,
        label_lgt=label_lengths,
        env_ids=environment_ids,
    )
    if output["counterfactual"] is None:
        raise AssertionError("Training forward did not generate an intervention view")
    loss = model.criterion_calculation(output, labels, label_lengths)
    loss.backward()
    torch.cuda.synchronize()

    if not torch.isfinite(loss):
        raise AssertionError("Loss is not finite: {}".format(loss.item()))
    for key in [
        "conv_logits",
        "sequence_logits",
        "environment_logits",
        "semantic_environment_logits",
    ]:
        if not torch.isfinite(output[key]).all():
            raise AssertionError("{} contains non-finite values".format(key))

    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    finite_gradients = sum(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in trainable
    )
    if finite_gradients != len(trainable):
        raise AssertionError(
            "Only {}/{} trainable parameter tensors have finite gradients".format(
                finite_gradients, len(trainable)
            )
        )

    print("loss={:.6f}".format(loss.item()))
    print("feat_len={}".format(output["feat_len"].tolist()))
    print(
        "conv_logits={} sequence_logits={}".format(
            tuple(output["conv_logits"].shape),
            tuple(output["sequence_logits"].shape),
        )
    )
    print("finite_gradients={}/{}".format(finite_gradients, len(trainable)))
    print(
        "max_gpu_memory_mb={:.1f}".format(
            torch.cuda.max_memory_allocated() / 1024 ** 2
        )
    )


if __name__ == "__main__":
    main()
