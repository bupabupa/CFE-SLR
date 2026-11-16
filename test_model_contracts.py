import torch
import torch.nn as nn

from cfe_slr_network import (
    CFEResNetEncoder,
    NormLinear,
    _temporary_eval_mode,
    _unwrap_parallel_module,
)
from modules.tconv import TemporalConv
from utils.device import GpuDataParallel


def count_batchnorm(module):
    return sum(
        isinstance(child, nn.modules.batchnorm._BatchNorm)
        for child in module.modules()
    )


def main():
    temporal_without_bn = TemporalConv(
        input_size=8,
        hidden_size=16,
        conv_type=2,
        use_bn=False,
        num_classes=5,
    )
    temporal_with_bn = TemporalConv(
        input_size=8,
        hidden_size=16,
        conv_type=2,
        use_bn=True,
        num_classes=5,
    )
    if count_batchnorm(temporal_without_bn) != 0:
        raise AssertionError("use_bn=False still inserted BatchNorm1d")
    if count_batchnorm(temporal_with_bn) != 2:
        raise AssertionError("use_bn=True did not insert the expected BatchNorm1d")

    norm_linear = NormLinear(16, 5)
    features = torch.randn(3, 16, requires_grad=True)
    logits = norm_linear(features)
    logits.sum().backward()
    if logits.shape != (3, 5):
        raise AssertionError("NormLinear produced an unexpected shape")
    if norm_linear.weight.grad is None:
        raise AssertionError("NormLinear did not receive gradients")

    checkpointed_encoder = CFEResNetEncoder(
        c2d_type="resnet18",
        pretrained=False,
        gradient_checkpointing=True,
    )
    checkpointed_encoder.train()
    if any(
        child.training
        for child in checkpointed_encoder.modules()
        if isinstance(child, nn.modules.batchnorm._BatchNorm)
    ):
        raise AssertionError("Checkpointed encoder must freeze BN statistics")

    cpu_device = GpuDataParallel()
    cpu_device.set_device("None")
    if cpu_device.output_device != "cpu" or cpu_device.gpu_list:
        raise AssertionError("CPU device selection failed")

    wrapped_encoder = nn.DataParallel(
        checkpointed_encoder,
        device_ids=[0],
    )
    if _unwrap_parallel_module(wrapped_encoder) is not checkpointed_encoder:
        raise AssertionError("Parallel encoder unwrapping failed")

    class DeterministicHarness(nn.Module):
        def __init__(self):
            super(DeterministicHarness, self).__init__()
            self.batch_norm = nn.BatchNorm1d(4)
            self.dropout = nn.Dropout(p=0.5)

        def forward(self, inputs):
            return self.batch_norm(self.dropout(inputs))

    deterministic_harness = DeterministicHarness()
    deterministic_harness.train()
    running_mean = deterministic_harness.batch_norm.running_mean.clone()
    inputs = torch.randn(8, 4)
    with _temporary_eval_mode(deterministic_harness):
        first = deterministic_harness(inputs)
        second = deterministic_harness(inputs)
        if not torch.allclose(first, second):
            raise AssertionError("Temporary eval mode left dropout enabled")
        if not torch.equal(
            deterministic_harness.batch_norm.running_mean,
            running_mean,
        ):
            raise AssertionError("Temporary eval mode updated BatchNorm stats")
    if not deterministic_harness.training:
        raise AssertionError("Temporary eval mode did not restore training mode")

    print("temporal_conv_use_bn=passed")
    print("norm_linear=passed")
    print("checkpoint_batchnorm=passed")
    print("cpu_device=passed")
    print("parallel_unwrap=passed")
    print("deterministic_intervention=passed")


if __name__ == "__main__":
    main()
