import types

import torch
import torch.nn as nn

from seq_scripts import seq_train


class NonFiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("nan"))


class GuardedModel(nn.Module):
    def __init__(self):
        super(GuardedModel, self).__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.batch_index = 0

    def forward(self, videos, lengths, **kwargs):
        return {}

    def criterion_calculation(self, outputs, labels, label_lengths):
        index = self.batch_index
        self.batch_index += 1
        if index == 0:
            return self.weight * float("nan")
        if index == 1:
            return NonFiniteGradient.apply(self.weight)
        return self.weight.square()


class DummyDevice(object):
    @staticmethod
    def data_to_device(data):
        return data


class DummyRecorder(object):
    def __init__(self):
        self.messages = []
        self.log_interval = 20

    def print_log(self, message, path=None):
        self.messages.append(message)


class DummyOptimizer(object):
    def __init__(self, model):
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        self.scheduler = types.SimpleNamespace(step=lambda: None)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()


def main():
    model = GuardedModel()
    optimizer = DummyOptimizer(model)
    recorder = DummyRecorder()
    batch = (
        torch.zeros(1),
        torch.ones(1),
        torch.zeros(1),
        torch.ones(1),
        ("sample",),
        torch.zeros(1),
    )
    loader = [batch, batch, batch]

    losses = seq_train(
        loader,
        model,
        optimizer,
        DummyDevice(),
        epoch_idx=0,
        recoder=recorder,
        grad_clip=5.0,
    )

    if len(losses) != 1:
        raise AssertionError("Only the finite-loss batch should be recorded")
    if model.batch_index != 3:
        raise AssertionError("All three batches should be processed")
    if not torch.isfinite(model.weight).item():
        raise AssertionError("Skipped gradients contaminated model parameters")
    if not any("non-finite loss" in message for message in recorder.messages):
        raise AssertionError("Non-finite loss was not reported")
    if not any("non-finite gradient" in message for message in recorder.messages):
        raise AssertionError("Non-finite gradient was not reported")
    print("nonfinite_guards=passed")


if __name__ == "__main__":
    main()
