import torch
import torch.nn as nn
import torch.nn.functional as F

from cfe_slr_network import CFESLRModel


class DummyTextTeacher(nn.Module):
    def __init__(self, num_classes):
        super(DummyTextTeacher, self).__init__()
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, gloss_ids, gloss_mask):
        logits = F.one_hot(
            gloss_ids.clamp(min=0),
            num_classes=self.bias.numel(),
        ).float() * 2.0
        logits = logits + self.bias
        return logits * gloss_mask.unsqueeze(-1).float()


class TeacherHarness(nn.Module):
    def __init__(self, num_classes):
        super(TeacherHarness, self).__init__()
        self.text_teacher = DummyTextTeacher(num_classes)
        self.teacher_temperature = 2.0


def main():
    num_classes = 12
    frames = 8
    harness = TeacherHarness(num_classes)
    sequence_logits = torch.full((frames, 1, num_classes), -16.0)
    for timestep, class_index in [
        (0, 0),
        (1, 4),
        (2, 4),
        (3, 0),
        (4, 7),
        (5, 7),
        (6, 0),
        (7, 0),
    ]:
        sequence_logits[timestep, 0, class_index] = 16.0
    sequence_logits.requires_grad_(True)
    output = {
        "sequence_logits": sequence_logits,
        "feat_len": torch.tensor([frames]),
    }
    labels = torch.tensor([4, 7])
    label_lengths = torch.tensor([2])

    loss = CFESLRModel._privileged_teacher_loss(
        harness,
        output,
        labels,
        label_lengths,
    )
    loss.backward()

    if not torch.isfinite(loss):
        raise AssertionError("Teacher distillation loss is not finite")
    if (
        harness.text_teacher.bias.grad is None
        or not torch.isfinite(harness.text_teacher.bias.grad).all()
        or harness.text_teacher.bias.grad.abs().sum().item() == 0.0
    ):
        raise AssertionError("Privileged teacher does not receive CE gradients")
    if (
        sequence_logits.grad is None
        or not torch.isfinite(sequence_logits.grad).all()
        or sequence_logits.grad.abs().sum().item() == 0.0
    ):
        raise AssertionError("Visual student does not receive distillation gradients")

    print("teacher_distillation_loss={:.6f}".format(loss.item()))
    print("teacher_gradient=passed")
    print("student_gradient=passed")


if __name__ == "__main__":
    main()
