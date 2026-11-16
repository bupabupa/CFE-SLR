from types import SimpleNamespace

import torch

from main import Processor


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, environment_balanced):
        self.environment_balanced = environment_balanced

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return index

    def collate_fn(self, batch):
        return tuple(batch)

    def environment_balanced_batch_sampler(self, batch_size, drop_last=True):
        sampler = torch.utils.data.SequentialSampler(self)
        return torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=drop_last,
        )


def build_processor():
    processor = Processor.__new__(Processor)
    processor.arg = SimpleNamespace(
        batch_size=2,
        test_batch_size=2,
        num_worker=0,
    )
    return processor


def main():
    processor = build_processor()
    for environment_balanced in (False, True):
        dataset = DummyDataset(environment_balanced)
        loader = processor.build_dataloader(
            dataset,
            mode="train",
            train_flag=True,
        )
        batch = next(iter(loader))
        if len(batch) != 2:
            raise AssertionError("DataLoader did not call the bound collate_fn")
    print("bound_collate_fn=passed")


if __name__ == "__main__":
    main()
