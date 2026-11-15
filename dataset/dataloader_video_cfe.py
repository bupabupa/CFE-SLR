import os
import math
import random

import numpy as np
import torch
from torch.utils.data.sampler import Sampler

from dataset.dataloader_video import BaseFeeder


class EnvironmentBalancedBatchSampler(Sampler):
    """Build batches that contain samples from as many environments as possible."""

    def __init__(
            self,
            sample_environments,
            batch_size,
            drop_last=True,
            shuffle=True
    ):
        self.sample_environments = list(sample_environments)
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.shuffle = shuffle
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        total = len(self.sample_environments)
        if self.drop_last:
            self.batch_count = total // self.batch_size
        else:
            self.batch_count = int(math.ceil(total / float(self.batch_size)))

    def __iter__(self):
        queues = {}
        for index, environment in enumerate(self.sample_environments):
            queues.setdefault(environment, []).append(index)
        for indices in queues.values():
            if self.shuffle:
                random.shuffle(indices)

        for _ in range(self.batch_count):
            active_environments = [
                environment
                for environment, indices in queues.items()
                if indices
            ]
            if self.shuffle:
                random.shuffle(active_environments)
            active_environments.sort(
                key=lambda environment: len(queues[environment]),
                reverse=True,
            )
            batch = [
                queues[environment].pop()
                for environment in active_environments[:self.batch_size]
            ]

            if len(batch) < self.batch_size and self.drop_last:
                break
            if batch:
                yield batch

    def __len__(self):
        return self.batch_count


class CFEFeeder(BaseFeeder):
    """Video feeder that also returns the signer/environment group id."""

    def __init__(
            self,
            prefix,
            gloss_dict,
            info_dir="./preprocess/phoenix2014",
            environment_field="signer",
            environment_balanced=False,
            **kwargs
    ):
        super(CFEFeeder, self).__init__(
            prefix=prefix,
            gloss_dict=gloss_dict,
            info_dir=info_dir,
            **kwargs
    )
        self.environment_field = environment_field
        train_info = np.load(
            os.path.join(info_dir, "train_info.npy"),
            allow_pickle=True,
        ).item()
        environments = sorted(
            set(
                sample[environment_field]
                for sample in train_info.values()
                if isinstance(sample, dict) and environment_field in sample
            )
        )
        if not environments:
            raise ValueError(
                "No environments found in train metadata field '{}'".format(
                    environment_field
                )
            )
        self.environment_to_id = {
            environment: index for index, environment in enumerate(environments)
        }
        self.num_environments = len(environments)
        self.environment_balanced = bool(environment_balanced)
        sample_count = len(self.inputs_list) - 1
        self.sample_environments = [
            self.environment_id(self.inputs_list[index]["original_info"])
            for index in range(sample_count)
        ]

    def environment_balanced_batch_sampler(self, batch_size, drop_last=True):
        return EnvironmentBalancedBatchSampler(
            sample_environments=self.sample_environments,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=True,
        )

    def environment_id(self, original_info):
        parts = original_info.split("|")
        if len(parts) < 3:
            raise ValueError(
                "Expected signer information in original_info: {}".format(
                    original_info
                )
            )
        environment = parts[2]
        if environment not in self.environment_to_id:
            raise ValueError(
                "Unknown environment '{}'; training environments are {}".format(
                    environment,
                    sorted(self.environment_to_id),
                )
            )
        return self.environment_to_id[environment]

    def __getitem__(self, index):
        input_data, label, original_info = super(CFEFeeder, self).__getitem__(index)
        environment = torch.tensor(
            self.environment_id(original_info),
            dtype=torch.long,
        )
        return input_data, label, original_info, environment

    def collate_fn(self, batch):
        environment_ids = torch.stack([sample[3] for sample in batch])
        base_batch = super(CFEFeeder, self).collate_fn(
            [sample[:3] for sample in batch]
        )
        return tuple(base_batch) + (environment_ids,)
