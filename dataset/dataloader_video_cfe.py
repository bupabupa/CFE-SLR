import os

import numpy as np
import torch

from dataset.dataloader_video import BaseFeeder


class CFEFeeder(BaseFeeder):
    """Video feeder that also returns the signer/environment group id."""

    def __init__(
            self,
            prefix,
            gloss_dict,
            info_dir="./preprocess/phoenix2014",
            environment_field="signer",
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
