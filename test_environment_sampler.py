from collections import Counter

from dataset.dataloader_video_cfe import EnvironmentBalancedBatchSampler


def main():
    sample_environments = [0] * 15 + [1] * 15 + [2] * 15
    sampler = EnvironmentBalancedBatchSampler(
        sample_environments=sample_environments,
        batch_size=2,
        drop_last=True,
        shuffle=True,
    )
    batches = list(sampler)
    if len(batches) != len(sampler):
        raise AssertionError("Sampler length does not match iteration count")
    if any(len(batch) != 2 for batch in batches):
        raise AssertionError("Every retained batch must have batch_size samples")
    if any(
        sample_environments[batch[0]] == sample_environments[batch[1]]
        for batch in batches
    ):
        raise AssertionError("Environment-balanced batches must mix signers")

    observed = Counter()
    for batch in batches:
        observed.update(batch)
    expected_samples = (len(sample_environments) // 2) * 2
    if len(observed) != expected_samples:
        raise AssertionError("Sampler did not retain the expected sample count")

    counts = Counter(
        sample_environments[index]
        for batch in batches
        for index in batch
    )
    if max(counts.values()) - min(counts.values()) > 1:
        raise AssertionError("Environment coverage is not balanced")
    print("batches={}".format(len(batches)))
    print("environment_counts={}".format(dict(sorted(counts.items()))))

    real_signer_counts = [
        1475, 49, 470, 836, 1296, 30, 646, 704, 165
    ]
    sample_environments = [
        environment
        for environment, count in enumerate(real_signer_counts)
        for _ in range(count)
    ]
    sampler = EnvironmentBalancedBatchSampler(
        sample_environments=sample_environments,
        batch_size=2,
        drop_last=True,
        shuffle=True,
    )
    batches = list(sampler)
    sampled_samples = sum(len(batch) for batch in batches)
    expected_samples = len(sample_environments) - (
        len(sample_environments) % 2
    )
    if sampled_samples != expected_samples:
        raise AssertionError(
            "Imbalanced sampler dropped more than the final partial batch"
        )
    if sampler.last_epoch_stats["sampled_samples"] != sampled_samples:
        raise AssertionError("Sampler statistics are inconsistent")

    oversized_counts = [100, 1, 1, 1]
    sample_environments = [
        environment
        for environment, count in enumerate(oversized_counts)
        for _ in range(count)
    ]
    sampler = EnvironmentBalancedBatchSampler(
        sample_environments=sample_environments,
        batch_size=8,
        drop_last=True,
        shuffle=False,
    )
    batches = list(sampler)
    if len(batches) != len(sampler):
        raise AssertionError("Oversized batches ended before __len__")
    if any(len(batch) != 8 for batch in batches):
        raise AssertionError("Remaining samples did not fill full batches")
    if sampler.last_epoch_stats["dropped_samples"] != (
        len(sample_environments) % 8
    ):
        raise AssertionError("Oversized batch test dropped unexpected samples")

    print("imbalanced_sampler=passed")
    print("sampler_stats=passed")


if __name__ == "__main__":
    main()
