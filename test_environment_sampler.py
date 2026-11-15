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


if __name__ == "__main__":
    main()
