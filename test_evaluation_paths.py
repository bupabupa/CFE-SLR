import os

from evaluation.slr_eval.wer_calculation import resolve_groundtruth_path


def main():
    relative = resolve_groundtruth_path(
        "/tmp/evaluation",
        "phoenix2014-groundtruth",
        "dev",
    )
    expected_relative = os.path.join(
        "/tmp/evaluation",
        "phoenix2014-groundtruth-dev.stm",
    )
    if relative != expected_relative:
        raise AssertionError("Relative ground-truth path resolution failed")

    absolute_prefix = "/data/phoenix/phoenix2014-groundtruth"
    absolute = resolve_groundtruth_path(
        "/tmp/evaluation",
        absolute_prefix,
        "test",
    )
    if absolute != absolute_prefix + "-test.stm":
        raise AssertionError("Absolute ground-truth path resolution failed")

    print("evaluation_paths=passed")


if __name__ == "__main__":
    main()
