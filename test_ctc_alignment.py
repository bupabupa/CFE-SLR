import torch

from cfe_slr_network import _ctc_forced_align


def make_log_probs(frames, classes, peak_classes):
    logits = torch.full((frames, classes), -16.0)
    for timestep, class_index in peak_classes:
        logits[timestep, class_index] = 16.0
    return logits.log_softmax(dim=-1)


def assert_alignment(name, log_probs, labels, expected):
    aligned = _ctc_forced_align(
        log_probs,
        torch.tensor(labels),
        len(labels),
        log_probs.size(0),
    )
    if aligned != expected:
        raise AssertionError(
            "{} alignment mismatch: {} != {}".format(name, aligned, expected)
        )
    flattened = [frame for frames in aligned for frame in frames]
    if flattened != sorted(flattened):
        raise AssertionError("{} alignment is not monotonic".format(name))
    print("{}={}".format(name, aligned))


def main():
    assert_alignment(
        "single_gloss",
        make_log_probs(
            6,
            12,
            [(0, 0), (1, 3), (2, 3), (3, 3), (4, 0), (5, 0)],
        ),
        [3],
        [[1, 2, 3]],
    )
    assert_alignment(
        "two_distinct_glosses",
        make_log_probs(
            8,
            12,
            [
                (0, 0),
                (1, 4),
                (2, 4),
                (3, 0),
                (4, 7),
                (5, 7),
                (6, 0),
                (7, 0),
            ],
        ),
        [4, 7],
        [[1, 2], [4, 5]],
    )
    assert_alignment(
        "direct_distinct_glosses",
        make_log_probs(
            2,
            12,
            [(0, 4), (1, 7)],
        ),
        [4, 7],
        [[0], [1]],
    )
    assert_alignment(
        "repeated_glosses",
        make_log_probs(
            7,
            12,
            [(0, 0), (1, 3), (2, 0), (3, 3), (4, 0), (5, 0), (6, 0)],
        ),
        [3, 3],
        [[1], [3]],
    )
    assert_alignment(
        "multiple_blanks",
        make_log_probs(
            9,
            14,
            [
                (0, 0),
                (1, 0),
                (2, 5),
                (3, 0),
                (4, 0),
                (5, 11),
                (6, 11),
                (7, 0),
                (8, 0),
            ],
        ),
        [5, 11],
        [[2], [5, 6]],
    )

    try:
        _ctc_forced_align(
            make_log_probs(1, 8, [(0, 0)]),
            torch.tensor([1, 2]),
            2,
            1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Too-short input should fail CTC alignment")

    try:
        _ctc_forced_align(
            make_log_probs(2, 12, [(0, 3), (1, 3)]),
            torch.tensor([3, 3]),
            2,
            2,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Repeated glosses must not skip an omitted blank")

    print("ctc_alignment_tests=passed")


if __name__ == "__main__":
    main()
