# CEM-SLR Reproduction

This implementation is based on the attached paper
"Counterfactual-Enhanced Multimodal Continuous Sign Language Recognition"
and the VAC_CSLR training framework.

The public repository `vddong-zjut/CFE-SLR` currently contains only a README
stating that the code will be released. This reproduction therefore follows
the equations and architectural descriptions in the paper.

## Implemented Components

- ResNet34 visual encoder with LMA at the layer3 and layer4 stages.
- Counterfactual attention sampled from a mixture of Gaussian and uniform
  distributions.
- CFA supervision from the difference between factual and counterfactual
  framewise logits.
- K5-P2-K5-P2 temporal convolution followed by the paper's interactive
  contextual learning module.
- Two-layer BiLSTM with 1024 hidden units.
- Gloss-level visual features extracted by CTC forced alignment.
- Frozen German BERT (`bert-base-german-cased`) replacing the unavailable
  CoPE text encoder, as requested.
- Causal multimodal fusion with the paper's
  `L_CF`, `L_cau_KL`, and `L_cau_sep` losses.
- Loss weights: `CFA=1.0`, `CauKL=0.25`, `CauSep=0.9`, plus the VAC
  `ConvCTC=1.0` and `SeqCTC=1.0`.
- Final inference uses only the visual branch.

## Reproduction Notes

- The paper does not release its source code. Details not specified in the
  paper use conservative, documented choices.
- The paper uses beam-search alignment with beam width 5 for gloss-level
  visual features. Training here uses CTC forced alignment, which finds the
  maximum-probability monotonic path and avoids unstable decode operations in
  the optimization loop.
- The CoPE text encoder is replaced by a frozen pretrained German BERT, which
  preserves the role of sign-language-domain semantic features without
  pretraining another language model.
- Mixed precision was tested but not used because it was slower for the
  measured PHOENIX14 batch on this A40 system.

## Training

```bash
conda activate VAC
cd /var/lib/docker/wd/Sign/codes/VAC_CSLR
python main.py \
  --work-dir ./work_dir/cem_phoenix14_20261115 \
  --config ./configs/cem_phoenix14.yaml \
  --device 0
```

## Evaluation

After training, load the best checkpoint and run both dev and test:

```bash
python main.py \
  --phase test \
  --config ./configs/cem_phoenix14.yaml \
  --load-weights ./work_dir/cem_phoenix14_20261115/best_dev_XX.XX_epochXX_model.pt \
  --device 0
```

## Current Training Files

- Main log:
  `/var/lib/docker/wd/Sign/codes/VAC_CSLR/logs/cem_phoenix14_20261115.log`
- Parsed progress log:
  `/var/lib/docker/wd/Sign/codes/VAC_CSLR/work_dir/cem_phoenix14_20261115/log.txt`
- PID:
  `/var/lib/docker/wd/Sign/codes/VAC_CSLR/logs/cem_phoenix14_20261115.pid`
