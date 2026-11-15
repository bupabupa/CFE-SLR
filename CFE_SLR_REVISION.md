# CFE-SLR revision notes

This branch is the reviewer-driven revision. The pre-revision implementation is
preserved on `main`.

## What changed

The old implementation used random attention masks and random logits as
"counterfactual" variables. Those quantities do not identify causal effects.
The revision therefore removes that claim and implements an intervention proxy
with explicit assumptions:

1. A shared visual trunk predicts a semantic representation `Z_s` and an
   environment/signer representation `Z_e`.
2. CTC and ICLM operate only on the visual semantic path, so inference never
   consumes gloss text.
3. A signer classifier is trained on `Z_e`, while a gradient-reversal signer
   classifier discourages signer leakage from `Z_s`.
4. A scaled linear-kernel HSIC penalty reduces dependence between the two
   representations. Batches with fewer than three samples use a cross-covariance
   fallback because the unbiased HSIC estimator is unstable there.
5. An environment-balanced sampler composes each training batch from different
   signers whenever possible.
6. A group-risk variance penalty uses per-signer CTC risks, so invariance is
   optimized across observed environments rather than inferred from a single
   sample.
7. Brightness, contrast, gamma, and small sensor-noise perturbations construct
   semantic-preserving intervention views. Prediction consistency encourages
   stable visual CTC outputs under these views.

The perturbation is called an **intervention proxy**, not a structural
counterfactual. It can support robustness experiments, but it does not by
itself prove that the learned representation is causally identified.

## Privileged text

The GT-gloss text encoder is optional and isolated in
`PrivilegedGlossTeacher`. It is enabled only when
`loss_weights.PrivilegedTeacher > 0` and a local BERT path is supplied. Its
outputs are detached soft targets. It is training-only privileged information,
not a multimodal input available at inference, and the default revision config
keeps it disabled.

## Loss

The implemented objective is:

```text
L = L_CTC(conv) + L_CTC(seq)
    + lambda_cf * KL(P(Y|X) || P(Y|T_env(X)))
    + lambda_env * CE(D_e(Z_e), signer)
    + lambda_adv * CE(D_a(GRL(Z_s)), signer)
    + lambda_inv * Var_e(R_e)
    + lambda_dis * HSIC(Z_s, Z_e)
    + lambda_text * L_privileged
```

`KL` consistency uses the perturbed view as a stop-gradient target. The clean
network is trained to remain stable, avoiding a second backpropagation graph.
`R_e` is the mean CTC risk of samples from environment `e` in the current batch.
This is a risk-extrapolation-style invariance penalty, not a proof of causal
identification.

## Claims that must not be reused

- Do not call fused hidden features `H` a confounder. They are a mediator by
  construction.
- Do not report random-mask prediction differences as natural direct or total
  indirect causal effects.
- Do not claim that the signer classifier, HSIC, or photometric augmentation
  alone proves causal identification.
- Do not call GT glosses an inference-time multimodal input.
- Report multiple seeds, confidence intervals, parameter count, FLOPs, latency,
  and GPU memory in the paper.

## Commands

Synthetic forward/backward test:

```bash
CUDA_VISIBLE_DEVICES=1 /var/lib/docker/wd/miniconda3/envs/VAC/bin/python \
  test_cfe_model.py
```

Real PHOENIX14 batch test:

```bash
CUDA_VISIBLE_DEVICES=1 /var/lib/docker/wd/miniconda3/envs/VAC/bin/python \
  test_cfe_real_batch.py
```

Training:

```bash
CUDA_VISIBLE_DEVICES=1 /var/lib/docker/wd/miniconda3/envs/VAC/bin/python \
  main.py --config ./configs/cfe_phoenix14.yaml \
  --work-dir ./work_dir/cfe_phoenix14_revision --device 1
```
