# Pre-revision snapshot

This repository state is the CEM-SLR implementation before the reviewer-driven
revision. It is preserved for traceability and comparison with
`CFE-SLR-revision`.

Known limitations in this snapshot:

- Learnable attention masks are perturbed randomly and interpreted as
  counterfactual interventions.
- Fused features are treated as a confounder even though the implementation
  constructs them as a mediator.
- Ground-truth glosses are encoded by BERT during training and used by the
  fusion classifier.
- The fusion objective contains no explicit nested counterfactual or causal
  identification assumptions.

These limitations are addressed, rather than hidden, in the revision notes.
