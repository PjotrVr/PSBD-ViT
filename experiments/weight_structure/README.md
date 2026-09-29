# Spectral signature of the backdoor in weight space (H33)

## Question

Every other measurement in this project reads activations. If the backdoor is a
single direction the model learned to write, it should also be visible in the
weights, and specifically as a low-rank perturbation: subtract a benign model from
its matched backdoored twin, take the SVD. The top singular value should carry
most of the variance in the late layers. That would give a detector needing no
triggered data and no forward pass at all.

## Running it

    python experiments/weight_structure/weight_spectral_signature.py

CPU only. It loads 2 checkpoints per comparison and takes the SVD of each weight
matrix difference, so it is memory bound rather than compute bound. Output goes to
`results/weight_spectral_signature.json`.

## Finding

Refuted. Across 40 (dataset, attack, rate) combinations the weight difference is not
low-rank in the encoder layers. The backdoor perturbation is spread across many
dimensions. The 1 matrix that does read rank 1 is `class_token`, which is a single
vector and therefore trivially rank 1 on all 40, so it carries no information.

The 40 combinations (`results/weight_spectral_signature.json`, CIFAR-100 and Tiny) include 4
TaCT models, CIFAR-100 at 1%, 5% and 10% and Tiny at 10%, which the ledger now excludes as
source-mapped, and several cells below the 0.85 ASR bar. The verdict is that no combination
is low-rank, so dropping the 4 leaves it unchanged on 36. The experiment has not been re-run
on the current panel.

The activation-space direction is real and the weight-space signature of it is not
recoverable this way. A backdoor can be a clean direction in the residual stream
while the weight updates producing it stay diffuse.

Hypothesis doc: `docs/hypothesis/H33-weight-spectral-signature.md`.
