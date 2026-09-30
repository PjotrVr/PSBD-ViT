# Spectral Signatures, training-set poison filtering

Spectral Signatures looks for poisoned examples inside a labeled training set. It is one of the baselines of the PSBD paper's training-set tables and the only one of them with a reference under `third_party/`, so it is ported in `detectors/spectral_signatures.py` for the training-set detection experiment (`experiments/training_set_detection/`). It is not in the detector registry, because every registered detector scores a test input with a `(model, loader)` call and this one scores rows of a training set from their learned representations and training labels.

## The idea in plain words

A backdoor makes the model map every poisoned example of the target label through the trigger rather than through the object. In the representation the classifier head reads, the poisoned examples of that label therefore sit apart from the clean ones along a single direction. Centering the representations of 1 label and taking the top right singular vector finds that direction when the poisoned examples are numerous enough to dominate the variance. Each example's squared projection on it is its outlier score, and the highest scores are removed.

## The original method

Tran et al., "Spectral Signatures in Backdoor Attacks", NeurIPS 2018, arXiv:1811.00636, Algorithm 1. For every label $y$ with examples $x_1, \ldots, x_n$

$$
\begin{aligned}
\hat{R} &= \left[ R(x_i) - \bar{R} \right]_{i=1}^{n} \\
\tau_i &= \left( \left( R(x_i) - \bar{R} \right) \cdot v \right)^2
\end{aligned}
$$

with $v$ the top right singular vector of $\hat{R}$, and the examples with the top $1.5 \epsilon n$ scores are removed.

| symbol | meaning |
|---|---|
| $R(x_i)$ | the learned representation of example $i$, the vector the classifier head reads |
| $\bar{R}$ | the mean representation over the examples labeled $y$ |
| $\hat{R}$ | the centered representation matrix of label $y$ |
| $v$ | the top right singular vector of $\hat{R}$ |
| $\tau_i$ | the outlier score of example $i$ |
| $\epsilon$ | an upper bound on the poisoned fraction, known to the defender |
| $n$ | the number of examples labeled $y$ |

## The reference and the port

The reference is `cleansers_tool_box/spectral_signature.py` of backdoor-toolbox at the commit `third_party.lock` pins. The PSBD paper's own repository (github.com/WL-619/PSBD, `detection/spectral_signature.py`) is a copy of it, so it is the baseline their tables report. It removes $\min(\lfloor 1.5 \epsilon N \rfloor, \lfloor n / 2 \rfloor)$ examples from every label, with $N$ the size of the whole training set, and gives $\epsilon$ the true poison rate as an oracle. The port follows the reference rather than Algorithm 1 on this count, since the comparison is with the paper's numbers.

The representation is the class token after the encoder's final LayerNorm on ViT-B/16 and the pooled input to the final linear layer on ResNet-18, the vector each head reads, as the reference's `model(x, True)` returns.

## Deviations

1. `torch.linalg.svd` with `full_matrices=False` replaces the deprecated `torch.svd(some=False)`. The top right singular vector agrees up to sign and the score squares the projection.
2. The reference loops over examples for the dot product. The port computes 1 matrix product.

`tests/test_detectors_spectral_signatures.py` runs the reference cleanser on a synthetic training set with a planted poisoned cluster and checks that the port removes the identical index set.

## Use in the training-set experiment

The experiment scores every poisoned image and every clean image of the target label, so the target label's scores and removals are exact. Every other label holds no poisoned example, so its removal count is fixed by the formula above and needs no feature at all, and the false-positive rate over the whole training set follows from the label counts.
