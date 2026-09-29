# Is the backdoor the robust path in every perturbation modality? No. (H46)

## Question

Every detector in the PSBD family rests on 1 asymmetry: under perturbation the backdoor
pathway survives and the clean pathway degrades. H28 shows PSBD, STRIP, SCALE-UP and
IBD-PSC are 1 family in which the operator only selects a Jacobian, and this project has
measured that asymmetry under activation dropout, gradient-guided head-weight ablation,
weight interpolation toward the pretrained initialization, and prediction depth.

4 probes, all pointing the same way, invites the general claim: **the backdoor is the
robust path in every modality**. That claim is false, and this measures where it breaks.

## The common statistic

Modalities are not comparable at a shared nominal strength, so everything is read at
**matched clean damage**. For each modality sweep the disturbance and record

    ASR retention = ASR(d) / ASR(0)
    CA retention  = clean accuracy(d) / clean accuracy(0)

then report ASR retention at the disturbance where CA retention first falls to 0.75. The
backdoor is the more robust path exactly when ASR retention exceeds CA retention there.

## Result

No record of this measurement is on disk and `measure.py`, which the reproduce line below
names, is not in this directory or in the history of it, so none of the values in this
README can be checked or recomputed. The population was GTSRB at 5% and 10% with ASR at
least 0.5, below the canonical 0.85 bar. It included `badnet_a2a` and `adaptive_blend`
cells that are not on the current panel. Whether it held `vit_gtsrb_tact_0_1`, which the
ledger now excludes as source-mapped, cannot be read. The experiment has not been re-run on
the current panel.

3 further problems stand against the table below, from `docs/why-psbd-works-theory.md`
("Existing claims the derivations contradict", item 4). The protocol reads ASR retention
where CA retention first falls to 0.75, but the table's CA retention is 0.527 for dropout and
0.718 for Fourier amplitude, so the 2 modalities are not compared at matched clean damage.
`measure.py` is missing, as stated above. And an ASR retention above 1, the Adaptive-Blend
"anti-fragility" below, is not corrected for a damaged model collapsing onto the target
class, which alone produces a raw retention of 1.04 on WaNet, so it needs the excess form
before it can be read as the attack getting stronger.

GTSRB, 5% and 10%, attacks that actually implanted (ASR >= 0.5).

| modality | what is perturbed | ASR retention | CA retention | backdoor more robust |
|---|---|---|---|---|
| **activation dropout** | token activations at `before_attention_norm` | **0.910** | 0.527 | **12 / 14** |
| **Fourier amplitude** | the input's texture, phase preserved | 0.627 | 0.718 | **2 / 5** |

**The asymmetry is not universal. It is specific to activation-space perturbation, and it
reverses in the input-frequency modality.**

The 2 cells that break the dropout column are both `badnet_a2a`, at ASR retention 0.190 and
0.151 against clean retention 0.470 and 0.709. That is H5's inversion appearing as an
internal control: the one attack family whose trigger-to-target map is content-dependent is
the one whose backdoor is the *fragile* path.

## The sharpest sub-result: adaptive_blend is anti-fragile

| cell | modality | ASR retention |
|---|---|---|
| `vit_gtsrb_adaptive_blend_0_05` | dropout | **1.282** |
| `vit_gtsrb_adaptive_blend_0_1` | dropout | **1.192** |
| `vit_gtsrb_adaptive_blend_0_05` | Fourier amplitude | **1.125** |

Raw retention above 1 reads as the attack getting **more** effective as the model is damaged, in 2
mechanistically unrelated modalities. Adaptive-Blend exists to suppress latent separability
and it is the hardest attack in this panel for every detector measured here. Before the collapse correction above, the reading offered was this.
Damaging the model damages the carrier's own class evidence faster than the trigger's, so
the trigger wins more often, and a detector that reads "how much did the prediction move"
is reading a quantity that moves the wrong way.

## Consequences

1. A defense built on perturbation consistency is exploiting an asymmetry that exists in
   1 modality. That is not a flaw in itself, but it bounds what the family can do and it
   predicts the failure on Adaptive-Blend rather than discovering it.
2. Input-space purification is a different mechanism from activation perturbation, not a weaker version of it. The
   sign of the asymmetry differs, which is a mechanism for the failure recorded in
   [H45](../style_content_split/README.md).
3. The generalization "the backdoor is always the robust path" was proposed by the
   orchestrator and is refuted here by its own measurement. It survives only for
   activation-space probes.

## What is not established

- GTSRB only, 2 modalities, 1 architecture, single seed. The weight-space modalities
  (interpolation toward pretrained, gradient-guided ablation) were measured on single cells
  and are not in the table because they were not swept to matched clean damage.
- The Fourier column is 5 cells, of which 2 support and 3 do not. It establishes that the
  asymmetry is not universal. It does not establish its sign in that modality.

## Reproduce

    PYTHONPATH=. python experiments/retention_curves/measure.py
