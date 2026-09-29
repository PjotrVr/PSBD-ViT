# Where clean samples move under dropout, in latent space (H7)

## Question

H7 rests on 1 measurement: a histogram of which class label a shifted clean
prediction lands on, counted from the cached per-pass argmax. That is entirely
prediction space. It says the label changed and where it went, and nothing about
whether the representation moved the way PSBD's mechanism requires.

PSBD's stated mechanism is a claim about representations. Under dropout the model
loses the clean class evidence and falls back on the strongest learned association,
the trigger-to-target path, so a clean sample should drift toward wherever the
target class lives. That is directly testable and had not been tested here.

## The 4 measurements

All at the final block's CLS features, all on the same samples.

| quantity | what it says |
| --- | --- |
| `toward_target` | cosine between the dropout-induced displacement and the direction from the sample's own class centroid to the target class centroid. Positive means clean samples drift toward the target |
| `toward_landed` | the same but toward the centroid of whichever class the sample actually shifted to. This is the control: if the displacement is simply toward wherever the prediction went, `toward_target` carries no extra information |
| `along_backdoor` | projection of the displacement onto the backdoor direction. The sharpest form of the claim: does dropout push a clean sample along the very direction the trigger uses? |
| `displacement_norm` | scale, so the cosines can be read as more than angles |

A benign model probed with the same trigger is measured alongside, because a
pretrained backbone has its own fallback behavior under heavy perturbation and that
has to be subtracted from any claim about poisoning.

## Running it

    PYTHONPATH=. python experiments/shift_in_latent_space/measure.py \
        --checkpoint-folder vit_cifar10_blend_0_1 vit_cifar10_badnet_a2o_0_1

Writes `results/<folder>/shift_latent.json`, and with `--umap` a projection coloured
by where each sample landed. Needs a GPU.

## Records and reading

5 records exist, `results/vit_cifar10_{badnet_a2o,blend,bpp,lf}_0_1/shift_latent.json` and
`results/vit_cifar10_benign/shift_latent.json`, all CIFAR-10 at 10% poisoning. Every one was
measured with dropout at `pre_residual` at a fixed rate of 0.5, with the block range left at
the script's default of blocks 5 to 8, which is neither a canonical placement nor a rate a
canonical rule chose. The benign model is probed with the BadNets trigger, target 0. 800
clean samples each, final block CLS features.

| model | shifted | landed on target | cos toward target | cos toward landed class | projection along backdoor direction |
|---|---:|---:|---:|---:|---:|
| badnet_a2o | 91 | 0.374 | 0.472 | 0.399 | 0.622 |
| bpp | 90 | 0.300 | 0.463 | 0.378 | 1.859 |
| lf | 103 | 0.214 | 0.509 | 0.461 | 1.354 |
| blend | 69 | 0.145 | 0.430 | 0.344 | 1.461 |
| benign, BadNets probe | 95 | 0.295 | 0.478 | 0.403 | 0.422 |

## Finding

Target-class drift is not supported. The benign control, a model with no backdoor, lands
0.295 of its shifted clean samples on the target class with a cosine toward the target of
0.478. That matches the backdoored models (0.214 to 0.374 landed, 0.430 to 0.509 cosine), so
the drift toward class 0 is a property of the backbone under heavy perturbation and not of
the poisoning. Only the BadNets projection along its own backdoor direction can be set
against the control, and it exceeds it modestly (0.622 against 0.422). The other attacks have
no benign reading along their own directions.

The version of this README before 2026-09-29 quoted a blend landed share of 0.052, a 2 to 5
times chance drift and a trend with poison rate over both BadNets variants. None of those
values is in the saved records, which hold 1 poison rate and no all-to-all model.

The pass that decides where a sample landed and the pass that gives its displacement are 2
separate forward passes, each started from the same seed (`measure.py`), so they share
dropout masks only if both consume the random stream identically, and nothing checks it. A test of H7 that would count needs the
canonical placement at its adaptive rate, a benign control per trigger and more than 1
poison rate. None has been run, so H7 stays open on this evidence.

Hypothesis doc: `docs/hypothesis/H7-clean-shifts-to-target.md`.
