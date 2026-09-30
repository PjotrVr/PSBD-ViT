# Notebooks

Executable documentation of the whole project, written to be read in order by
someone re-learning it. Every notebook runs top to bottom against the real
repository on CPU, calls the same library functions and `scripts/paper/`
generators the paper is built with, and is committed with its outputs and every
figure embedded, so its numbers are readable without running anything. Each
notebook also checks its numbers against the macro sidecars under
`paper/tables/`, so a notebook that still executes still agrees with the paper.
Prose that states a result is rendered from the data by the cell that computes
it (`display(Markdown(...))`), so a rerun after the results move rewrites the
text, and qualitative claims the text makes are asserted in the same cell. The
GPU half of the method, the perturbed forward passes, is read from the stage-1
cache under `results/<folder>/psbd/` rather than recomputed.

The diagrams under `figures/` are TikZ sources compiled by `figures/render.py`
(tectonic, then pypdfium2 to PNG). The rendered PNGs are what the notebooks embed.

## Reading order

1. `start-here.ipynb`. The vocabulary, the ViT block with every perturbation
   position, the PSBD pipeline from k perturbed passes to the verdict, which
   command writes which file, the 98-cell model panel and the headline in 1
   figure: PSBD-TM against PSBD-RD on the 57 headline models.
2. `how-every-number-is-computed.ipynb`. Every formula behind a number in the
   paper, with its code path.
3. `data-and-attacks.ipynb`. Every trigger drawn on 1 image and across the 4
   datasets, the patch and global trigger families, the training and evaluation
   eligibility rules of the 3 label policies, the clean-label poison-rate cap per
   dataset, and which trained models implanted, diverged or map their TaCT source
   class without a trigger.
4. `psbd-end-to-end.ipynb`. PSBD-TM and PSBD-RD analyzed on 1 checkpoint
   (BadNets at 1% on CIFAR-100) from its cache: the splits, the baseline, 1 token
   mask drawn on the image, the shift ratio and AUROC over the rate ladder, the
   rate rules, the score histograms, every quantile, ROC curves and where shifted
   predictions land, each checked against `psbd_metrics.json`.
5. `prediction-shift-phenomenon.ipynb`. The prediction shift itself, measured
   before any detector is built on it.
6. `placement-walk.ipynb`. The 4 staircase tables that separate the position
   from the operator, each as a figure with paired intervals, the gain per
   attack, the ranking of the 27-placement basis and a position by operator map.
7. `placements-and-operators.ipynb`. The full grid of positions and operators:
   every operator of `defenses/operators.py` with its source and a worked example
   on a toy tensor, every position of `models/positions.py` with the tensor shape
   it receives, recorded live on randomly initialized ViT-B/16, Swin-S and
   ResNet-18, and the sweep as the sequence of hypotheses (H1, H3, H9, H10, H16,
   H20 to H23, H26, H27, H47) that produced it, each step with its paired figure.
   `docs/placement-rationale.md` is its prose companion with the generated ledger.
8. `depth-bands.ipynb`. PSBD-TM restricted to blocks 1 to 4, 5 to 8 or 9 to 12
   against all 12 at the matched rule and per attack, with the same bands for
   residual dropout, which behave the other way.
9. `attention-probes.ipynb`. Every attention-side probe, inside and outside the
   basis, against PSBD-TM: unpaired means, paired gaps, the strongest probes per
   attack and PSBD-TM against its twin model by model.
10. `mechanism.ipynb`. Why PSBD-TM separates patch triggers: which predictions
    each placement changes, the depth at which the backdoor direction appears,
    its ablation, activation patching, attention routing, the causal masking
    test and LayerNorm absorption.
11. `backdoor-manifestation.ipynb`. How each attack shows up inside ViT-B/16 and
    Swin-S on the 42 successful panel models, each read beside a benign model
    shown the same trigger: the triggers in pixel space, per-token change maps,
    class-token attention on the trigger's tokens, the direction onset by depth,
    top TAC neurons against the backdoor direction with energy-matched and
    target-class controls, head geometry and the readout alignment. Reads
    `results/_experiments/backdoor_manifestation/` only and runs in about a minute.
12. `why-psbd-works.ipynb`, `why-psbd-works-general.ipynb` and
    `why-psbd-tm.ipynb`. The explanation of the detector in general and of
    PSBD-TM in particular. `why-psbd-works.ipynb` is the index of every claim
    about why PSBD and PSBD-TM work or fail on ViT-B/16 and Swin-S: Li et
    al.'s 4 observations, the breaking-point curves, patch-trigger routing,
    clean fragility, confidence, out of distribution, LayerNorm absorption,
    IBD-PSC's theorem, neurons against directions, WaNet, global-trigger
    redundancy, the cache readouts and the general principles. It opens on a
    claims table with 1 row per claim (verdict true, false, partial, open or
    pending, the key number, the control and the file and field of the
    evidence), gives each claim a section with its prediction, experiment,
    control, figure and what the evidence does not show, and reads only the
    records the experiments wrote, so a rebuild picks up what they add.
    `why-psbd-works-general.ipynb` tests margin,
    backdoor-direction dominance, token redundancy, low dimension and flatness
    per attack category on ViT-B/16 and Swin-S, gives the explanations that do
    not hold (confidence, MC-dropout uncertainty, out of distribution, trigger
    neurons, neuron bias, an easy target class) their own step and ends on a
    grid of verdicts. It reads the JSON of `experiments/why_psbd_works/` only.
    `why-psbd-tm.ipynb` follows a patch trigger through
    ViT-B/16 and Swin-S in 12 steps: where it is read, what PSBD-TM's masks and
    PSBD-RD's dropout do to it, the site and the operator separated on the
    cached sweeps and in a causal grid at matched clean damage, the trigger
    signal each probe leaves per block, and the explanations that fail. It reads
    the records of `experiments/why_token_masking_works/` only.
13. `competitor-defenses.ipynb`. The 11 ported detectors, what each does and how
    each was checked against its reference: the idea, a figure of what each
    perturbs or reads, the source of its scoring function, its score
    distributions on 2 models and its AUROC per attack beside PSBD-TM and
    PSBD-RD, then all 13 defenses ranked with their cost. The pages under
    `docs/detectors/` hold the per-rate tables `scripts/detector_doc_results.py`
    generates.
14. `detectors-per-attack.ipynb`. Every defense's AUROC ranked overall, per
    attack, per dataset and at the 10% budget, with PSBD-TM against the calibrated
    IBD-PSC model by model.
15. `swin-and-robustness.ipynb`. The Swin-S window layout, the placements on
    Swin, Swin against ViT on the same attack cells, an attacker who trains
    against the probe, the union of probes and the cost in forward passes.
16. `swin-per-attack.ipynb`. PSBD-TM against PSBD-RD on Swin-S per attack and
    dataset, and the per-attack gain on Swin against ViT.
17. `reproducing-the-paper.ipynb`. How `paper/` is regenerated from `results/`,
    the coverage ledger and basis coverage, the mistakes that changed a number
    and the checks that catch them, the seed spread and what is still running, with 1
    macro traced back to its files.
18. `all-numbers.ipynb`. Every number the paper prints, with its source.

## Running

```bash
PYTHONPATH=. .venv/bin/python -m jupyter lab                      # interactive
PYTHONPATH=. .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    notebooks/NAME.ipynb --ExecutePreprocessor.timeout=1800       # re-execute 1
.venv/bin/python notebooks/figures/render.py                       # re-render the diagrams
```

The 11 notebooks this README describes in full (1, 3, 4, 6, 8, 9, 10, 14, 15, 16
and 17) each run in 20 seconds to 2 minutes on a login node, most of it parsing
the 11 MB `psbd_metrics.json` of every clearing model. Rendering the diagrams
needs outbound network for tectonic's package cache, so export the proxy first
on Supek (`export https_proxy=http://10.150.1.1:3128`). When the coverage ledger
or `paper/tables/` is regenerated, re-execute the notebooks. The rendered prose
follows the data, and an assert that fails names the macro or the claim that no
longer holds.
