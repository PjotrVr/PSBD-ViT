---
name: psbd-bibliography-audit-2026-09-24
description: Findings of the 2026-09-24 audit of paper/references.bib that recur, chiefly that PSBD is a training-data detector and the paper frames it as a test-time one
metadata:
  type: project
---

On 2026-09-24 the 43 entries in paper/references.bib were audited. None was invented. The 2 content fixes were STRIP's 2nd author (Chang Xu, not the ACM typo "Change Xu") and the spelling Karayal\c{c}in, which the authors print with a dotted i.

Citation-use findings to recheck in future audits:
- PSBD (Li et al., CVPR 2025) was proposed and evaluated as a detector of poisoned training samples, with Spectral Signatures, Spectre and SCAn as baselines. The paper presents it as a test-time input detector and dismisses training-pool detectors.
- Doan et al. (AAAI 2023) PatchDrop is input-level token masking, the closest prior work to PSBD-TM.
- The project's LF, TaCT, Adaptive-Blend and LC implementations differ from the cited papers: LF uses fixed noise instead of an optimized trigger, TaCT has cover samples, Adaptive-Blend uses one opacity, and LC uses a clean surrogate for PGD.
- The table CKA is debiased (Nguyen et al. 2021, Song et al. 2012), but only Kornblith is cited.

**Why:** these are the claims a reviewer who knows the cited work will catch.
**How to apply:** check whether the paper text was fixed before repeating these findings.
