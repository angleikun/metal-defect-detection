# Rejected Auto-Generated Annotations

This directory contains 100 weak-supervision masks generated automatically
via `bbox → Otsu threshold → morphology` pipeline on Day 5.5.

**Why rejected**: The auto-generated masks did not meet quality standards
for U-Net training (visual inspection showed boundary errors, missing thin
defects). After review, this entire batch was discarded.

**Replacement**: 30 manually annotated polygon masks created using LabelMe
(see `data/annotations_manual/`). U-Net trained on manual masks achieved
Macro IoU = 0.413 vs Macro IoU = 0.366 on auto masks (negative transfer
documented in DEVLOG.md Insight #4).

**Why kept in repo**: As physical evidence of the engineering decision
to reject AI-generated training data in favor of manual annotation.
This is an important reproducibility artifact, not training data.

DO NOT use these masks for training.
