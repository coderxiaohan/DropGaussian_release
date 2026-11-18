Research document status — 2026-11-18

This directory saves existing drafts and figure revisions. It does not imply
that the historical manuscript has been updated to the current implementation.

Current implementation
See ../README.md and ../scripts/mvs_complement/ for geometry-complementary
initialization (preserve reconstructed geometry and fill gaps with MVS), followed
by EGD/TBP density control. Internal CLI names remain vcd/vcp.
The main Blender entry now uses original DropGaussian random initialization
(10,000 points), without MVS, while retaining EGD/TBP. This initialization
exception needs disclosure in experimental settings; it is not an adaptive gate.
The density controller is adapted from FastGS ideas; renaming does not remove
that provenance. DropGaussian remains the underlying training baseline.

Current pipeline illustration
figures/geometry_complement_egd_tbp_pipeline_v10.png
Its corresponding .prompt.txt records the built-in imagegen edit prompt.
The last change removes an orphaned diagnostic input arrow and places Periodic
on the Current Gaussians -> Diagnostics connection.
This saved figure predates the Blender random-initialization exception and has
not been regenerated to depict it.

Historical material — not the current method specification
- main.tex, main_zh.tex, main.pdf, main_zh.pdf and main (1).pdf are existing
  CAPA/adaptive-MVS drafts or historical document artifacts. The source drafts
  discuss counterfactual reliability, anchored admission and residual Gaussian
  attributes, which are not implemented by the current complement method.
- TBD entries remain unfilled measurements, not experimental results.
- ../docs/research/mvsdrop_report.md is also an earlier adaptive-method report.
- figures/capa_pipeline_v1.png depicts the old method.
- Pipeline v3/v4/v8/v9 are retained intermediate raster revisions.
- v1/v2/v5/v6/v7 have prompt records only, without corresponding saved PNGs here.
- Local .codex/generated_images paths inside historical prompts are editing
  provenance only, not downloadable assets or a promise of exact reproducibility.

No scientific manuscript body was rewritten as part of this save/push operation.
Existing measured results and their limitations are recorded separately in
../docs/results/2026-11-18/README.txt.
