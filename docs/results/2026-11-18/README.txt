Experiment snapshot — 2026-11-18

This directory preserves existing experiment results; no training was rerun to
produce this snapshot. All source files were copied verbatim from output/.
The output/ subdirectory here is a historical snapshot, not a new training target.

Contents
- 63 metrics files: metrics_mean.txt, metrics_per_scene.csv, metrics_summary.json.
- 183 original per-scene manifest.json files, including protocol, initialization
  provenance, density settings/events, convergence samples and resource records.
- SHA256SUMS verifies every copied file against the original source at capture.
- No model checkpoints, point clouds, rendered images or datasets are included.

Methods and coverage
1. output/mvs_complement/coverage/keep_base: 9 settings, 91 scene runs.
   LLFF 3/6/9 views (8 scenes each), DTU 3/6/9 (15 each),
   Mip-NeRF360 12/24 (7 each), Blender 8 (8 scenes).
2. output/mvs_vcd_vcp: direct MVS + EGD/TBP (CLI vcd_vcp), 6 settings, 46 runs.
3. output/mvs: direct MVS + original density control, 6 settings, 46 runs.
   Both direct-MVS comparisons have LLFF 3/6/9, Mip-NeRF360 12/24 and Blender 8.
   They do not contain DTU results; this is not a complete four-dataset paired study.

All 183 runs completed 10,000 iterations with seed 0. They are single-run
measurements, not evidence of statistical significance across repetitions.
All 21 CSV row counts and means were checked against their JSON summaries.

Provenance and limitations
- All manifests record git_dirty=true. A commit ID alone does not reconstruct
  the exact historical working tree that produced these measurements.
- Complement runs record base commit 28e7b9a7496ca04a32b58d58bae9e5a18fa9b75a.
- The direct-MVS comparisons record d337ee0981225857a36d0cc30037405bd3aedaa3.
- These measurements predate the cleanup commit that stores this snapshot.
  Do not attribute them to a newly trained clean checkout of that commit.
- The complement manifests retain a coarse historical method string,
  DirectMVS-DG+VCD_VCP. Use initialization_source.fusion_policy to distinguish
  baseline_geometry_mvs_complement (83 real-scene runs) from
  direct_mvs_no_reconstructed_base (8 Blender runs). Labels were not rewritten.
- Blender complement runs retained zero base points and fell back to direct
  MVS: 100,000 points in seven scenes, 15,519 in ficus. Their results are not
  evidence of successful geometry completion on Blender.
- The default keep_base protocol does not fix total point count to the direct
  MVS budget. Equal-budget ablations are needed to isolate selection benefits.
- Base geometry's training-view-only provenance is not independently verified;
  base_camera_provenance_verified remains false in the recorded manifests.
- All retained Blender runs evaluate 25 test cameras, not all 200 test views.
- DTU CSV scene names omit the _colmap suffix found in manifest.scene.
- Absolute local paths in original manifests are provenance records, not
  portable dataset paths. The files intentionally retain those original values.

Retired results
The user requested deletion of output/mvsall and output/mvsinit_vcd_vcp.
Those result directories are not included. Other retired TPS/surface results
were archived outside the repository and are also not part of this snapshot.
