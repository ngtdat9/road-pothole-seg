# Weekly log

Short notes per week. Experiments themselves are in `docs/results.md`; dataset
decisions in `docs/datasets.md`.

## Week 2 (Nov 24–30): merge, pseudo road, baseline

- **Training pipeline:** `train.py` profiles in `configs/train.yaml`; the Colab notebook
  is a thin launcher. Data loading was the bottleneck (2 CPU cores, mosaic decodes ~4
  images per sample), so the defaults are now `cache: disk` and `workers: 2`.
- **Drive:** Colab writes thousands of small files to Drive unreliably. Every Colab job
  now writes locally and uploads one zip, then calls `drive.flush_and_unmount()`.
  Drive for desktop must keep syncing on the Mac.
- **Road teacher:** trained on real-road images only, with road mask mAP50-95 0.785 on
  `val_real_road`.
- **Pseudo road (`pseudo_label.py`):** the teacher draws road around potholes, while
  hand labels cover them, so kept potholes are filled into the pseudo road. An off-road
  check (ring cover < 0.5) found 14 labels; 13 were dropped, mostly vehicles that GNSoft2
  labels as potholes. Review rejected 5 images. Test is never pseudo-labelled.
- **merged-v1:** 2,196 / 471 / 473 images in train / val / test. Its zip is on Drive.
- **Baseline (nano, 100 epochs):** pothole mask mAP50-95 0.408, road 0.789. Road equals
  the teacher, so pseudo road did no harm. The run was still improving at epoch 100.

## Week 3 (Dec 1–7): tuning and failure analysis

Plan: train small (exp 2), naive merge (exp 1), tune-1 (150 epochs), and tune-2 (chosen
from the failures). Evaluate at 640/480/320 and run a failure gallery on val (exp 5).
Choose the model on val only, then score test once.

- 2026-11-26: added `small` and `tune1` profiles, `evaluate.py` (multi-size val and a
  predictions zip) and `failures.py` (IoU matching, recall by size, gallery,
  `docs/failures_<run>.csv`).
- 2026-11-26: live training panel (`monitor.py`) in the Colab notebook: run progress and
  ETA, mAP per class with the best epoch, early-stop countdown, speed, loss charts; every
  epoch also saved to `runs/<run>/epochs.csv`.
- 2026-11-26: baseline evaluated on val at 640/480/320 (`docs/runs/baseline/eval_val.json`;
  640 matches the training summary exactly). Pothole drops 0.408 → 0.375 → 0.317, road
  stays ~0.79–0.80.
- 2026-11-26: failure analysis of baseline on val (conf 0.25, IoU 0.5;
  `docs/runs/baseline/failures_val.json`, `docs/failures_baseline.csv`):
  - pothole precision 0.752, recall 0.701; recall by size: tiny 0.55, small 0.73,
    medium 0.74, large 0.90. Missed tiny/distant potholes are the biggest group.
  - 20 worst images by cause: distant 5, crack 4, label error 3, other 3, puddle 2,
    patch 2, small 1. Label errors: a truck wheel, cones and a motorbike dashboard
    labelled as pothole.
  - Road: mean IoU 0.908 on real-road images; all 8 worst are close-ups where the
    predicted road stops at the pothole edge while the labels cover it. Filling
    predicted potholes into the road mask gives 0.927 (images below 0.8: 24 → 13).
- 2026-11-26: small (exp 2) finished: pothole 0.403 vs nano 0.408, road 0.801 vs 0.789. A
  bigger model does not help pothole on this data, so nano stays and tuning goes on with
  nano. The run was cut by a Colab disconnect at epoch 87 and resumed from `last.pt`.
  Fixed the live panel logging Ultralytics' final validation of `best.pt` as an extra epoch.
- 2026-11-26: built `merged-v1-naive` for exp 1 (`naive_split.py`): merged-v1's test groups
  held out, the rest split at random per image with duplicates kept. 64% of naive test
  images share a dedup group with train. Zip on Drive; `naive` profile added.
- 2026-11-26: tune1 (150 epochs) finished: pothole 0.400 vs baseline 0.408, road 0.794. Longer
  training only stretches the learning-rate schedule; it reaches the same plateau later.
  Baseline, small and tune1 all sit within 0.01 on pothole, so gains below ~0.01 are
  likely run-to-run noise. tune2 changed to baseline + scale 0.7 (100 epochs).
- 2026-11-26: tune2 (baseline + scale 0.7) finished: pothole 0.400, mAP50 0.707 (both below
  baseline), road 0.792. Neither tuning run beat the baseline, so the nano baseline is
  the v1.0 candidate. Remaining: naive run (exp 1), then the final test for baseline.
- 2026-11-26: wrote `docs/failures.md` (exp 5): overall numbers, the 20 worst pothole and 8
  worst road images by cause, and README limitations. About a third of the worst cases are
  label problems (wrong labels, cracks labelled as potholes).
- 2026-11-26: naive run finished. On its own (leaky) val it looks better than the baseline:
  pothole mAP50 0.771 vs 0.722, mAP50-95 0.443 vs 0.408. Waiting for both models on the
  clean test before drawing the conclusion.
- 2026-11-26: test opened once. Baseline on test: pothole mAP50 0.720 / mAP50-95 0.401, road
  0.806 (val was 0.722 / 0.408 / 0.789). Exp 1: the naive model scores 0.780 on its own
  test but 0.714 on the clean test, equal to the baseline. Duplicates inflate the
  reported pothole mAP50 by ~0.07 and add no real accuracy.

**Week 3 summary.** Model chosen: nano baseline (100 epochs, 640 px). Small model, 150
epochs and stronger zoom-out all scored within 0.01 of it on val, so the limit is data,
not settings. Test confirms val (pothole mAP50 0.720, road mAP50-95 0.806). Leakage
measured (+0.07 mAP50 from a naive merge). Failure analysis in `docs/failures.md`.
Carried to Week 4: export ONNX at 320/480/640; fill potholes into the road mask in
post-processing; 480 px costs 0.026 pothole mAP50 and 320 px 0.127, to weigh against FPS.

## Week 4 — package: segmenter, post-processing, tracking

- 2026-11-27: exported the baseline to ONNX at 640/480/320 (`export_onnx.py`, opset 17, on
  the Mac). The model is not end-to-end (`end2end=False`), so the output is raw (38 x
  anchors) and the package does its own class-aware NMS (IoU 0.5, Ultralytics' predict
  default). ONNX metadata says the weights are AGPL-3.0 (Ultralytics), to settle before
  the v1.0 release licence.
- 2026-11-27: real `Segmenter` on ONNX Runtime: letterbox, NMS, mask decoding; new optional
  `iou` and `fill_road` (road = road ∪ potholes; all-False road mask when none is found).
  Matches Ultralytics on the same ONNX file (mask IoU 0.996, identical scores, 80 frames
  in 4 shapes). ONNX val equals best.pt with square input, so the export is lossless.
- 2026-11-27: `clean_mask`: keeps the largest part plus parts >= 0.1 x its area, fills
  enclosed holes. On val (ONNX 640) it changes 37 pothole masks; against the labels 21 get
  closer, 5 further, mean IoU unchanged (0.780): a safe tidy-up, not an accuracy gain.
  Severity `frac` bins set to the tertiles of labelled pothole area in train
  (0.12% and 0.66% of the frame) instead of the Week 1 placeholders (0.2% / 1%).
- 2026-11-27: ByteTrack is no longer in supervision (0.31.7 has no tracker), so
  `PotholeTracker` is our own ByteTrack in numpy (Kalman filter on the box, two-stage
  IoU matching, IDs given after `min_hits` frames). supervision is dropped: the package
  now needs only numpy, opencv and onnxruntime, enforced by `tests/test_imports.py`.
- 2026-11-27: tracker settings checked on synthetic clips from 20 val images
  (`benchmarks/track_sim.py`; labels warped with the frames give true tracks). Dashcam
  (zoom towards the horizon, 30 FPS): coverage 0.742 of visible pothole frames (detector
  alone 0.811), 0 ID switches, 6 false tracks. Drone (pan at 5 Hz): 0.725 (detector 0.825),
  0 switches, 2 false tracks. Looser start (track_thresh 0.3) gains ~0.03 coverage but
  doubles false tracks; buffered IoU (boxes +30% per side) adds 0.05 coverage at 5 Hz and
  changes nothing at 30 FPS, so `match_buffer` = 0.3. Static photos have no blur, so real
  video will score lower; re-run on the Week 5 demo footage.
- 2026-11-27: end-to-end on a 60-frame 1280x720 clip (`apps/run_video.py`, Mac CPU): 19 FPS
  at 640. Per stage (median ms): 640 px preprocess 2.3, inference 41.4, mask decoding 10.8,
  analyse + track 4.3 (59 ms total, 17 FPS); 480 px 39 ms (26 FPS); 320 px 14 ms but no
  potholes found on that clip. Mask decoding at frame size is the next cost to cut if
  the PC is too slow. Note: the CSV `first_frame` is the frame a track got its ID, one
  frame after it was first seen (`min_hits` = 2).

**Week 4 summary.** Package complete and tested (170 tests): ONNX export at 640/480/320
(lossless: ONNX val = best.pt with square input), `Segmenter` on ONNX Runtime matching
Ultralytics on the same ONNX (mask IoU 0.996, identical scores), `clean_mask`, severity
bins from data, numpy ByteTrack. Dependency change: supervision removed. Open for
Week 5: real demo footage, Mac + PC benchmark, README, and the **licence**: Ultralytics
marks the exported weights AGPL-3.0, which conflicts with releasing them under MIT.

## Week 5 — demo, benchmark, release

- 2026-11-27: demo from a road video (`apps/make_demo.py` -> `docs/figures/demo.gif`, 6 s,
  600 px, 6.3 MB). Overlay: potholes coloured by severity, legend, readable labels. CSV
  `first_frame` now uses the tracker's first sighting (`PotholeTracker.first_seen`).
- 2026-11-27: `benchmarks/fps.py` on the Mac (whole 383-frame clip at 1280x720) first gave
  14.4 FPS at 640, under the 15 FPS target: mask decoding 12.4 ms and measuring 5.1 ms
  per frame on top of 49 ms inference. Fixes: the final mask resize is done per mask
  around its box with exact float bilinear weights, and mask extents use row/column
  `any`. Now 4.0 + 0.3 ms, 16.7 FPS at 640. Masks also match Ultralytics more closely
  (check_onnx mean IoU 0.996 -> 1.000), as the old code rounded in uint8. ONNX Runtime's
  default thread count is fastest on the M1 (4 threads +5%, 8 threads +37% slower).
  CoreML provider (Mac only, for reference; not parity-checked): 51.6 FPS at 640.
- 2026-11-28: PC benchmark dropped from the plan; speed is reported for the Mac only
  (target 15+ FPS on the Mac CPU: met, 16.7 FPS at 640). Experiment 4 is now ONNX Runtime
  CPU vs the CoreML provider on the Mac (16.7 vs 51.6 FPS at 640).
- 2026-11-28: version 1.0.0, tag `v1.0`.
- 2026-11-28: clean-install test. Fresh Python 3.11 venv outside the repo,
  `pip install git+https://github.com/ngtdat9/road-pothole-seg.git@v1.0`: installs
  road-pothole-seg 1.0.0 with numpy, opencv-python and onnxruntime only (no torch, no
  PyYAML). Both README quick-start examples run unchanged with `baseline-640.onnx`
  (image: one medium pothole; 10-frame clip: one track, ID 1 from frame 2).

**Week 5 summary.** Demo app (`apps/run_video.py`: overlay video + pothole CSV) and README
GIF; per-stage benchmark (Mac M1 CPU at 1280x720: 16.7 / 25.1 / 50.5 FPS at 640 / 480 /
320, target 15+ met after cutting mask decoding from 12.4 to 4.0 ms); README for v1.0;
released `v1.0`. The PC benchmark was dropped.

## Project 1 summary (v1.0, 2026-11-28)

- **What shipped:** an installable package (`Segmenter`, `analyse_potholes`,
  `PotholeTracker`, types) that segments road and potholes in any BGR frame on CPU with
  ONNX Runtime, measures and grades each pothole, and tracks it with a numpy ByteTrack.
  Three dependencies; API frozen for v1.x.
- **Data:** three CC BY 4.0 Roboflow sets merged to 3,140 images (634 near-duplicates
  removed), seeded 70/15/15 split grouped by source; road pseudo-labelled on the
  pothole-only set and checked by eye.
- **Accuracy (test, opened once):** pothole mask mAP50 0.720, mAP50-95 0.401; road
  mAP50-95 0.806 at 640. Nano, small and tuning runs were all within 0.01: the data is
  the limit.
- **Leakage finding:** a naive merge overstates pothole mAP50 by ~0.07 (0.780 on its own
  split vs 0.714 on the clean test).
- **Fidelity:** ONNX export lossless; package masks match Ultralytics at mean IoU 1.000.
- **Speed:** 16.7 FPS at 640 on a Mac M1 CPU (CoreML: 51.6).
- **Tracking:** 0 ID switches on synthetic dashcam and drone clips.
- **Known limits:** small distant potholes (recall 0.55 under 0.1% of the frame),
  look-alikes (water, cracks, patches), no top-down training data. Deferred: label size
  that scales with the frame, dropping off-road potholes.
- **For Project 2:** install from `@v1.0`; the road mask is always returned, and
  `metres_per_px` gives `area_m2` on top-down frames. Drone fine-tuning comes back here
  as v1.1.
