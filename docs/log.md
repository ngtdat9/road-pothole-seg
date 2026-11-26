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
