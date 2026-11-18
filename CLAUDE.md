# CLAUDE.md — road-pothole-seg (Project 1 of 2)

## The big picture (read this first)

This repo is **Project 1** of a two-project portfolio by Ngo Thanh Dat
(AI student, Sejong University). Both projects share one perception model.

| | Project 1 — this repo | Project 2 — future repo |
|---|---|---|
| Name | `road-pothole-seg` | `pothole-drone` |
| What | Real-time road + pothole instance segmentation on road video, CPU-only, released as an installable Python package | Simulated PX4 drone (Gazebo, ROS 2 Humble, Ubuntu 22.04) that follows a road using this package's road mask and maps potholes |
| When | 2026-11-17 → 2026-12-21, release `v1.0` | 2026-12-22 → 2027-02-08 |
| Plan | `docs/plans/Project1_RoadPothole_Plan.md` | `docs/plans/Project2_Drone_Plan.md` |

**How they connect:** Project 2 installs this repo with
`pip install git+https://github.com/<user>/road-pothole-seg.git@v1.0`
and never copies or patches its code. Project 2 later fine-tunes the model on
top-down drone frames and that model is released **here** as `v1.1`.

**So every decision in this repo must work for both:**
- a dashcam/handheld video frame (Project 1), and
- a top-down drone camera frame of any size, called from a ROS 2 node (Project 2).

We are working on **Project 1 only**. Do not create drone, ROS or simulator code
in this repo. If a choice here would make Project 2 harder, say so before doing it.

Read the relevant plan file before starting any week's work. The plans are the
source of truth for scope, timeline and targets.

## Current phase
Week 1 (Nov 17–23): repo setup and dataset selection.
<!-- Update this line at the start of each week. -->

## Architecture rules
- `road_pothole_seg/` is the package. It imports only numpy, opencv-python,
  onnxruntime and supervision — never ultralytics, torch, ROS, or camera/file I/O.
- Ultralytics is used only in `training/` for training and ONNX export.
- Inference in the package uses ONNX Runtime on CPU.
- Public API (frozen at v1.0): `Segmenter`, `analyse_potholes`, `PotholeTracker`,
  and the data classes in `types.py`. Do not change signatures without a version
  bump; additions must be optional arguments or new functions.
- `Segmenter` accepts any HxWx3 BGR frame, letterboxes internally, and returns
  masks at the original frame size.
- Always return the road mask as well as pothole masks.
- `area_m2` is filled only when the caller passes `metres_per_px`. A single
  `metres_per_px` is only valid for top-down frames; on dashcam frames it stays `None`.
- Severity uses `area_frac` bins by default and m² bins when `metres_per_px` is given
  (`SeverityBins` in `types.py`).
- `Segmenter(input_size=None)` reads the size from the ONNX model; a conflicting
  explicit size raises `ValueError`.
- The package never reads `configs/`. Functions take plain arguments whose defaults
  equal `configs/pipeline.yaml`; apps load the YAML and pass values in. A test keeps
  the two in sync.

## Data rules
- Classes are exactly `road` and `pothole`.
- De-duplicate (perceptual hashing) before splitting; log how many were removed.
- Fixed, seeded 70/15/15 split, grouped by source video or scene where possible.
- Record every dataset's source, version, license and citation in `docs/datasets.md`.
- Never use test-split images for anything except final evaluation.
- Pothole-only sets get **pseudo-labelled road** (road teacher, reviewed by eye) in
  training images only; never leave visible road or potholes unlabelled.
- Real top-down/UAV images (Pothole UAV Segmentation, the `DJI_*` frames of road
  segmentation v2) are reserved for Project 2's aerial test set: never used to train
  or tune v1.0.
- No box-only label line may reach training: Ultralytics drops every polygon in a
  dataset that mixes boxes and polygons. Drop the box or the image, never keep both.
- Per-source keep/drop decisions live in `docs/datasets.md`; the merge script follows them.
- Raw downloads go in `data/raw/<slug>-v<version>/` (gitignored); merged sets are
  versioned (`merged-v1`, ...) and uploaded to Drive as one zip.

## Conventions
- Thresholds, input sizes and severity bins live in `configs/`, never hardcoded.
- Type hints on all functions; short docstrings on public functions.
- Every module in `road_pothole_seg/` has a matching test in `tests/`.
- `ruff` for lint and format; `pytest` must pass before a commit.
- Small, reviewable changes: one module or function at a time.
- Write the test together with the implementation.
- Commit after each working milestone, not mid-edit.

## Machines
- **Mac:** all coding, tests, demo, benchmark. Claude Code runs here.
- **Google Colab:** all training. Code reaches Colab only through git; checkpoints
  go to Google Drive. Do not try to train locally.
- **Windows PC (RX 580, 12 GB RAM):** CPU benchmark only. No CUDA.

## Experiment logging
Every experiment gets one row in `docs/results.md`: date, config, metrics,
one-line conclusion. Weekly notes go in `docs/log.md`.