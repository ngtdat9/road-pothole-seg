# Project 1 — Real-Time Road and Pothole Segmentation — Project Plan

**Type:** Personal portfolio project (computer vision, edge AI)
**Timeline:** 5 weeks, solo, part-time (~10–12 hours/week), 2026-11-17 to 2026-12-21
**Release:** `road-pothole-seg v1.0` at the end of Week 5 (public GitHub repo + ONNX weights)
**Datasets:** Public pothole/road instance-segmentation sets from Roboflow Universe (final selection in Week 1)
**Related:** Project 2 (`pothole-drone`, see `Project2_Drone_Plan.md`) installs this repo as a dependency. This project is complete on its own and its README does not depend on Project 2.

## 1. Problem & Motivation

**Current situation:** Road maintenance teams find potholes mostly through citizen
complaints and periodic manual inspection — someone drives or walks the road and
logs damage by eye.

**Problem:** Manual inspection is slow, inconsistent between inspectors, and covers
only a fraction of the road network at a time. Potholes get reported after they
have already damaged vehicles or caused accidents.

**Impact:** Late detection means higher repair costs (small potholes grow quickly
under traffic and water), more vehicle damage claims, and real safety risk for
cyclists and motorcyclists.

**Opportunity:** A lightweight segmentation model that finds, tracks and measures
potholes in real time on cheap hardware could run on any vehicle-mounted camera,
turning ordinary driving into continuous road inspection.

**One-sentence goal (must pass the "can someone else understand it" test):**
Build a real-time system that finds, tracks and sizes potholes in road video on
low-cost hardware without a CUDA GPU, packaged so other projects can reuse it.

## 2. Scope

| In scope | Out of scope |
|---|---|
| Instance segmentation of `pothole` and `road` with YOLO26-seg (nano, small for comparison) | Drones, robotics, simulation (Project 2) |
| Merged multi-source dataset with de-duplication and leakage checks | Collecting own road photos or video |
| ONNX export and CPU inference with ONNX Runtime | Training a model from scratch |
| Post-processing: mask cleanup, area, severity bucket | Pothole depth estimation |
| Tracking with one stable ID per pothole | Mobile app or web service |
| Video/webcam demo with overlays and a CSV pothole log | Real-world GPS mapping |
| Per-stage latency benchmark on two machines | — |
| An installable Python package with a stable, documented API | — |

## 3. Dataset Quality Checklist

Scored against the 10-point "Goodness of the AI Dataset" framework. Gaps are named
rather than hidden, each with a planned response.

| Criterion | Assessment |
|---|---|
| Relevant | Pothole masks on real road images match the goal directly. The main candidate set ("Potholes and Roads Instance Segmentation") also has a `road` class |
| Sufficient | Expect 2,000–4,000 original images after merging 2–3 sets — enough to fine-tune a pretrained model. Exact count confirmed in Week 1 |
| Representative — caveat | Mostly dashcam and handheld views from a few countries; few or no Korean roads. Stated as a limitation |
| Diverse | Mixed lighting, wet and dry roads, several countries — checked by sampling in Week 1, not assumed |
| **Balanced — gap** | Pixel imbalance: road covers most of each image, potholes a small fraction, many small or distant potholes. Monitored with per-class metrics and a failure gallery grouped by pothole size |
| **Accurate — caveat** | Community-labelled with no guaranteed review. Each set sampled (50 images) and rejected if masks are loose or wrong |
| Consistent | Different class names, sizes and splits across sets — unified by the merge script |
| **Well-Annotated — gap** | Polygon quality varies from tight edges to rough shapes. Noted in the failure analysis |
| Ethically Collected — caveat | Public sets under CC BY 4.0 or similar, licenses recorded per set. Images can contain faces and plates, so frames shown publicly are checked and blurred. No human subjects, no IRB |
| Reproducible | Roboflow versions pinned, merge script and split seed committed, sources and citations in `docs/datasets.md` |

**On "Early Fusion" (combining datasets):** many pothole sets on Roboflow Universe
are forks of the same source images. Merging them without de-duplication puts
near-identical photos in both train and test and inflates scores. The merge
script removes near-duplicates by perceptual hashing **before** splitting, and
splits by source video or scene where possible. The report shows the number of
duplicates found and the metric difference between a naive merge and the
de-duplicated one — a result most pothole projects never measure.

## 4. Goal (technical)

Fine-tune an Ultralytics YOLO26-seg model to segment `pothole` and `road`, export
it to ONNX, and wrap it in an installable package (`road_pothole_seg`) whose
inference runs on ONNX Runtime alone, with no Ultralytics or PyTorch dependency at
run time. The package handles mask post-processing, ByteTrack tracking, and area
and severity estimation, and ships with a video/webcam demo and a per-stage
latency benchmark on CPU-only hardware.

**What makes it more than a tutorial** — these are not optional when time is short:
1. Measured de-duplication and leakage analysis across merged datasets.
2. Per-stage latency benchmarks on two machines and three input sizes.
3. Tracking plus area and severity, not just boxes on a video.
4. A written failure analysis grouped by cause.
5. A clean, versioned, documented API another project actually installs.

**Designed for reuse by Project 2** (decided now, so v1.0 never needs reopening):
- Always outputs the `road` mask as well as pothole masks.
- Input size, confidence and mask thresholds are configurable, not hardcoded.
- Accepts any frame size and aspect ratio (letterboxed internally, masks returned
  at the original frame size).
- No ROS, camera or file I/O inside the package — only arrays in, data classes out.
- Area in square metres is optional: filled when the caller passes metres-per-pixel.
- Each pothole carries its score, box and area as a fraction of the frame, so
  tracking and severity work the same at any resolution (see Section 6).

## 5. Resource checklist

| Resource | Status |
|---|---|
| Time | 5 weeks part-time — confirmed |
| Dataset access | Roboflow Universe, free account, public download |
| Dataset size | ~1–3 GB after merging |
| Training compute | Google Colab GPU compute units — confirmed |
| Development machine | Mac — code, tests, demo, benchmark |
| Second benchmark machine | Windows PC (RX 580, 12 GB RAM) — CPU inference only |
| YOLO / segmentation skills | Confirmed from earlier projects |
| Claude subscription | Confirmed |

**Compute note:** training happens only on Colab — push code via git, pull it in
the notebook, train, save checkpoints to Google Drive. Run a 5-epoch sanity check
before every full run so a config typo doesn't waste compute units. Inference is
CPU-only by design: the RX 580 has no CUDA, and "real-time without a CUDA GPU" is
the edge-AI result this project reports.

## 6. Repository structure

```
road-pothole-seg/
├── CLAUDE.md
├── README.md
├── pyproject.toml             # installable: pip install git+...@v1.0
├── .gitignore
├── configs/
│   ├── data.yaml              # merged dataset paths and class names
│   ├── train.yaml             # training hyperparameters
│   └── pipeline.yaml          # input size, thresholds, severity bins
├── road_pothole_seg/          # the package — pure Python, ONNX Runtime only
│   ├── __init__.py            # public API: Segmenter, analyse_potholes, PotholeTracker
│   ├── types.py               # Segmentation, Pothole data classes
│   ├── segmenter.py           # ONNX wrapper: letterbox, inference, mask decoding
│   ├── postprocess.py         # mask cleanup, area, severity
│   └── tracker.py             # ByteTrack: one ID per pothole
├── apps/
│   └── run_video.py           # video file or webcam → overlay + CSV log
├── training/
│   ├── notebooks/             # Colab training notebooks
│   └── scripts/
│       ├── merge_datasets.py  # class remap, de-duplication, fixed split
│       └── export_onnx.py
├── benchmarks/
│   └── fps.py                 # latency per stage, per machine and input size
├── tests/                     # pytest, one file per package module
├── models/                    # gitignored; weights published as GitHub release assets
└── docs/
    ├── datasets.md            # sources, licenses, citations, duplicate counts
    ├── results.md             # one row per experiment
    ├── log.md                 # weekly notes
    └── figures/
```

### Public API (frozen at v1.0)

```python
@dataclass
class Segmentation:
    road_mask: np.ndarray | None      # HxW bool, at original frame size
    pothole_masks: list[np.ndarray]   # one HxW bool mask per pothole
    scores: list[float]

@dataclass
class Pothole:
    track_id: int                          # -1 until PotholeTracker assigns one
    mask: np.ndarray                       # HxW bool, at original frame size
    score: float                           # detection confidence (ByteTrack needs it)
    bbox_xyxy: tuple[int, int, int, int]   # pixel box (ByteTrack matches on it)
    area_px: int
    area_frac: float                       # area_px / frame area, resolution-independent
    area_m2: float | None                  # filled only when metres_per_px is given
    severity: str                          # "small", "medium", "large"
    center_px: tuple[int, int]

@dataclass(frozen=True)
class SeverityBins:
    frac: tuple[float, float]              # upper bounds of small / medium, as area_frac
    m2: tuple[float, float]                # upper bounds of small / medium, in m²

class Segmenter:
    def __init__(self, model_path: str, input_size: int | None = None,
                 conf: float = 0.25, mask_threshold: float = 0.5): ...
    def __call__(self, frame: np.ndarray) -> Segmentation: ...   # any HxWx3 BGR frame

def analyse_potholes(seg: Segmentation,
                     metres_per_px: float | None = None,
                     severity_bins: SeverityBins | None = None) -> list[Pothole]: ...

class PotholeTracker:
    def __init__(self, frame_rate: float = 30.0, **matching_thresholds): ...
    def update(self, potholes: list[Pothole]) -> list[Pothole]: ...  # assigns track_id
```

**API decisions (agreed 2026-11-17, before the freeze):**
- **Config without file I/O.** `configs/pipeline.yaml` is the single source of truth
  for thresholds, input size and severity bins. The package never reads it: functions
  take plain arguments whose defaults equal the YAML values. Apps (`run_video.py`,
  `fps.py`, Project 2's ROS node) load the YAML and pass values in. A test asserts
  the YAML and the package defaults match.
- **Severity bins.** Without `metres_per_px`, severity uses `area_frac` bins (same
  meaning at any resolution). With `metres_per_px`, it uses the m² bins (starting
  point: small < 0.1, medium 0.1–0.5, large > 0.5 m²; tuned later). A single
  `metres_per_px` is only valid for a top-down camera — on dashcam frames
  `area_m2` stays `None` and this is stated in the docstring and README.
- **`input_size`.** `None` reads the size from the ONNX input shape. An explicit size
  that conflicts with a fixed-shape model raises `ValueError`.
- **Masks** stay full-frame HxW bool; memory is small enough at both frame sizes.
- **`score` and `bbox_xyxy`** are part of `Pothole` so the tracker never recomputes them.
- **`PotholeTracker`** takes `frame_rate` and matching thresholds as optional
  arguments (dashcam ≈ 30 FPS, drone 5–10 Hz).

Breaking changes after v1.0 mean a new major version. Additions (new optional
arguments, new functions) are minor versions.

## 7. Week-by-week plan

### Week 1 (Nov 17–23) — Setup and dataset selection
- Create the repo from Section 6 with `pyproject.toml`, write `types.py`, and add
  stub functions returning dummy values so `run_video.py` runs end to end on day one.
- Set up `pytest`, `ruff` and GitHub Actions for lint and tests.
- Run the Roboflow agent search. Inspect each candidate before writing merge code:
  original image count, classes, annotations per class, license, viewpoint, and 50
  sampled masks by eye.
- Pick 2–3 sets, record sources, licenses and citations in `docs/datasets.md`.
- Deliverable: repo skeleton with CI, dataset shortlist with a written quality check

### Week 2 (Nov 24–30) — Merged dataset and baseline model
- Write `merge_datasets.py`: unify class names to `road` and `pothole`, find and
  remove near-duplicates (log how many), and make a fixed, seeded 70/15/15 split
  grouped by source where possible.
- Confirm the class imbalance from the real numbers.
- 5-epoch sanity run on Colab, then the nano baseline (640 px, 100 epochs).
- Also train once on the naive merge (no de-duplication) for the leakage comparison.
- Deliverable: merged dataset, baseline `best.pt`, first rows in `docs/results.md`

### Week 3 (Dec 1–7) — Tuning and failure analysis
- Train the small model, then 2–3 tuning runs (input size, augmentation, epochs).
- Collect 20 failure cases and group them by cause: shadows, puddles, patched
  asphalt, manholes, distant potholes, small potholes.
- Deliverable: best model chosen, results table, failure gallery

### Week 4 (Dec 8–14) — Package: segmenter, post-processing, tracking
- Export to ONNX at 320, 480 and 640 px.
- Implement `Segmenter` on ONNX Runtime, including letterboxing for any frame size.
  Check its masks against Ultralytics output on 20 images, including non-square frames.
- Implement `postprocess.py` and `tracker.py`, each with unit tests on synthetic masks.
- Deliverable: package modules tested and matching the reference output

### Week 5 (Dec 15–21) — Demo, benchmark and release (v1.0)
- Build `run_video.py`: overlays, FPS counter, CSV log
  (`track_id, first_frame, last_frame, max_area, severity`).
- Demo footage: held-out test-split sequences plus public road videos whose
  license allows reuse. Blur faces and plates in anything shown publicly.
- Benchmark latency per stage on the Mac and the PC CPU at all three input sizes.
- Write the README (Section 7.3), tag `v1.0`, attach ONNX weights to the release.
- Test the install from a clean environment:
  `pip install git+https://github.com/<you>/road-pothole-seg.git@v1.0`.
- Deliverable: **public release v1.0.** Target: 15+ FPS on the Mac, 8+ FPS on the PC CPU

### 7.1 Experiments

| # | Question | Comparison | Metric |
|---|---|---|---|
| 1 | How much do duplicates inflate scores? | Naive merge vs de-duplicated merge | Mask mAP50 on each test split |
| 2 | How much accuracy does nano give up? | Nano vs small | Mask mAP50, mAP50-95 per class, road IoU |
| 3 | Best input size for real time? | 320 vs 480 vs 640 px | mAP and FPS |
| 4 | Does the CPU runtime matter? | ONNX Runtime default vs OpenVINO provider (PC) | Latency per stage |
| 5 | Where does the model fail? | Failure cases by cause and pothole size | Counts, examples |

### 7.2 Risks

| Risk | Response |
|---|---|
| Poor labels in public sets | Sample before merging, drop bad sets |
| Few sets with a `road` class | Keep the main road+pothole set; train road on that set only if needed and document it |
| ONNX mask decoding doesn't match Ultralytics | Compare on 20 images in Week 4 before building on it |
| Exams in Weeks 4–5 | Weeks 4–5 tasks are small and independent; the release can slip one week without affecting Project 2's start much |
| Colab units run out | 5-epoch sanity runs; Kaggle free GPU as backup |

### 7.3 README contents (v1.0)
- Demo GIF at the top, one-paragraph summary
- Install and quick-start (three lines of code)
- API reference for the four public names
- Results: mAP table, leakage comparison, FPS table with hardware listed
- Failure analysis summary and limitations (Section 3 caveats stated plainly)
- Dataset credits and licenses
- No mention of unfinished drone work. When Project 2 ships, add one "Used by" link.

### 7.4 After v1.0
Project 2's Week 10 fine-tunes this model on drone-view frames. That model is
released here as **v1.1** (same API, new weights, a short "Aerial view" section in
the README). This repo gains an improvement; Project 2 only updates its pinned version.

## 8. Using Claude Code for this project

### Install
```bash
# macOS / Linux / WSL
curl -fsSL https://claude.ai/install.sh | bash
```

### Set up
```bash
mkdir road-pothole-seg && cd road-pothole-seg
git init
claude
```
Paste the repo tree and the public API from Section 6 into the first prompt.

### CLAUDE.md
```markdown
# Project context
Real-time road + pothole instance segmentation (YOLO26-seg, ONNX Runtime on CPU),
released as an installable package. Project 2 (a drone repo) installs this
package, so the public API is a contract. See Project1_RoadPothole_Plan.md.

# Architecture rules
- road_pothole_seg/ imports only numpy, opencv, onnxruntime and supervision —
  never ultralytics, torch, ROS or any camera/file I/O
- Ultralytics is used only in training/ for training and export
- Public API = Segmenter, analyse_potholes, PotholeTracker, types.py classes.
  Do not change signatures without a version bump; additions must be optional
- Segmenter accepts any frame size and returns masks at the original size
- Always return the road mask as well as pothole masks

# Conventions
- Thresholds, input sizes and severity bins live in configs/, never hardcoded
- Type hints on all functions; every package module has a test in tests/
- Dataset: de-duplicate before splitting, fixed seed, group by source
- Machines: Mac = development, Colab = training, PC = CPU benchmark only
- Commit after each working milestone
```

### Workflow per week
1. Paste that week's section from this plan and start with one module.
2. Ask for the test with the implementation, not after.
3. Paste full tracebacks rather than describing errors.
4. Run Colab training directly; bring logs and numbers back for analysis.
5. Commit after each milestone.

## 9. References
- Ultralytics docs: https://docs.ultralytics.com
- Potholes and Roads Instance Segmentation dataset: https://universe.roboflow.com/road-dataset-rpv8i/potholes-and-roads-instance-segmentation-nga49
- ONNX Runtime: https://onnxruntime.ai
