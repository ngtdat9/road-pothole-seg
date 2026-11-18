# Datasets

Every dataset used in this project: source, pinned version, license, citation and
quality check. Target classes are exactly `road` and `pothole`.

## How to inspect a candidate

1. On Roboflow Universe, open the dataset, pick a version, and download it in the
   **YOLOv8** or **YOLO11** format (identical polygon labels). Write the version
   number down. Check that label lines have many numbers (polygons), not 5 (boxes).
2. Unzip it to `data/raw/<slug>-v<version>/` (gitignored), so the folder has
   `data.yaml` and `train/`, `valid/`, `test/`.
3. Run:
   ```bash
   .venv/bin/python -m training.scripts.inspect_dataset data/raw/<slug>-v<version>
   ```
   Paste the printed counts block into the dataset's section below.
4. Open `outputs/inspect/<slug>-v<version>/sheet_*.jpg` (50 images, 5 sheets) and
   fill in the quality check by eye.

Notes:
- **Original images** is the number that counts. Roboflow exports with augmentation
  contain several `name.rf.<hash>.jpg` copies of one photo; the script removes them.
- Each set's own train/valid/test split is ignored later. Week 2's merge script
  de-duplicates across all sets and makes one new grouped, seeded 70/15/15 split.

## Shortlist

| # | Dataset | Version | Original images | Classes (raw names) | `road` class? | License | Viewpoint | Masks OK? | Decision |
|---|---|---|---|---|---|---|---|---|---|
| 1 | [Potholes and Roads Instance Segmentation](https://universe.roboflow.com/road-dataset-rpv8i/potholes-and-roads-instance-segmentation-nga49) | v1 | 1,355 (904 fully masked) | `0`, `road` | yes | CC BY 4.0 | mixed ground | road good; 931 pothole boxes without masks | use, partially |
| 2 | [road segmentation](https://universe.roboflow.com/university-evipv/road-segmentation-onvjk) | v2 | 916 | `pothole`, `road` | yes | CC BY 4.0 | ground + 30 drone | yes, tight | use |
| 3 | [Pothole segmentation — VNR Project](https://universe.roboflow.com/vnr-project/pothole-segmentation-8esmn) | — | (3,271 reported) | `POTHOLE`, `pothole` | no | CC BY 4.0 | — | **boxes only, no masks** (checked on Roboflow 2026-11-18) | reject |
| 4 | [Pothole Segmentation — GNSoft2](https://universe.roboflow.com/gnsoft2/pothole-segmentation-7fnkw) | v2 | 1,990 | `manhole`, `pothole`, `sewer`, `vehicle` | no | CC BY 4.0 | Korean dashcam + handheld | yes, all polygons | use (road pseudo-labelled) |
| 5 | [Pothole — UMAT](https://universe.roboflow.com/umat/pothole-abpqz) | | (555 reported) | `pothole` | no | CC BY 4.0 | | | backup |
| 6 | [Pothole Segmentation — Muhammad Moin](https://universe.roboflow.com/muhammad-moin-qbwes/pothole-segmentation-eqkr6) | | (780 reported) | `Pothole` | no | CC BY 4.0 | | | backup (432 px resize) |

Decision is one of: **use**, **use for pothole only** (road gets pseudo-labels in
Week 2), **reject (reason)**. "Reported" counts are Roboflow's image totals, not
verified originals.

### Reserved for Project 2 — never used in v1.0 training

| Dataset | Reported images | Classes | License | Role |
|---|---|---|---|---|
| [Pothole UAV Segmentation](https://universe.roboflow.com/dimz-fypwn/pothole-uav-segmentation) | 807 | `Pathole` (sic) | CC BY 4.0 | Project 2 real-aerial test set (viewpoint gap) |
| `DJI_*` frames from road segmentation v2 (dataset 2) | 30 originals | `pothole`, `road` | CC BY 4.0 | Same; the only real aerial images with `road` masks |

### Rejected without download

| Dataset | Reason |
|---|---|
| [road-damage-uav-ags](https://universe.roboflow.com/juan-francisco-gamez-cortes/road-damage-uav-ags) | Object detection only (no masks) |
| [Road Damage Classification (Drone)](https://universe.roboflow.com/syadil/road-damage-classification-drone) | Object detection only, 100 images |
| [road-pothole segmentation](https://universe.roboflow.com/road-dataset-rpv8i/road-pothole-segmentation-dqvml) | 36 images |
| [Drone Road](https://universe.roboflow.com/road-damage-detection-xqmig/drone-road-oozbq) | 50 images, unclear classes (`1`, `Drone-Road`) |

---

## 1. Potholes and Roads Instance Segmentation

- **Source:** https://universe.roboflow.com/road-dataset-rpv8i/potholes-and-roads-instance-segmentation-nga49
- **Version / download date:** v1 (2025-01-05), downloaded 2026-11-18, YOLO11 format
- **Local folder:** `data/raw/potholes-and-roads-instance-segmentation-nga49-v1/`
- **License:** CC BY 4.0
- **Preprocessing / augmentation:** none listed
- **Citation:** dataset of the paper "An End-To-End Pothole Detection and Avoidance
  System for Autonomous Ground Vehicles" (README says citation pending — find the
  paper's reference before release)

#### Counts — `potholes-and-roads-instance-segmentation-nga49-v1`

- Classes: `0`, `road` (`0` is pothole — confirmed on contact sheets)
- Images: 1355 (train 1076, valid 138, test 141)
- Original images (Roboflow `.rf.` copies removed): 1355 (train 1076, valid 138, test 141)
- Originals appearing in more than one split (leakage): 0
- Images without labels: 0
- Box-only labels (no polygon): 3430 — **all class `0`**

| Class | Annotations | Images | Median area | tiny <0.1% | small 0.1-1% | medium 1-5% | large >=5% |
|---|---|---|---|---|---|---|---|
| `0` | 6268 | 1309 | 0.31% | 891 | 997 | 591 | 359 |
| `road` | 1387 | 1354 | 35.20% | 7 | 14 | 11 | 1355 |

(Size buckets count polygons only: 2838 pothole polygons.)

**Box-only labels, analysed:** of the 3430 pothole boxes, 2499 overlap a pothole
polygon in the same image (box vs polygon-box IoU > 0.5) — the same pothole labelled
twice, safe to drop. **931 boxes are potholes with no mask**, spread over 451 images
(train 368, valid 41, test 42). Ultralytics drops *all* polygons from a dataset that
mixes boxes and polygons, so no box line may reach training.

### Quality check (50 sampled images)

| Check | Notes |
|---|---|
| Viewpoint | Mixed: dashcam (some with car interior visible), handheld close-ups, street level |
| Countries / road types | Mixed; snowy highway dashcam series, city streets, rural |
| Lighting and weather | Day; dry, wet and snowy roads |
| Pothole masks | Polygons that exist are reasonably tight; many potholes are boxes only (see above) |
| Road masks | Good — whole drivable surface, one polygon per image |
| Missing potholes | Not seen in sample beyond the box-only issue |
| Near-duplicate frames | Yes — `pothole-*` and `vlcsnap-*` are consecutive frames of a few dashcam videos; group by video when splitting |
| Faces / plates | Cars and plates visible in some frames; blur if shown publicly |

**Decision (2026-11-18):** **use 904 originals.** Drop the 2,499 duplicate boxes;
**drop the 451 images that contain a mask-less pothole** (they stay in `data/raw/`).
Revisit with box-prompted SAM only if the merged set falls short of target or Week 3
shows weak dashcam pothole recall.

---

## 2. road segmentation (university-evipv)

- **Source:** https://universe.roboflow.com/university-evipv/road-segmentation-onvjk
- **Version / download date:** v2 (2023-09-13), downloaded 2026-11-18, YOLO11 format
- **Local folder:** `data/raw/road-segmentation-onvjk-v2/`
- **License:** CC BY 4.0
- **Preprocessing / augmentation:** resized to 640×640 **with stretch** (aspect ratio
  distorted); 3 versions per training image with a random 0–20% crop
- **Citation:** none given — cite the Roboflow Universe page
- **Provenance note:** file names `India_*`, `Japan_*`, `Norway_*`, `United_States_*`
  look like RDD2022 (Road Damage Dataset 2022) images, which are CC BY-SA 4.0
  upstream — confirm and credit RDD2022 in the README.

#### Counts — `road-segmentation-onvjk-v2`

- Classes: `pothole`, `road`
- Images: 2268 (train 2010, valid 147, test 111)
- Original images (Roboflow `.rf.` copies removed): **916 unique** (per split: train 670, valid 145, test 111; 10 originals sit in two splits)
- Originals appearing in more than one split (leakage): 10
- Images without labels: 0
- Box-only labels (no polygon): 0

| Class | Annotations | Images | Median area | tiny <0.1% | small 0.1-1% | medium 1-5% | large >=5% |
|---|---|---|---|---|---|---|---|
| `pothole` | 3178 | 1838 | 0.35% | 432 | 2034 | 633 | 79 |
| `road` | 2762 | 2268 | 30.46% | 10 | 33 | 326 | 2393 |

(Counts include the 3× augmented training copies.)

### Quality check (50 sampled images)

| Check | Notes |
|---|---|
| Viewpoint | Mostly street level: motorbike/dashcam screenshots, handheld; **30 `DJI_*` originals are drone frames** (oblique/top-down) |
| Countries / road types | India (largest share), Japan, Norway, US; city, rural, highway |
| Lighting and weather | Day; dry and wet roads |
| Pothole masks | Tight polygons |
| Road masks | Whole drivable surface, clean edges |
| Missing potholes | None obvious in sample |
| Near-duplicate frames | Yes — `VID_*`, `*_mp4_*`, `DJI_*_MOV-*` are video frames; group by video when splitting |
| Faces / plates | Riders and vehicles visible; blur if shown publicly |

**Decision (2026-11-18):** **use 886 originals.** Best two-class set. Keep one copy
per original; group by video. **Exclude the 30 `DJI_*` drone frames** from v1.0 so
it never sees a top-down view; they join Project 2's real-aerial test set.

---

### Cross-dataset overlap (1 vs 2)

dHash (64-bit) over one copy per original: **0** matches at Hamming ≤ 4; 12 loose
matches at ≤ 8 with unrelated file names (likely false positives). The two sets are
effectively independent. Week 2's merge script runs the full de-duplication.

---

## 4. Pothole Segmentation (GNSoft2)

- **Source:** https://universe.roboflow.com/gnsoft2/pothole-segmentation-7fnkw
- **Version / download date:** v2 (2024-07-12), downloaded 2026-11-18, YOLO11 format
- **Local folder:** `data/raw/pothole-segmentation-7fnkw-v2/`
- **License:** CC BY 4.0
- **Preprocessing / augmentation:** auto-orient; resized to 640×640 **with stretch**;
  3 versions per training image with random brightness ±15%
- **Citation:** none given — cite the Roboflow Universe page

#### Counts — `pothole-segmentation-7fnkw-v2`

- Classes: `manhole`, `pothole`, `sewer`, `vehicle`
- Images: 4793 (train 4205, valid 392, test 196)
- Original images (Roboflow `.rf.` copies removed): 1990 (train 1402, valid 392, test 196)
- Originals appearing in more than one split (leakage): 0
- Images without labels: 16 (6 originals) — natural negatives
- Box-only labels (no polygon): 0 — every label is a polygon
- Originals with at least one pothole: 1,930

| Class | Annotations | Images | Median area | tiny <0.1% | small 0.1-1% | medium 1-5% | large >=5% |
|---|---|---|---|---|---|---|---|
| `vehicle` | 11594 | 3408 | 0.51% | 1716 | 5493 | 3317 | 1068 |
| `pothole` | 6632 | 4643 | 0.13% | 2776 | 3031 | 797 | 28 |
| `sewer` | 1053 | 688 | 0.20% | 325 | 588 | 114 | 26 |
| `manhole` | 916 | 661 | 0.24% | 251 | 491 | 171 | 3 |

(Counts include the 3× augmented training copies.)

**Within-set near-duplicates:** 772 of 1,990 originals have another original at
dHash Hamming ≤ 4 — consecutive dashcam frames (`PM_*`/`AM_*` with a frame index,
`frame_<video>_<n>`, `V*F_HY_*`). Expect a large drop in de-duplication; group by video.

### Quality check (50 sampled images)

| Check | Notes |
|---|---|
| Viewpoint | Mostly dashcam (windscreen and bonnet visible), some handheld close-ups; no drone frames |
| Countries / road types | **Korea** — Hangul road markings, file names with Busan, Seoul, Yuseong-gu; city streets, alleys, main roads |
| Lighting and weather | Day; sunny and rainy (names include `sunny`, `rainy`, `Rain`) |
| Pothole masks | Tight polygons; potholes are mostly small and distant (median 0.13% of image) |
| Road masks | **None** — road visible but unlabelled → pseudo-label road |
| Missing potholes | None obvious in sample |
| Near-duplicate frames | Many (see above) |
| Faces / plates | Parked cars with plates, some pedestrians; blur if shown publicly |

**Decision (2026-11-18):** **use.** Keep `pothole`. Delete `manhole` and `sewer`
labels so they become hard negatives (look-alikes that are not potholes); delete
`vehicle` labels (background). Road gets pseudo-labels in Week 2. Fills the
"few or no Korean roads" gap in the plan.

### Cross-dataset overlap (4 vs 1 and 2)

dHash: **0** matches at Hamming ≤ 4 with either set; 36 (vs 1) and 25 (vs 2) loose
matches at ≤ 8 with unrelated file names (likely false positives).

---

## Label budget so far (before cross-set de-duplication)

| Source | Originals | `road` labels | Full pothole masks |
|---|---|---|---|
| 1. Potholes and Roads (after dropping images with mask-less potholes) | 904 | real | yes |
| 2. road segmentation (minus 30 DJI frames) | 886 | real | yes |
| 4. GNSoft2 | 1,990 | **pseudo** | yes |
| **Total** | **3,780** | 1,790 real + 1,990 pseudo | |

Within-set duplicates (esp. GNSoft2's video frames) will lower this in Week 2. The
plan's 2,000–4,000 target is likely met, so SAM conversion of dataset 1 stays shelved.
