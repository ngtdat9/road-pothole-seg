# Failure analysis — baseline (nano, 640 px) on val

Experiment 5 of the plan: where does the chosen model fail, and why?
Model: `runs/baseline/weights/best.pt` (commit 29bd2a4). Split: **val** (471 images), never
test. Matching: a prediction with confidence ≥ 0.25 (`configs/pipeline.yaml`) finds a
labelled pothole when their masks overlap with IoU ≥ 0.5 (greedy, highest confidence
first). Made with `training/scripts/failures.py`; per-image numbers in
`docs/runs/baseline/failures_val.json`, the 28 cases in `docs/failures_baseline.csv`.

The gallery sheets (`outputs/failures/baseline/*.jpg`, not in git) show missed labels in
yellow, found labels in white, false predictions in red, matches in blue and predicted
road in green. Frames are not published until faces and plates are blurred.

## Overall

| Pothole | Value |
|---|---|
| Labels / predictions | 696 / 649 |
| Found / missed / false | 488 / 208 / 161 |
| Precision / recall / F1 | 0.752 / 0.701 / 0.726 |
| Mean mask IoU of found potholes | 0.782 |

| Pothole size (share of image) | Labels | Recall |
|---|---|---|
| tiny < 0.1% | 148 | **0.55** |
| small 0.1–1% | 367 | 0.73 |
| medium 1–5% | 133 | 0.74 |
| large ≥ 5% | 48 | 0.90 |

Road (264 images with hand-labelled road): mean IoU **0.908**; 24 images below 0.8.

## The 20 worst pothole images, by cause

Ranked by missed + false potholes, then missed area.

| Cause | Images | What happens |
|---|---|---|
| distant | 5 | Potholes far ahead are a few pixels wide: missed, or found but shifted so IoU < 0.5 |
| crack | 4 | Long cracks and joints are labelled as potholes in some sets (GNSoft2 `ys_`); the model splits or skips them |
| label error | 3 | Wrong labels: a truck wheel and cones, a motorbike dashboard, unlabelled broken asphalt |
| other | 3 | Faint, shallow damage with little contrast; one large close-up only partly covered |
| puddle | 2 | Water-filled potholes reflect the sky; far ones are missed |
| patch | 2 | Patched or sunken asphalt where the pothole edge is ambiguous |
| small | 1 | Dense cluster of small potholes; neighbours merge or split differently from the labels |

Shadows and manholes, both named in the plan, did not appear among the 20 worst pothole
cases (one manhole shows up in the road list).

**About a third of the worst cases are label problems** (3 label errors and the 4 crack
cases, whose labels are inconsistent across sets). Some of the model's "errors" there are
correct behaviour, so val scores understate it slightly.

## The 8 worst road images

All 8 are close-ups of one large pothole: the model's road stops at the pothole edge,
while the hand labels draw road over it. Filling predicted potholes into the road mask
raises mean road IoU from 0.908 to **0.927** (images below 0.8: 24 → 13). This becomes a
post-processing step in the package (Week 4 decision: the road mask returned is road ∪
potholes).

## What the tuning runs showed

Three attempts to beat the baseline all landed within 0.01 on pothole mask mAP50-95
(small model 0.403, 150 epochs 0.400, stronger zoom-out 0.400 vs 0.408), so the errors
above are not a settings problem. Tiny and distant potholes need more pixels (a larger
input, which costs CPU speed) or more and cleaner data.

## Limitations to state in the README

- Distant and tiny potholes are often missed in a single frame (recall 0.55). On video the
  tracker sees them again as they get closer, so the logged count is higher than this.
- Mask edges match labels at IoU ~0.78, so area and severity are estimates (roughly ±20%).
- Water-filled potholes, cracks and patched asphalt are confused with potholes, both ways.
- The training sets contain some wrong labels (vehicles, cracks); a label audit would help v1.1.
