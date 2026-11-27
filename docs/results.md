# Results

One row per experiment. `training/scripts/train.py` prints the row;
add a one-line conclusion. Pothole metrics come from the run's val list, road
metrics only from `val_real_road.txt` (images with hand-drawn road labels).
Mask mAP from Ultralytics; test-split numbers appear only in the final evaluation.

| Date | Run | Model | Train list | Epochs | imgsz | Pothole mAP50 | Pothole mAP50-95 | Road mAP50 | Road mAP50-95 | Commit | Conclusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-25 | sanity | yolo26n-seg | train.txt | 5 | 640 | 0.653 | 0.346 | 0.946 | 0.702 | 2e77d2a | Pipeline works end to end (disk cache, Drive saves, split metrics); ~57 s/epoch on T4; still improving at epoch 5; mosaic was off throughout (close_mosaic 10 ≥ 5 epochs) |
| 2026-11-25 | teacher | yolo26n-seg | train_real_road.txt | 30 | 640 | 0.653 | 0.340 | 0.958 | 0.785 | 2e77d2a | Road teacher ready: road mask mAP50-95 0.785 on val_real_road (sanity 0.702, same list); plateaued from ~epoch 25; ~36 s/epoch. Pothole here is on real-road val only, not comparable to sanity |
| 2026-11-25 | baseline | yolo26n-seg | train.txt | 100 | 640 | 0.722 | 0.408 | 0.955 | 0.789 | 29bd2a4 | First real baseline on merged-v1: pothole mask mAP50-95 0.408 (sanity 0.346 on base val); road 0.789 = teacher 0.785, so pseudo road did not hurt real-road accuracy. Best epoch 90, still creeping up at 100; ~62 s/epoch, 1.7 h. merged-v1 val differs from base val (2 no-road images removed, vehicle labels dropped), so pothole vs sanity is only roughly comparable |
| 2026-11-26 | baseline @480 | yolo26n-seg | train.txt | 100 | 480 | 0.687 | 0.375 | 0.957 | 0.805 | 7af1cfd | Same weights validated at 480 (trained at 640): pothole −0.034 mAP50-95, road unchanged |
| 2026-11-26 | baseline @320 | yolo26n-seg | train.txt | 100 | 320 | 0.640 | 0.317 | 0.953 | 0.785 | 7af1cfd | At 320 pothole drops −0.091 (small potholes vanish); road holds. Road is size-robust, pothole is not |
| 2026-11-26 | small | yolo26s-seg | train.txt | 100 | 640 | 0.717 | 0.403 | 0.959 | 0.801 | 69f891e || Exp 2: small does not beat nano on pothole (0.403 vs 0.408, −0.006); road +0.012 (0.801 vs 0.789). ~75 s/epoch vs 62, 2.1 h; resumed once after a Colab disconnect at epoch 87. Nano stays the choice (within 0.01, prefer nano; small costs ~3× on CPU) |
| 2026-11-26 | tune1 | yolo26n-seg | train.txt | 150 | 640 | 0.723 | 0.400 | 0.957 | 0.794 | e2986ac || 150 epochs did not help: pothole 0.400 vs baseline 0.408 (−0.008), road 0.794 (+0.005). The longer LR schedule only shifts the curve (in-training val 0.394 at epoch 100, best 0.408 at 146); 100 epochs is already the plateau. ~63 s/epoch, 2.6 h |
| 2026-11-26 | tune2 | yolo26n-seg | train.txt | 100 | 640 | 0.707 | 0.400 | 0.959 | 0.792 | d467f9c || Stronger zoom-out (scale 0.7) did not help: pothole 0.400 vs baseline 0.408 (−0.008), mAP50 0.707 vs 0.722 (−0.015); road 0.792 (+0.003). Baseline stays the best model; tiny-pothole misses need more data or input size, not augmentation |
| 2026-11-26 | naive | yolo26n-seg | train.txt | 100 | 640 | 0.771 | 0.443 | 0.963 | 0.815 | 58c3ffe || Exp 1, dataset merged-v1-naive (duplicates kept, random split). Scores are on the naive val, where 62% of images share a dedup group with train: pothole 0.443 / mAP50 0.771 vs baseline 0.408 / 0.722 on clean val. Inflated, not comparable; the honest comparison is both models on test_clean (pending) |
| 2026-11-26 | baseline TEST @640 | yolo26n-seg | train.txt | 100 | 640 | 0.720 | 0.401 | 0.980 | 0.806 | 83cdde2 | **Final test of the chosen model (opened once).** Matches val (0.722 / 0.408), so model selection on val did not overfit |
| 2026-11-26 | baseline TEST @480 | yolo26n-seg | train.txt | 100 | 480 | 0.694 | 0.372 | 0.982 | 0.818 | 83cdde2 | Final test at 480: pothole −0.026 mAP50, road unchanged |
| 2026-11-26 | baseline TEST @320 | yolo26n-seg | train.txt | 100 | 320 | 0.593 | 0.303 | 0.971 | 0.809 | 83cdde2 | Final test at 320: pothole −0.127 mAP50; road holds |
| 2026-11-26 | naive on naive test | yolo26n-seg | merged-v1-naive | 100 | 640 | 0.780 | 0.454 | 0.981 | 0.819 | 83cdde2 | Exp 1: the score a naive merge would report |
| 2026-11-26 | naive on clean test | yolo26n-seg | merged-v1-naive | 100 | 640 | 0.714 | 0.399 | 0.975 | 0.800 | 83cdde2 | Exp 1: same model on merged-v1's test: −0.067 mAP50. Equal to baseline (0.720), so duplicates add no real skill, only inflate the reported score |
| 2026-11-27 | baseline ONNX @640 | yolo26n-seg | train.txt | 100 | 640 | 0.715 | 0.402 | 0.953 | 0.803 | 946d823 | Exported model on val (square input, batch 1). Equal to best.pt with square input (0.715 / 0.402 / 0.803), so the export is lossless; −0.007 mAP50 vs the batched rect val above comes from square padding |
| 2026-11-27 | baseline ONNX @480 | yolo26n-seg | train.txt | 100 | 480 | 0.683 | 0.362 | 0.961 | 0.808 | 946d823 | ONNX at 480 on val |
| 2026-11-27 | baseline ONNX @320 | yolo26n-seg | train.txt | 100 | 320 | 0.612 | 0.307 | 0.945 | 0.770 | 946d823 | ONNX at 320 on val; best.pt with square input gives 0.612 / 0.307 / 0.770, identical |

## Experiment 1 — duplicate leakage (test, 640 px, pothole mask)

| Model | Scored on | mAP50 | mAP50-95 |
|---|---|---|---|
| naive merge | its own random test (leaky) | 0.780 | 0.454 |
| naive merge | clean test | 0.714 | 0.399 |
| de-duplicated, grouped (baseline) | clean test | 0.720 | 0.401 |

A naive merge overstates pothole mask mAP50 by 0.06–0.07 (about 9%) and mAP50-95 by
0.05 (13%). On the same clean test the two models are equal, so the extra duplicate
images taught nothing new. Road barely moves (+0.02 on the leaky test), as road looks
alike in every image.

## ONNX export check (Week 4)

`training/scripts/check_onnx.py` on 20 val images, each in 4 shapes (original, 16:9 crop,
9:16 crop, 400x300), 80 frames, 640 px model. Details in `docs/runs/baseline/onnx_check.json`.

| Package `Segmenter` vs | Potholes matched | Mean / min mask IoU | Max score diff | Unmatched (ours / ref) | Mean road IoU |
|---|---|---|---|---|---|
| Ultralytics, same ONNX file | 115 | **0.996** / 0.924 | 0.000 | 0 / 0 | 0.999 |
| Ultralytics, best.pt | 105 | 0.977 / 0.774 | 0.116 | 10 / 1 | 0.985 |

Against the same ONNX file the detections and scores are identical; the remaining mask
difference is rounding at mask edges (lowest on the small 400x300 frames, 0.988). Against
best.pt the input differs on non-square frames: PyTorch letterboxes to the smallest
32-multiple rectangle, ONNX to a fixed square. This does not cost accuracy (val above:
ONNX = best.pt with square input at 640 and 320). A first version that thresholded after
resizing to the frame scored only 0.982 (small potholes grew by a row of pixels), so the
package now follows Ultralytics' order: threshold at input size, crop to box, then resize.

## Tracker on synthetic clips (Week 4)

`benchmarks/track_sim.py`, 20 val images per mode, 640 px ONNX, settings from
`configs/pipeline.yaml` (track_thresh 0.4, min_hits 2, match_buffer 0.3). Coverage =
share of visible labelled-pothole frames with a tracked pothole (box IoU ≥ 0.3).

| Mode | GT tracks | Detector alone | Tracker coverage | ID switches | False tracks |
|---|---|---|---|---|---|
| dashcam (zoom, 30 FPS, 60 frames) | 29 | 0.811 | 0.742 | 0 | 6 |
| drone (pan, 5 Hz, 20 frames) | 30 | 0.825 | 0.725 | 0 | 2 |
