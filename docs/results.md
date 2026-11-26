# Results

One row per experiment (CLAUDE.md). `training/scripts/train.py` prints the row;
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
