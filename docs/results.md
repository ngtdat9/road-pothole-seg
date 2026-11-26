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
