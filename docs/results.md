# Results

One row per experiment (CLAUDE.md). `training/scripts/train.py` prints the row;
add a one-line conclusion. Pothole metrics come from the run's val list, road
metrics only from `val_real_road.txt` (images with hand-drawn road labels).
Mask mAP from Ultralytics; test-split numbers appear only in the final evaluation.

| Date | Run | Model | Train list | Epochs | imgsz | Pothole mAP50 | Pothole mAP50-95 | Road mAP50 | Road mAP50-95 | Commit | Conclusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
