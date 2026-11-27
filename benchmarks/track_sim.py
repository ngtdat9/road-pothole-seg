"""Tracker check on synthetic clips made from val images (Mac; val only, never test).

dashcam: 60 frames at 30 FPS zooming 1x -> 2.5x about a point on the horizon, so
potholes grow and slide down and out, as when driving forward.
drone: 20 frames at 5 Hz panning across the image (6% of the width per frame).
Labels are warped with the frames, giving ground-truth tracks. A tracked pothole
matches a ground-truth one at box IoU >= 0.3. Reports, per tracker setting: coverage
(share of visible ground-truth frames with a tracked pothole), ID switches and false
tracks (IDs never matched to a ground-truth pothole), plus the detector-only coverage.
Static photos have no blur or lighting change, so real video will score lower.

Usage:
    python -m benchmarks.track_sim dashcam
    python -m benchmarks.track_sim drone
"""

import itertools
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from road_pothole_seg import PotholeTracker, Segmenter, analyse_potholes
from road_pothole_seg.tracker import box_iou, greedy_match

REPO = Path(__file__).resolve().parents[1]
root = REPO / "data" / "merged-v1"
cfg = yaml.safe_load((REPO / "configs/pipeline.yaml").read_text())


def polys(line, w, h):
    out = []
    for row in (root / "labels" / (Path(line).stem + ".txt")).read_text().splitlines():
        v = row.split()
        if v and v[0] == "1":
            out.append(np.array(v[1:], float).reshape(-1, 2) * [w, h])
    return out


def warps(mode, w, h):
    if mode == "dashcam":
        fx, fy = w / 2, 0.35 * h
        ow, oh = 1280, round(1280 * h / w)
        for k in range(60):
            z = 1 + 1.5 * k / 59
            s = ow / (w / z)
            yield np.float32([[s, 0, -(fx - fx / z) * s], [0, s, -(fy - fy / z) * s]]), (ow, oh)
    else:  # drone pan: window 60% of the width moving right
        cw = 0.6 * w
        ow = 960
        s = ow / cw
        oh = round(h * s)
        for k in range(20):
            x0 = 0.4 * w * k / 19
            yield np.float32([[s, 0, -x0 * s], [0, s, 0]]), (ow, oh)


def clips(mode, n, seed=0):
    lines = (root / "val.txt").read_text().split()
    random.Random(seed).shuffle(lines)
    seg = Segmenter(str(REPO / "models/baseline-640.onnx"), **cfg["segmenter"])
    made = 0
    for line in lines:
        img = cv2.imread(str(root / line))
        h, w = img.shape[:2]
        P = polys(line, w, h)
        if w < 1.2 * h or not P:
            continue
        frames = []
        for M, size in warps(mode, w, h):
            f = cv2.warpAffine(img, M, size, flags=cv2.INTER_LINEAR)
            gt = []
            for g, poly in enumerate(P):
                q = poly @ M[:, :2].T + M[:, 2]
                full = np.r_[q.min(0), q.max(0)]
                cl = np.r_[np.clip(full[:2], 0, size), np.clip(full[2:], 0, size)]
                a_full = np.prod(full[2:] - full[:2])
                a_cl = np.prod((cl[2:] - cl[:2]).clip(0))
                if a_full > 0 and a_cl / a_full >= 0.5:
                    gt.append((g, cl))
            dets = analyse_potholes(seg(f), **cfg["postprocess"])
            frames.append((gt, dets))
        yield Path(line).name, frames
        made += 1
        if made == n:
            return


def score(clip_frames, tracker_kwargs):
    tr = PotholeTracker(**{**cfg["tracker"], **tracker_kwargs})
    hist = {}  # gt id -> track id per visible frame (None: not tracked)
    out_ids, matched_ids = set(), set()
    for gt, dets in clip_frames:
        tracked = tr.update(dets)
        tb = np.array([p.bbox_xyxy for p in tracked], float).reshape(-1, 4)
        gb = np.array([b for _, b in gt], float).reshape(-1, 4)
        pairs = dict(greedy_match(box_iou(gb, tb), 0.3))
        for r, (g, _) in enumerate(gt):
            hist.setdefault(g, []).append(tracked[pairs[r]].track_id if r in pairs else None)
        out_ids |= {p.track_id for p in tracked}
        matched_ids |= {tracked[c].track_id for c in pairs.values()}
    res = {"gt": 0, "vis": 0, "cov": 0, "ids": 0, "switch": 0}
    for seq in hist.values():
        if len(seq) < 5:
            continue
        ids = [x for x in seq if x is not None]
        res["gt"] += 1
        res["vis"] += len(seq)
        res["cov"] += len(ids)
        res["ids"] += len(set(ids))
        res["switch"] += sum(a != b for a, b in zip(ids, ids[1:], strict=False))
    res["false_tracks"] = len(out_ids - matched_ids)
    return res


def detector_coverage(data):
    vis = cov = 0
    for _, fr in data:
        hist = {}
        for gt, dets in fr:
            db = np.array([p.bbox_xyxy for p in dets], float).reshape(-1, 4)
            gb = np.array([b for _, b in gt], float).reshape(-1, 4)
            pairs = dict(greedy_match(box_iou(gb, db), 0.3))
            for r, (g, _) in enumerate(gt):
                hist.setdefault(g, []).append(r in pairs)
        for seq in hist.values():
            if len(seq) >= 5:
                vis += len(seq)
                cov += sum(seq)
    return cov / vis


if __name__ == "__main__":
    mode = sys.argv[1]
    data = list(clips(mode, 20))
    conf = cfg["segmenter"]["conf"]
    print(f"{mode}: detector alone (conf {conf}) coverage {detector_coverage(data):.3f}")
    rate = {"frame_rate": 5.0} if mode == "drone" else {}
    grid = itertools.product([0.25, 0.3, 0.4, 0.5], [1, 2, 3], [0.0, 0.3])
    for th, mh, buf in grid:
        g = dict(track_thresh=th, min_hits=mh, match_buffer=buf, **rate)
        tot = {}
        for _, fr in data:
            for k, v in score(fr, g).items():
                tot[k] = tot.get(k, 0) + v
        print(
            f"track_thresh {th} min_hits {mh} match_buffer {buf}: coverage "
            f"{tot['cov'] / tot['vis']:.3f}, switches {tot['switch']}, "
            f"false tracks {tot['false_tracks']} (GT tracks {tot['gt']})"
        )
