from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from apps.make_demo import gif_frames, write_gif


def _video(path: Path, n: int = 30, fps: float = 10.0) -> None:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 120))
    for i in range(n):
        w.write(np.full((120, 160, 3), i * 8, dtype=np.uint8))
    w.release()


def test_gif_frames_samples_the_time_range(tmp_path: Path) -> None:
    video = tmp_path / "v.avi"
    _video(video)  # 3 s at 10 FPS
    frames = gif_frames(video, start_s=1.0, duration_s=1.0, gif_fps=5.0, width=80)
    assert len(frames) == 6  # 1.0, 1.2, ... 2.0 s
    assert frames[0].shape == (60, 80, 3)


def test_write_gif_loops_all_frames(tmp_path: Path) -> None:
    frames = [np.full((20, 30, 3), v, dtype=np.uint8) for v in (0, 120, 240)]
    path = tmp_path / "out" / "demo.gif"
    write_gif(frames, path, gif_fps=10.0)
    with Image.open(path) as gif:
        assert gif.n_frames == 3 and gif.info["loop"] == 0
