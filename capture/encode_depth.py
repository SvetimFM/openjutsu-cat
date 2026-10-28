#!/usr/bin/env python3
"""Encode KinectBurst raw depth frames (fNNN.bin, 512x424 uint16 LE, mm)
into the benchmarked ComfyUI control-input spec: grayscale mp4, near=bright,
640x640 (upscale to 640x508 + black pad), 16 fps.

Mapping matches the KinectBurst viz (KinectBurst.cs SaveViz):
  d == 0 -> 0 ; else v = clamp(255 - (d - 500) * 255 / 4000)
Invalid/zero reads as far (black), same as padding.

Timing: KinectBurst's real frame interval is the requested interval PLUS the
time it takes to write each frame (~30-50 ms), and the sensor occasionally
stalls for seconds. Stamping frames at 16 fps as-is would play the performance
back too fast. So when burst-log.csv is present, frames are resampled by their
logged timestamps onto an exact 16 fps grid (nearest frame per tick), which
plays back in real time, and any stall longer than --max-gap-ms is reported.

Usage: encode_depth.py <frames_dir> <out.mp4> [fps=16] [--no-resample] [--max-gap-ms 150]
Requires: numpy, ffmpeg on PATH."""
import argparse
import csv
import glob
import os
import re
import subprocess
import sys

import numpy as np

W, H = 512, 424
FPS = 16


def depth_to_gray(d):
    d = d.astype(np.float32)
    v = np.where(d == 0, 0, 255.0 - (d - 500.0) * 255.0 / 4000.0)
    return np.clip(v, 0, 255).astype(np.uint8)


def list_frames(frames_dir):
    """[(index, path)] sorted by frame number (f999 < f1000)."""
    out = []
    for p in glob.glob(os.path.join(frames_dir, "f*.bin")):
        m = re.fullmatch(r"f(\d+)\.bin", os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def read_frame(path):
    d = np.fromfile(path, dtype="<u2")
    if d.size != W * H:
        raise ValueError(f"{path}: {d.size} px, expected {W * H}")
    return depth_to_gray(d.reshape(H, W))


def read_log(frames_dir):
    """{frame index: unix_ms} from burst-log.csv, or {} when there is no log."""
    path = os.path.join(frames_dir, "burst-log.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {int(r["index"]): int(r["unix_ms"]) for r in csv.DictReader(f)}


def resample(ts_ms, fps=FPS, max_gap_ms=150):
    """Map a sorted list of capture timestamps onto an exact fps grid.

    Returns (picks, blocked): picks[k] is the position in ts_ms of the frame
    nearest to tick k; blocked[k] is True when tick k falls inside a capture
    stall longer than max_gap_ms (the picked frame is then a frozen stand-in).
    """
    ts = np.asarray(ts_ms, dtype=np.float64)
    if len(ts) < 2:
        return list(range(len(ts))), [False] * len(ts)
    step = 1000.0 / fps
    ticks = ts[0] + step * np.arange(int((ts[-1] - ts[0]) // step) + 1)
    after = np.searchsorted(ts, ticks, side="left").clip(1, len(ts) - 1)
    before = after - 1
    picks = np.where(ticks - ts[before] <= ts[after] - ticks, before, after)
    blocked = (ts[after] - ts[before]) > max_gap_ms
    return picks.tolist(), blocked.tolist()


class Take:
    """A capture take, as 16 fps gray frames ready for the control clip."""

    def __init__(self, frames_dir, fps=FPS, do_resample=True, max_gap_ms=150):
        listed = list_frames(frames_dir)
        if not listed:
            raise ValueError(f"no fNNN.bin frames in {frames_dir}")
        log = read_log(frames_dir)
        if do_resample and log:
            listed = [(i, p) for i, p in listed if i in log]
            ts = [log[i] for i, _ in listed]
            picks, self.blocked = resample(ts, fps, max_gap_ms)
            gaps = np.diff(ts)
            self.stats = {
                "captured_frames": len(listed),
                "span_s": round((ts[-1] - ts[0]) / 1000.0, 2),
                "median_interval_ms": float(np.median(gaps)) if len(gaps) else None,
                "achieved_fps": round(1000.0 * (len(ts) - 1) / max(ts[-1] - ts[0], 1), 2),
                "stalls_ms": [int(g) for g in gaps if g > max_gap_ms],
                "resampled": True,
            }
        else:
            picks, self.blocked = list(range(len(listed))), [False] * len(listed)
            self.stats = {"captured_frames": len(listed), "resampled": False}
        self.source_index = [listed[k][0] for k in picks]
        cache = {}
        self.frames = np.stack([cache.setdefault(k, read_frame(listed[k][1])) for k in picks])
        self.fps = fps
        self.stats["output_frames"] = len(self.frames)


def write_mp4(frames, out, fps=FPS):
    """uint8 [N, 424, 512] gray -> 640x640 control mp4 (H.264, yuv420p)."""
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
        "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
        "-vf", "scale=640:508:flags=lanczos,pad=640:640:0:66:black",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out,
    ]
    p = subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode(errors="replace")[-2000:])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("frames_dir")
    p.add_argument("out")
    p.add_argument("fps", nargs="?", type=int, default=FPS)
    p.add_argument("--no-resample", action="store_true", help="stamp frames in order, ignoring timestamps")
    p.add_argument("--max-gap-ms", type=int, default=150)
    a = p.parse_args(argv)
    take = Take(a.frames_dir, a.fps, not a.no_resample, a.max_gap_ms)
    write_mp4(take.frames, a.out, a.fps)
    print(f"encoded {take.stats} -> {a.out}")
    if take.stats.get("stalls_ms"):
        print(f"WARNING: sensor stalls {take.stats['stalls_ms']} ms - pick a window that avoids them "
              f"(capture/pick_window.py / process_take.py do this for you)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError) as e:
        sys.exit(f"error: {e}")
