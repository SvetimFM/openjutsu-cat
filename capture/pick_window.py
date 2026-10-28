#!/usr/bin/env python3
"""Find the stretch of a depth take where the performer moves most.

Cats don't perform on cue: a 25 s take might hold 3 s of crossing. Motion is
scored per frame pair as the fraction of pixels whose depth changed by more
than `--thresh` gray levels (~190 mm at the default 12) — sensor speckle and
invalid (black) pixels don't count — and the best N-frame window is the one
with the most total motion that doesn't overlap a capture stall.

  pick_window.py control.mp4 --frames 49                    # print the window
  pick_window.py control.mp4 --frames 97 --write hero.mp4   # and cut it out
Requires: numpy, ffmpeg on PATH."""
import argparse
import subprocess
import sys

import numpy as np


def motion_scores(frames, thresh=12):
    """frames: uint8 [N, H, W] -> float [N-1], motion between frame i and i+1."""
    a = frames[:-1].astype(np.int16)
    b = frames[1:].astype(np.int16)
    valid = (a > 0) & (b > 0)
    moved = (np.abs(b - a) > thresh) & valid
    return moved.reshape(len(moved), -1).mean(axis=1)


def best_window(scores, n, blocked=None):
    """(start, total) of the n-frame window with the most motion, skipping any
    window that contains a blocked frame; None if no window fits."""
    total_frames = len(scores) + 1
    if n > total_frames:
        return None
    csum = np.concatenate([[0.0], np.cumsum(scores)])
    bad = np.concatenate([[0], np.cumsum(np.asarray(blocked if blocked is not None else
                                                    [False] * total_frames, dtype=int))])
    best = None
    for s in range(total_frames - n + 1):
        if bad[s + n] - bad[s]:
            continue
        tot = csum[s + n - 1] - csum[s]
        if best is None or tot > best[1]:
            best = (s, float(tot))
    return best


def read_mp4_gray(path, size=640):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode(errors="replace")[-1500:])
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", path], capture_output=True, text=True)
    w, h = (int(x) for x in probe.stdout.strip().split(","))
    return np.frombuffer(out.stdout, dtype=np.uint8).reshape(-1, h, w)


def cut(src, start, n, dst, fps=16):
    """Copy frames [start, start+n) of src into dst, re-encoded at fps."""
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-vf",
                    f"select='between(n\\,{start}\\,{start + n - 1})',setpts=N/({fps}*TB)",
                    "-r", str(fps), "-frames:v", str(n), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", dst], check=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--frames", type=int, default=49)
    p.add_argument("--thresh", type=int, default=12)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--write", help="also write the chosen window to this mp4")
    a = p.parse_args(argv)
    frames = read_mp4_gray(a.video)
    win = best_window(motion_scores(frames, a.thresh), a.frames)
    if win is None:
        sys.exit(f"{a.video} has {len(frames)} frames, fewer than --frames {a.frames}")
    start, score = win
    print(f"best {a.frames}-frame window: frames {start}..{start + a.frames - 1} "
          f"({start / a.fps:.2f}s-{(start + a.frames) / a.fps:.2f}s), motion {score:.3f}")
    if a.write:
        cut(a.video, start, a.frames, a.write, a.fps)
        print(f"wrote {a.write}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
