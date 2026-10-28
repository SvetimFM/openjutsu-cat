# RGB video -> depth-style control video in the exact spec of the hero pipeline
# (640x640, 16 fps, near = bright), using Depth-Anything-V2-Small (open weights).
# Decoding/encoding via ffmpeg CLI; depth via transformers pipeline. Stdlib + torch/PIL.
# Usage: depth_estimate.py <input_video> <output_mp4> [start_seconds] [frames]
import os, subprocess, sys, glob
import numpy as np
from PIL import Image

def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("CMD FAILED:", " ".join(cmd), "\n", r.stderr[-1500:], flush=True)
        sys.exit(2)

def main():
    src, out = sys.argv[1], sys.argv[2]
    start = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    nframes = int(sys.argv[4]) if len(sys.argv) > 4 else 49
    tmp = os.path.join(os.path.dirname(os.path.abspath(out)), "depthtmp")
    os.makedirs(tmp, exist_ok=True)
    # 1. extract frames at 16 fps, center-crop square, 640x640
    sh(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start), "-i", src,
        "-vf", f"fps=16,crop='min(iw,ih)':'min(iw,ih)',scale=640:640", "-frames:v", str(nframes),
        os.path.join(tmp, "rgb_%03d.png"), "-y"])
    frames = sorted(glob.glob(os.path.join(tmp, "rgb_*.png")))
    print(f"extracted {len(frames)} frames", flush=True)
    if not frames:
        print("NO FRAMES EXTRACTED", flush=True); sys.exit(3)
    # 2. depth per frame (collect raw maps first for clip-global normalization)
    import torch
    from transformers import pipeline
    dev = 0 if torch.cuda.is_available() else -1
    print("device:", "cuda" if dev == 0 else "cpu", flush=True)
    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=dev)
    raws = []
    for f in frames:
        img = Image.open(f).convert("RGB")
        d = pipe(img)["depth"]  # PIL, higher = closer (inverse depth)
        raws.append(np.asarray(d, dtype=np.float32))
        print(".", end="", flush=True)
    print(f"\ndepth done for {len(raws)} frames", flush=True)
    lo = min(r.min() for r in raws); hi = max(r.max() for r in raws)
    print(f"clip depth range {lo:.2f}..{hi:.2f}", flush=True)
    # 3. normalize clip-globally, near = bright; write gray frames; encode
    for i, r in enumerate(raws):
        n = (r - lo) / max(hi - lo, 1e-6)
        Image.fromarray((n * 255).astype(np.uint8), mode="L").convert("RGB").save(
            os.path.join(tmp, f"dep_{i:03d}.png"))
    sh(["ffmpeg", "-hide_banner", "-loglevel", "error", "-framerate", "16",
        "-i", os.path.join(tmp, "dep_%03d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        out, "-y"])
    print("WROTE", out, os.path.getsize(out), "bytes", flush=True)

if __name__ == "__main__":
    main()
