#!/usr/bin/env python3
"""One command from a raw KinectBurst take to judged-ready clips.

  process_take.py <take_dir>               # encode, pick windows, render, side-by-sides
  process_take.py <take_dir> --no-render   # just the control clips + plan (no GPU)

1. Encode the take at exactly 16 fps from its logged timestamps (real-time
   playback; sensor stalls are marked and never used).
2. Pick the highest-motion stall-free windows: a long hero window (97 frames,
   ~6 s) and a slate window (49 frames, ~3 s).
3. Render the hero costume on the hero window, then each slate costume on the
   slate window, through the shipped depth workflow (run_local.depth_graph).
4. Build a 1280x640 side-by-side (depth | render) for every render.

Everything lands in a NEW folder <take_dir>/process-<timestamp>/ — nothing
existing is overwritten — with process.json recording windows, timings and files.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import comfy_client  # noqa: E402
from encode_depth import Take, write_mp4  # noqa: E402
from pick_window import best_window, motion_scores  # noqa: E402
from run_local import check_frames, depth_graph  # noqa: E402

# The costume slate, judged on the synthetic clip 2026-10-26 (seed 2077, fixed
# settings; only the prompt varies). Default order is the judged order.
COSTUMES = {
    "A": "A majestic cat made of glowing blue spirit light walks through a dark mystical forest, volumetric glow, cinematic",
    "B": "A cat made of glowing stained glass panels with dark lead outlines walks through a cathedral of coloured light, jewel tones, cinematic",
    "C": "A tiny cat knight in shining silver armour with a red plume walks through a miniature medieval castle hall, cinematic, detailed",
    "D": "A cat painted in black sumi-e ink brushstrokes walks across a blank white paper background, minimal, elegant, high contrast",
    "E": "A clay stop-motion cat with visible fingerprints in the clay walks across a miniature handmade cardboard set, warm practical light, whimsical",
    "F": "A cat made of flowing koi-patterned light, orange and white, swims-walks through a dark pond at night, ripples of light, cinematic",
}
# Processes that mean someone is using the GPU for something else (games).
DEFAULT_GPU_GUARD = "bf6.exe,Cyberpunk2077.exe"


def running_processes():
    if os.name != "nt":
        return set()
    out = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True).stdout
    return {line.split('","')[0].strip('"').lower() for line in out.splitlines() if line}


def side_by_side(control, render, dst):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", control, "-i", render, "-filter_complex",
                    "[0:v][1:v]hstack=inputs=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    dst], check=True)


def window_info(take, start, n):
    return {"start": start, "frames": n, "start_s": round(start / take.fps, 2),
            "end_s": round((start + n) / take.fps, 2),
            "source_frames": [take.source_index[start], take.source_index[start + n - 1]]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("take_dir")
    p.add_argument("--hero", default="D", help="costume rendered on the long hero window (default D, sumi-e)")
    p.add_argument("--hero-frames", type=int, default=97)
    p.add_argument("--costumes", default="D,C,A,F", help="slate costumes, in render order")
    p.add_argument("--frames", type=int, default=49, help="slate window length")
    p.add_argument("--max-gap-ms", type=int, default=150, help="longer capture stalls are never used")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--server", default=comfy_client.DEFAULT_SERVER)
    p.add_argument("--gpu-guard", default=DEFAULT_GPU_GUARD,
                   help="comma list of process names that block rendering ('' to disable)")
    a = p.parse_args(argv)
    costumes = [c for c in a.costumes.split(",") if c]
    for c in costumes + [a.hero]:
        if c not in COSTUMES:
            p.error(f"unknown costume {c!r}; choose from {sorted(COSTUMES)}")
    check_frames(a.frames)
    check_frames(a.hero_frames)

    take_name = os.path.basename(os.path.normpath(a.take_dir))
    out = os.path.join(a.take_dir, "process-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out)  # fails if it exists: never overwrite
    take = Take(a.take_dir, max_gap_ms=a.max_gap_ms)
    print(f"take {take_name}: {take.stats}")
    if take.stats.get("resampled") and not 13 <= take.stats["achieved_fps"] <= 19:
        print(f"note: captured at {take.stats['achieved_fps']} fps; resampled to 16 fps by timestamp")

    report = {"take": take_name, "stats": take.stats, "windows": {}, "renders": []}
    write_mp4(take.frames, os.path.join(out, "control-full.mp4"))
    scores = motion_scores(take.frames)
    controls = {}
    for label, n in (("hero", a.hero_frames), ("slate", a.frames)):
        win = best_window(scores, n, take.blocked)
        if win is None:
            print(f"no stall-free {n}-frame window in this take ({take.stats['output_frames']} frames at 16 fps)")
            continue
        start, score = win
        path = os.path.join(out, f"control-{label}-{n}f.mp4")
        write_mp4(take.frames[start:start + n], path)
        controls[label] = (path, n)
        report["windows"][label] = dict(window_info(take, start, n), motion=round(score, 3), file=path)
        print(f"{label}: frames {start}..{start + n - 1} ({start / 16:.2f}s-{(start + n) / 16:.2f}s), "
              f"motion {score:.3f} -> {path}")

    plan = []
    if "hero" in controls:
        plan.append((a.hero, "hero"))
    if "slate" in controls:
        plan += [(c, "slate") for c in costumes]

    if a.no_render or not plan:
        report["planned"] = [f"{c}@{controls[w][1]}f" for c, w in plan]
    else:
        guard = {g.strip().lower() for g in a.gpu_guard.split(",") if g.strip()}
        busy = sorted(guard & running_processes())
        if busy:
            sys.exit(f"GPU in use by {busy}; not rendering (control clips are in {out}). "
                     f"Re-run when it's free, or pass --gpu-guard '' to override.")
        uploaded = {w: comfy_client.upload(a.server, path) for w, (path, _) in controls.items()}
        for costume, w in plan:
            n = controls[w][1]
            tag = f"{costume}-{n}f"
            graph = depth_graph(video=uploaded[w], prompt=COSTUMES[costume], frames=n,
                                prefix=f"openjutsu/{take_name}/{tag}")
            t0 = time.time()
            try:
                files = comfy_client.run(a.server, graph, dest_dir=os.path.join(out, "renders"),
                                         log=lambda m, t=tag: print(f"[{t}] {m}"))
            except comfy_client.ComfyError as e:
                print(f"[{tag}] FAILED: {e}")
                report["renders"].append({"costume": costume, "frames": n, "error": str(e)})
                continue
            sbs = os.path.join(out, f"sbs-{tag}.mp4")
            side_by_side(controls[w][0], files[0], sbs)
            report["renders"].append({"costume": costume, "frames": n, "seconds": round(time.time() - t0),
                                      "render": files[0], "side_by_side": sbs})
            print(f"[{tag}] {files[0]}  |  {sbs}")

    with open(os.path.join(out, "process.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"report: {os.path.join(out, 'process.json')}")
    return 1 if any("error" in r for r in report["renders"]) else 0


if __name__ == "__main__":
    sys.exit(main())
