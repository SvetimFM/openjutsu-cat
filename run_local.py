#!/usr/bin/env python3
"""Run the Openjutsu workflows against a running ComfyUI server.

  python run_local.py depth                       # example depth clip -> mp4 in ./out
  python run_local.py depth --video my-cat.mp4 --prompt "A tiny cat knight ..."
  python run_local.py cinema --prompt "The camera cranes upward ..."
  python run_local.py base                        # TI2V-5B text-to-video benchmark

Every graph is built from the shipped workflow JSON (workflow-*-api.json); the
flags only change the prompt, control clip, frame count, seed and output name.
The control clip is uploaded for you, so ComfyUI can run on another machine.
Stdlib only: runs under ComfyUI's embedded python or any python >= 3.8.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# ComfyUI's portable python (._pth) doesn't put the script's folder on sys.path.
sys.path.insert(0, HERE)

import comfy_client  # noqa: E402
DEPTH_WORKFLOW = os.path.join(HERE, "workflow-depth-control-api.json")
CINEMA_WORKFLOW = os.path.join(HERE, "workflow-ltx-cinema-api.json")
EXAMPLE_CLIP = os.path.join(HERE, "examples", "test-depth-640x640-49f.mp4")
BASE_PROMPT = ("A fluffy grey Russian Blue cat walks slowly across a sunlit living room floor, "
               "soft cinematic light, photorealistic, smooth motion")


def load_workflow(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_frames(frames):
    # Wan and LTX both compress time 4x in the VAE: lengths must be 4n+1.
    if frames < 5 or (frames - 1) % 4:
        raise ValueError(f"--frames must be 4n+1 (e.g. 49, 97), got {frames}")


def depth_graph(video=None, prompt=None, frames=None, seed=None, prefix=None):
    """The depth-control workflow, with optional overrides. No overrides returns
    the shipped JSON unchanged."""
    g = load_workflow(DEPTH_WORKFLOW)
    if video is not None:
        g["12"]["inputs"]["file"] = video
    if prompt is not None:
        g["4"]["inputs"]["text"] = prompt
    if frames is not None:
        check_frames(frames)
        g["7"]["inputs"]["length"] = frames
    if seed is not None:
        g["8"]["inputs"]["seed"] = seed
    if prefix is not None:
        g["14"]["inputs"]["filename_prefix"] = prefix
    return g


def cinema_graph(prompt=None, frames=None, seed=None, prefix=None):
    """The LTX cinema-camera workflow, with optional overrides."""
    g = load_workflow(CINEMA_WORKFLOW)
    if prompt is not None:
        g["4"]["inputs"]["text"] = prompt
    if frames is not None:
        check_frames(frames)
        g["10"]["inputs"]["length"] = frames
    if seed is not None:
        g["8"]["inputs"]["seed"] = seed
    if prefix is not None:
        g["14"]["inputs"]["filename_prefix"] = prefix
    return g


def base_graph(prompt=None, frames=None, seed=None, prefix=None):
    """Benchmark only: Wan2.2 TI2V-5B text-to-video on the same loaders and
    sampler as the depth workflow, with the control input removed."""
    g = depth_graph(prompt=prompt or BASE_PROMPT, frames=frames, seed=seed,
                    prefix=prefix or "openjutsu/base")
    del g["12"], g["13"]
    g["1"]["inputs"]["unet_name"] = "Wan2.2-TI2V-5B-Q5_K_M.gguf"
    cond = g["7"]["inputs"]
    g["7"] = {"class_type": "WanImageToVideo",
              "inputs": {"positive": cond["positive"], "negative": cond["negative"], "vae": cond["vae"],
                         "width": 832, "height": 480, "length": cond["length"], "batch_size": 1}}
    g["11"]["inputs"]["fps"] = 24.0
    return g


def clip_frames(path):
    """Frame count of a local video via ffprobe, or None when ffprobe is absent."""
    if not shutil.which("ffprobe"):
        return None
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
                         capture_output=True, text=True)
    try:
        return int(out.stdout.strip())
    except ValueError:
        return None


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["depth", "cinema", "base", "c", "a", "b"],
                   help="depth = cat performance transfer, cinema = LTX camera move, "
                        "base = TI2V benchmark (c/a/b are the old aliases)")
    p.add_argument("--video", default=EXAMPLE_CLIP, help="depth mode: local control clip (near = bright)")
    p.add_argument("--prompt", help="positive prompt (defaults to the one in the workflow JSON)")
    p.add_argument("--frames", type=int, help="frame count, 4n+1 (default: the workflow's 49)")
    p.add_argument("--seed", type=int, help="sampler seed (default: the workflow's 2077)")
    p.add_argument("--prefix", help="output filename prefix on the server")
    p.add_argument("--out", default="out", help="local folder to download the result into")
    p.add_argument("--server", default=comfy_client.DEFAULT_SERVER)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    mode = {"c": "depth", "a": "cinema", "b": "base"}.get(a.mode, a.mode)

    if a.frames is not None:
        check_frames(a.frames)
    if mode == "depth":
        have = clip_frames(a.video)
        want = a.frames or 49
        if have is not None and have < want:
            raise ValueError(f"{a.video} has {have} frames but {want} were asked for: the control clip "
                             f"must cover every generated frame (lower --frames or use a longer clip)")
        name = comfy_client.upload(a.server, a.video)
        print(f"uploaded {a.video} as {name}")
        graph = depth_graph(video=name, prompt=a.prompt, frames=a.frames, seed=a.seed, prefix=a.prefix)
    elif mode == "cinema":
        graph = cinema_graph(prompt=a.prompt, frames=a.frames, seed=a.seed, prefix=a.prefix)
    else:
        graph = base_graph(prompt=a.prompt, frames=a.frames, seed=a.seed, prefix=a.prefix)

    for path in comfy_client.run(a.server, graph, dest_dir=a.out):
        print(f"saved {path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (comfy_client.ComfyError, ValueError, OSError) as e:
        sys.exit(f"error: {e}")
