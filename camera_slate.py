#!/usr/bin/env python3
"""LTX camera-studio slate: six camera-move prompts on the cinema workflow
(workflow-ltx-cinema-api.json: LTX-Video 2B distilled bf16 + T5-XXL fp16,
768x512, 49 frames, 20 steps euler/simple, cfg 3.0, seed 2077).

  python camera_slate.py            # all six moves -> mp4s in ./out
  python camera_slate.py B D E      # just these
"""
import argparse
import os
import sys

# ComfyUI's portable python (._pth) doesn't put the script's folder on sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import comfy_client  # noqa: E402
from run_local import cinema_graph  # noqa: E402

SLATE = {
  "A": "Slow cinematic dolly-in toward a grey cat sitting on a wooden porch at golden hour, the cat looks calmly into the lens, shallow depth of field, photorealistic",
  "B": "The camera cranes upward from street level, revealing a coastal city at dusk with rooftops, a harbor and warm lights coming on, cinematic aerial reveal, photorealistic",
  "C": "A slow orbital camera move around a lighthouse on rocky cliffs at sunset, waves crashing below, the light beam sweeping, cinematic, photorealistic",
  "D": "A side tracking shot follows a cyclist riding along a tree-lined street in morning light, the camera moving parallel at the same speed, cinematic, photorealistic",
  "E": "An FPV drone dives from high above a green valley down toward a winding river, fast smooth flight, cinematic, photorealistic",
  "F": "A low camera pushes forward through tall golden grass toward an old red barn at sunrise, grass blades brushing past the lens, cinematic, photorealistic",
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("keys", nargs="*", help="slate entries to render, A-F (default: all)")
    p.add_argument("--out", default="out")
    p.add_argument("--server", default=comfy_client.DEFAULT_SERVER)
    a = p.parse_args(argv)
    unknown = [k for k in a.keys if k not in SLATE]
    if unknown:
        p.error(f"unknown slate keys {unknown}; choose from {list(SLATE)}")
    failed = []
    for key in a.keys or list(SLATE):
        print(f"[{key}] {SLATE[key][:70]}...")
        try:
            for path in comfy_client.run(a.server, cinema_graph(prompt=SLATE[key], prefix=f"openjutsu/cam_{key}"),
                                         dest_dir=a.out, log=lambda m, k=key: print(f"[{k}] {m}")):
                print(f"[{key}] saved {path}")
        except comfy_client.ComfyError as e:
            print(f"[{key}] FAILED: {e}")
            failed.append(key)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
