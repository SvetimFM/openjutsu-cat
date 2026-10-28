#!/usr/bin/env python3
"""Check that the model files a pillar needs are installed with the exact byte
size (and optionally sha256) listed in models.json.

  python tools/verify_models.py --comfy-root path/to/ComfyUI
  python tools/verify_models.py --comfy-root ... --extra D:/models     # extra_model_paths base
  python tools/verify_models.py --comfy-root ... --roles depth,cinema --hash

Files are looked up in the order ComfyUI searches: the install's own models/
folder first, then each --extra base. The FIRST copy found is the one ComfyUI
loads, so a truncated copy in the install folder fails the check even when a
good copy sits on the extra drive (a 4.68 GB "OK" download of a 9.8 GB file
once shadowed the real one exactly like that).
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ComfyUI treats these model folders as one search path.
ALIASES = {"unet": ["unet", "diffusion_models"], "diffusion_models": ["diffusion_models", "unet"],
           "text_encoders": ["text_encoders", "clip"], "clip": ["clip", "text_encoders"]}


def load_manifest(path=os.path.join(HERE, "models.json")):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["models"]


def candidates(model, comfy_root, extras):
    _, folder, name = model["path"].split("/")
    folders = ALIASES.get(folder, [folder])
    paths = [os.path.join(comfy_root, "models", d, name) for d in folders]
    for base in extras:
        paths += [os.path.join(base, d, name) for d in folders]
    return paths


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def check(model, comfy_root, extras, do_hash=False):
    """Return (ok, message) for one manifest entry."""
    found = [p for p in candidates(model, comfy_root, extras) if os.path.isfile(p)]
    if not found:
        return False, f"MISSING  {model['path']}  (download: {model['url']})"
    used = found[0]
    size = os.path.getsize(used)
    if size != model["bytes"]:
        msg = f"BAD SIZE {used}: {size:,} bytes, expected {model['bytes']:,}"
        if len(found) > 1:
            msg += f"  (this copy shadows {found[1]})"
        return False, msg
    if do_hash and model.get("sha256"):
        digest = sha256(used)
        if digest != model["sha256"]:
            return False, f"BAD HASH {used}: sha256 {digest}, expected {model['sha256']}"
    return True, f"ok       {used}  {size:,} bytes" + ("  sha256 ok" if do_hash else "")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comfy-root", required=True, help="the ComfyUI folder that contains models/")
    p.add_argument("--extra", action="append", default=[], help="an extra_model_paths base_path (repeatable)")
    p.add_argument("--roles", default="depth,cinema",
                   help="comma list of manifest roles to require: depth, cinema, benchmark, optional, fallback")
    p.add_argument("--hash", action="store_true", help="also verify sha256 (slow: reads every byte)")
    a = p.parse_args(argv)
    roles = set(a.roles.split(","))
    failed = 0
    for model in load_manifest():
        if model["role"] not in roles:
            continue
        ok, msg = check(model, a.comfy_root, a.extra, a.hash)
        print(f"[{model['role']}] {msg}")
        failed += not ok
    print("all required models present and exact" if not failed else f"{failed} problem(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
