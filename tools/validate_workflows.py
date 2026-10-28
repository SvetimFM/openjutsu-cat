#!/usr/bin/env python3
"""Check the shipped workflows against a ComfyUI server's node definitions
(/object_info) without running them: every node class exists, every required
input is set, every link points at a real output of the right type, and every
fixed choice (sampler, model file, container...) is one the server offers.

  python tools/validate_workflows.py                      # live server
  python tools/validate_workflows.py --object-info f.json # saved snapshot

A missing model file shows up here as an unknown choice for its loader, so
this is also the quickest "is my install complete?" check.
"""
import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import comfy_client  # noqa: E402
from run_local import base_graph, cinema_graph, depth_graph  # noqa: E402

# Inputs whose value is a file uploaded at run time, not a fixed choice.
RUNTIME_FILE_INPUTS = {("LoadVideo", "file")}


def _choices(spec):
    """Allowed values for a combo-style input spec, or None if it isn't one."""
    kind = spec[0]
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if isinstance(kind, list):
        return kind
    if kind == "COMBO":
        return opts.get("options")
    if kind == "COMFY_DYNAMICCOMBO_V3":
        return [o["key"] for o in opts.get("options", [])]
    return None


def validate(graph, object_info):
    errors = []
    for nid, node in graph.items():
        cls = node.get("class_type")
        info = object_info.get(cls)
        if info is None:
            errors.append(f"node {nid}: unknown class_type {cls!r} (missing custom node pack?)")
            continue
        if info.get("api_node"):
            errors.append(f"node {nid}: {cls} is a partner/API node (entry is open weights only)")
        inputs = node.get("inputs", {})
        spec_all = {**info["input"].get("required", {}), **info["input"].get("optional", {})}
        for name in info["input"].get("required", {}):
            if name not in inputs:
                errors.append(f"node {nid} ({cls}): required input {name!r} not set")
        for name, value in inputs.items():
            spec = spec_all.get(name)
            if spec is None:
                errors.append(f"node {nid} ({cls}): unknown input {name!r}")
                continue
            if isinstance(value, list):
                src_id, idx = value
                src = graph.get(src_id)
                if src is None:
                    errors.append(f"node {nid} ({cls}).{name}: links to missing node {src_id}")
                    continue
                src_info = object_info.get(src["class_type"])
                if src_info is None:
                    continue  # already reported as unknown class
                outs = src_info["output"]
                if idx >= len(outs):
                    errors.append(f"node {nid} ({cls}).{name}: {src['class_type']} has no output {idx}")
                    continue
                want = spec[0]
                if isinstance(want, str) and want != "*" and outs[idx] != "*" \
                        and outs[idx] not in want.split(","):
                    errors.append(f"node {nid} ({cls}).{name}: expects {want}, "
                                  f"gets {outs[idx]} from node {src_id} ({src['class_type']})")
            elif (cls, name) not in RUNTIME_FILE_INPUTS:
                allowed = _choices(spec)
                if allowed is not None and value not in allowed:
                    errors.append(f"node {nid} ({cls}).{name}: {value!r} is not offered by this server")
    return errors


def shipped_graphs():
    return {"workflow-depth-control-api.json": depth_graph(),
            "workflow-ltx-cinema-api.json": cinema_graph(),
            "run_local.py base (benchmark)": base_graph()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default=comfy_client.DEFAULT_SERVER)
    p.add_argument("--object-info", help="validate against a saved /object_info JSON instead of a server")
    p.add_argument("--skip-base", action="store_true", help="don't require the TI2V benchmark model")
    a = p.parse_args(argv)
    if a.object_info:
        with open(a.object_info, encoding="utf-8") as f:
            info = json.load(f)
    else:
        with urllib.request.urlopen(f"{a.server}/object_info", timeout=60) as r:
            info = json.load(r)
    failed = False
    for name, graph in shipped_graphs().items():
        if a.skip_base and name.startswith("run_local.py base"):
            continue
        errs = validate(graph, info)
        print(f"{'OK  ' if not errs else 'FAIL'} {name}")
        for e in errs:
            print(f"     {e}")
        failed |= bool(errs)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
