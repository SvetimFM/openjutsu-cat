# Minimal ComfyUI HTTP client shared by run_local.py, camera_slate.py and the
# capture tools: upload an input file, submit an API-format graph, wait for it,
# download what it saved. Stdlib only, so it runs under ComfyUI's embedded python.
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

DEFAULT_SERVER = "http://127.0.0.1:8188"


class ComfyError(RuntimeError):
    pass


def _request(url, data=None, headers=None, timeout=60):
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise ComfyError(f"{url} -> HTTP {e.code}: {body[:2000]}") from None


def upload(server, path):
    """Upload a local file into ComfyUI's input/ folder and return the name to
    reference it by. Re-uploading identical content reuses the same name."""
    boundary = uuid.uuid4().hex
    name = os.path.basename(path)
    with open(path, "rb") as f:
        payload = f.read()
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\nfalse\r\n".encode(),
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
         "Content-Type: application/octet-stream\r\n\r\n").encode() + payload + b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    raw = _request(f"{server}/upload/image", data=b"".join(parts),
                   headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, timeout=300)
    resp = json.loads(raw)
    return f"{resp['subfolder']}/{resp['name']}" if resp.get("subfolder") else resp["name"]


def submit(server, graph):
    raw = _request(f"{server}/prompt", data=json.dumps({"prompt": graph}).encode(),
                   headers={"Content-Type": "application/json"})
    resp = json.loads(raw)
    if resp.get("node_errors"):
        raise ComfyError("node errors: " + json.dumps(resp["node_errors"])[:2000])
    return resp["prompt_id"]


def wait(server, prompt_id, timeout=3600, poll=5, log=print):
    """Poll /history until the prompt finishes; return its history entry."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        hist = json.loads(_request(f"{server}/history/{prompt_id}", timeout=30))
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = [str(m)[:600] for m in status.get("messages", []) if m and m[0] == "execution_error"]
                raise ComfyError("execution error: " + " | ".join(msgs))
            if status.get("completed"):
                log(f"done in {int(time.time() - t0)} s")
                return entry
        time.sleep(poll)
    raise ComfyError(f"timed out after {timeout} s waiting for {prompt_id}")


def saved_files(entry):
    """Files the graph saved (type == output). Inputs echoed back by load nodes
    (type == input) are skipped."""
    files = []
    for node_out in entry.get("outputs", {}).values():
        for key in ("images", "videos", "gifs"):
            for item in node_out.get(key, []):
                if item.get("type") == "output":
                    files.append(item)
    return files


def download(server, item, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    q = urllib.parse.urlencode({"filename": item["filename"], "subfolder": item.get("subfolder", ""),
                                "type": item.get("type", "output")})
    data = _request(f"{server}/view?{q}", timeout=300)
    dest = os.path.join(dest_dir, item["filename"])
    with open(dest, "wb") as f:
        f.write(data)
    return dest


def run(server, graph, dest_dir=None, timeout=3600, log=print):
    """Submit, wait, and (optionally) download every saved file. Returns local
    paths when dest_dir is given, otherwise the server-side file records."""
    pid = submit(server, graph)
    log(f"submitted prompt_id={pid}")
    entry = wait(server, pid, timeout=timeout, log=log)
    files = saved_files(entry)
    if dest_dir is None:
        return files
    return [download(server, item, dest_dir) for item in files]
