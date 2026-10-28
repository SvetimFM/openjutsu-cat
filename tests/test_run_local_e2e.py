"""run_local.py end to end against a fake ComfyUI HTTP server: upload, submit,
poll, filter outputs, download. No GPU, no real ComfyUI."""
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import comfy_client  # noqa: E402
import run_local  # noqa: E402

VIDEO_BYTES = b"\x00\x00\x00\x18ftypmp42 fake rendered video"


class FakeComfy(BaseHTTPRequestHandler):
    state = {}

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path == "/upload/image":
            name = re.search(rb'name="image"; filename="([^"]+)"', body).group(1).decode()
            self.state["uploaded"] = name
            self.state["upload_type"] = re.search(rb'name="type"\r\n\r\n(\w+)', body).group(1).decode()
            return self._json(200, {"name": name, "subfolder": "", "type": "input"})
        if self.path == "/prompt":
            graph = json.loads(body)["prompt"]
            self.state["graph"] = graph
            if self.state.get("reject"):
                return self._json(400, {"error": "Prompt outputs failed validation",
                                        "node_errors": {"1": {"errors": [{"message": "model missing"}]}}})
            return self._json(200, {"prompt_id": "pid-1", "number": 1, "node_errors": {}})
        self._json(404, {})

    def do_GET(self):
        if self.path.startswith("/history/"):
            self.state["polls"] = self.state.get("polls", 0) + 1
            if self.state["polls"] < 2:
                return self._json(200, {})  # still queued on the first poll
            if self.state.get("fail"):
                return self._json(200, {"pid-1": {"status": {"status_str": "error", "completed": False, "messages": [
                    ["execution_error", {"exception_message": "CUDA out of memory"}]]}, "outputs": {}}})
            return self._json(200, {"pid-1": {"status": {"status_str": "success", "completed": True}, "outputs": {
                "12": {"images": [{"filename": self.state.get("uploaded", "x.mp4"), "subfolder": "", "type": "input"}],
                       "animated": [True]},
                "14": {"images": [{"filename": "depth_00001_.mp4", "subfolder": "openjutsu", "type": "output"}],
                       "animated": [True]}}}})
        if self.path.startswith("/view?"):
            q = urllib.parse.parse_qs(self.path.split("?", 1)[1])
            self.state["viewed"] = (q["filename"][0], q["subfolder"][0], q["type"][0])
            self.send_response(200)
            self.send_header("Content-Length", str(len(VIDEO_BYTES)))
            self.end_headers()
            self.wfile.write(VIDEO_BYTES)
            return
        self._json(404, {})


class RunLocalEndToEnd(unittest.TestCase):
    def setUp(self):
        FakeComfy.state = {}
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeComfy)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}"
        self.tmp = tempfile.TemporaryDirectory()
        self._wait = comfy_client.wait
        comfy_client.wait = lambda *a, **k: self._wait(*a, **dict(k, poll=0.01))

    def tearDown(self):
        comfy_client.wait = self._wait
        self.srv.shutdown()
        self.srv.server_close()
        self.tmp.cleanup()

    def run_cli(self, *args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = run_local.main(list(args) + ["--server", self.url, "--out", self.tmp.name])
        return rc, buf.getvalue()

    def test_depth_uploads_clip_submits_shipped_graph_and_downloads_mp4(self):
        rc, out = self.run_cli("depth", "--prompt", "ink cat")
        self.assertEqual(rc, 0, out)
        st = FakeComfy.state
        self.assertEqual(st["uploaded"], "test-depth-640x640-49f.mp4")
        self.assertEqual(st["upload_type"], "input")
        self.assertEqual(st["graph"]["12"]["inputs"]["file"], "test-depth-640x640-49f.mp4")
        self.assertEqual(st["graph"]["4"]["inputs"]["text"], "ink cat")
        # only the saved output is downloaded, not the echoed input clip
        self.assertEqual(st["viewed"], ("depth_00001_.mp4", "openjutsu", "output"))
        with open(os.path.join(self.tmp.name, "depth_00001_.mp4"), "rb") as f:
            self.assertEqual(f.read(), VIDEO_BYTES)
        self.assertEqual(os.listdir(self.tmp.name), ["depth_00001_.mp4"])

    def test_cinema_submits_without_upload(self):
        rc, _ = self.run_cli("cinema", "--frames", "97")
        self.assertEqual(rc, 0)
        self.assertNotIn("uploaded", FakeComfy.state)
        self.assertEqual(FakeComfy.state["graph"]["10"]["inputs"]["length"], 97)

    def test_server_rejection_raises_with_node_errors(self):
        FakeComfy.state["reject"] = True
        with self.assertRaises(comfy_client.ComfyError) as cm:
            self.run_cli("depth")
        self.assertIn("model missing", str(cm.exception))

    def test_execution_error_raises_with_message(self):
        FakeComfy.state["fail"] = True
        with self.assertRaises(comfy_client.ComfyError) as cm:
            self.run_cli("cinema")
        self.assertIn("out of memory", str(cm.exception))

    def test_bad_frame_count_is_refused_before_anything_is_sent(self):
        with self.assertRaises(ValueError):
            self.run_cli("depth", "--frames", "50")
        self.assertEqual(FakeComfy.state, {})


if __name__ == "__main__":
    unittest.main()
