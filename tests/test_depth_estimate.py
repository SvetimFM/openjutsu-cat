"""capture/depth_estimate.py end to end, with a stand-in depth model (no torch,
no model download): output spec, and no leftovers from an earlier run."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "capture", "depth_estimate.py")

FAKE_TORCH = """class cuda:
    @staticmethod
    def is_available():
        return False
"""
# "depth" = the frame's own brightness, so the output still follows the input
FAKE_TRANSFORMERS = """import numpy as np
from PIL import Image

def pipeline(task, model=None, device=None):
    def run(img):
        return {"depth": Image.fromarray(np.asarray(img.convert("L")))}
    return run
"""

try:
    import PIL  # noqa: F401  (the script itself needs pillow)
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False


@unittest.skipUnless(shutil.which("ffmpeg") and HAVE_PIL, "needs ffmpeg and pillow")
class DepthEstimate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.fakes = os.path.join(self.tmp, "fakes")
        os.makedirs(self.fakes)
        for name, body in (("torch.py", FAKE_TORCH), ("transformers.py", FAKE_TRANSFORMERS)):
            with open(os.path.join(self.fakes, name), "w") as f:
                f.write(body)
        self.src = os.path.join(self.tmp, "phone clip.mov")
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=480x270:rate=30",
                        "-t", "4", "-pix_fmt", "yuv420p", self.src], check=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_script(self, *args):
        # runpy, not PYTHONPATH: ComfyUI's portable python ignores PYTHONPATH
        boot = (f"import sys, runpy; sys.path.insert(0, {self.fakes!r}); "
                f"sys.argv = [{SCRIPT!r}] + {list(args)!r}; runpy.run_path({SCRIPT!r}, run_name='__main__')")
        r = subprocess.run([sys.executable, "-c", boot], capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def probe(self, path):
        out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                              "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate",
                              "-of", "json", path], capture_output=True, text=True, check=True).stdout
        return json.loads(out)["streams"][0]

    def test_writes_the_control_spec(self):
        out = os.path.join(self.tmp, "depth.mp4")
        self.run_script(self.src, out, "0.5", "17")
        s = self.probe(out)
        self.assertEqual((s["width"], s["height"], s["r_frame_rate"], s["nb_read_frames"]),
                         (640, 640, "16/1", "17"))

    def test_second_run_into_the_same_folder_uses_only_its_own_frames(self):
        out = os.path.join(self.tmp, "depth.mp4")
        self.run_script(self.src, out, "0", "33")
        self.run_script(self.src, out, "0", "9")
        self.assertEqual(self.probe(out)["nb_read_frames"], "9")

    def test_leaves_nothing_behind_next_to_the_output(self):
        out_dir = os.path.join(self.tmp, "out")
        os.makedirs(out_dir)
        self.run_script(self.src, os.path.join(out_dir, "depth.mp4"), "0", "9")
        self.assertEqual(os.listdir(out_dir), ["depth.mp4"])


if __name__ == "__main__":
    unittest.main()
