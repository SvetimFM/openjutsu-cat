"""Capture tools: timestamp resampling, stall handling, motion windows, and the
encode -> pick -> process pipeline on synthetic KinectBurst takes. Needs numpy
(and ffmpeg for the end-to-end cases); skipped where they're missing."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "capture"))

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

HAVE_FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")
W, H = 512, 424


def write_take(d, ts_ms, mover=None, floor_mm=2000, cat_mm=900):
    """A KinectBurst-format take: fNNN.bin (uint16 mm) + burst-log.csv.
    mover(i) -> x position of a 60x60 'cat' in frame i, or None for an empty room."""
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "burst-log.csv"), "w") as log:
        log.write("index,unix_ms,min_mm,mean_mm,valid_px\n")
        for i, t in enumerate(ts_ms):
            depth = np.full((H, W), floor_mm, dtype="<u2")
            depth[:, :8] = 0  # invalid border, like the real sensor
            x = mover(i) if mover else None
            if x is not None:
                depth[300:360, x:x + 60] = cat_mm
            depth.tofile(os.path.join(d, f"f{i:03d}.bin"))
            log.write(f"{i},{t},0,0,0\n")


def ffprobe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    return s["width"], s["height"], s["r_frame_rate"], int(s["nb_read_frames"])


@unittest.skipIf(np is None, "numpy not installed")
class Resample(unittest.TestCase):
    def setUp(self):
        from encode_depth import resample
        self.resample = resample

    def test_slow_capture_is_stretched_to_real_time(self):
        # 94 ms per frame (what 62 ms requested actually gives): 100 frames = 9.3 s
        ts = [1000 + 94 * i for i in range(100)]
        picks, blocked = self.resample(ts, fps=16)
        self.assertEqual(len(picks), int(9306 // 62.5) + 1)  # one output frame per 62.5 ms of real time
        self.assertEqual(picks, sorted(picks))
        self.assertEqual(picks[0], 0)
        self.assertEqual(picks[-1], 98)  # last tick 9250 ms: frame 98 (9212) is nearer than 99 (9306)
        self.assertFalse(any(blocked))

    def test_picks_the_nearest_frame(self):
        ts = [0, 100, 200]
        picks, _ = self.resample(ts, fps=16)  # ticks 0, 62.5, 125, 187.5
        self.assertEqual(picks, [0, 1, 1, 2])

    def test_stall_is_blocked_not_hidden(self):
        ts = [62 * i for i in range(50)] + [62 * 49 + 4358 + 62 * i for i in range(50)]
        picks, blocked = self.resample(ts, fps=16, max_gap_ms=150)
        stalled = [k for k, b in enumerate(blocked) if b]
        self.assertGreaterEqual(len(stalled), 4358 // 63)
        self.assertFalse(any(blocked[:45]))
        self.assertFalse(any(blocked[-45:]))


@unittest.skipIf(np is None, "numpy not installed")
class Windows(unittest.TestCase):
    def test_best_window_finds_the_motion(self):
        from pick_window import best_window, motion_scores
        frames = np.full((300, 64, 64), 100, dtype=np.uint8)
        for i in range(100, 150):  # a square crosses only in frames 100..149
            frames[i, 20:30, i - 100:i - 90] = 200
        start, score = best_window(motion_scores(frames), 49)
        self.assertTrue(98 <= start <= 102, start)
        self.assertGreater(score, 0)

    def test_blocked_frames_are_never_inside_the_window(self):
        from pick_window import best_window
        scores = np.zeros(199)
        scores[100:150] = 1.0
        blocked = [False] * 200
        blocked[120] = True  # a stall right in the middle of the best motion
        start, _ = best_window(scores, 49, blocked)
        self.assertFalse(any(blocked[start:start + 49]), start)

    def test_too_short_take_has_no_window(self):
        from pick_window import best_window
        self.assertIsNone(best_window(np.zeros(10), 49))
        self.assertIsNone(best_window(np.zeros(99), 49, [True] * 100))

    def test_sensor_speckle_on_invalid_pixels_is_not_motion(self):
        from pick_window import motion_scores
        a = np.zeros((2, 32, 32), dtype=np.uint8)
        a[1, :, :16] = 255  # invalid (0) -> bright: dropout, not a performer
        self.assertEqual(motion_scores(a)[0], 0.0)


@unittest.skipIf(np is None or not HAVE_FFMPEG, "needs numpy + ffmpeg")
class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.take = os.path.join(self.tmp.name, "luna-01")
        # 400 frames at the real 28 ms setting's cadence (~63 ms), cat crosses in frames 200..279
        ts = [1_700_000_000_000 + 63 * i for i in range(400)]
        write_take(self.take, ts, mover=lambda i: 40 + (i - 200) * 5 if 200 <= i < 280 else None)

    def tearDown(self):
        self.tmp.cleanup()

    def test_encode_cli_writes_16fps_640_control_clip(self):
        out = os.path.join(self.tmp.name, "c.mp4")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "capture", "encode_depth.py"), self.take, out],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        w, h, fps, n = ffprobe(out)
        self.assertEqual((w, h, fps), (640, 640, "16/1"))
        self.assertEqual(n, int(399 * 63 // 62.5) + 1)  # real-time length, not 400

    def test_process_take_no_render_picks_the_crossing(self):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "capture", "process_take.py"), self.take,
                            "--no-render"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        (proc,) = [d for d in os.listdir(self.take) if d.startswith("process-")]
        with open(os.path.join(self.take, proc, "process.json")) as f:
            rep = json.load(f)
        for label, n in (("hero", 97), ("slate", 49)):
            win = rep["windows"][label]
            lo, hi = win["source_frames"]
            # the window must overlap the crossing (source frames 200..279)
            self.assertTrue(lo < 280 and hi > 200, (label, win))
            self.assertEqual(ffprobe(win["file"])[3], n)
        self.assertEqual(rep["planned"], ["D@97f", "D@49f", "C@49f", "A@49f", "F@49f"])

    def test_process_take_never_reuses_an_output_folder(self):
        cmd = [sys.executable, os.path.join(ROOT, "capture", "process_take.py"), self.take, "--no-render"]
        subprocess.run(cmd, capture_output=True, check=True)
        import time
        time.sleep(1.1)
        subprocess.run(cmd, capture_output=True, check=True)
        self.assertEqual(len([d for d in os.listdir(self.take) if d.startswith("process-")]), 2)


if __name__ == "__main__":
    unittest.main()
