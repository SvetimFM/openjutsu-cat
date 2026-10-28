import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import run_local  # noqa: E402


def shipped(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return json.load(f)


class GraphParity(unittest.TestCase):
    """run_local.py must submit exactly the shipped workflow JSON unless a flag
    changes it, so the published workflow is the one the numbers came from."""

    def test_depth_defaults_are_the_shipped_json(self):
        self.assertEqual(run_local.depth_graph(), shipped("workflow-depth-control-api.json"))

    def test_cinema_defaults_are_the_shipped_json(self):
        self.assertEqual(run_local.cinema_graph(), shipped("workflow-ltx-cinema-api.json"))

    def test_gate_settings_are_pinned(self):
        g = shipped("workflow-depth-control-api.json")
        self.assertEqual(g["1"]["inputs"]["unet_name"], "Wan2.2-Fun-5B-Control-Q5_K_M.gguf")
        self.assertEqual((g["7"]["inputs"]["width"], g["7"]["inputs"]["height"], g["7"]["inputs"]["length"]),
                         (640, 640, 49))
        s = g["8"]["inputs"]
        self.assertEqual((s["seed"], s["steps"], s["cfg"], s["sampler_name"], s["scheduler"]),
                         (2077, 20, 5.0, "euler", "simple"))
        self.assertEqual(g["6"]["inputs"]["shift"], 8.0)

    def test_outputs_are_mp4_at_the_control_frame_rate(self):
        for name, fps in [("workflow-depth-control-api.json", 16.0), ("workflow-ltx-cinema-api.json", 25.0)]:
            g = shipped(name)
            self.assertEqual(g["11"]["class_type"], "CreateVideo", name)
            self.assertEqual(g["11"]["inputs"]["fps"], fps, name)
            self.assertEqual(g["14"]["class_type"], "SaveVideo", name)
            self.assertEqual(g["14"]["inputs"]["format"], "mp4", name)
            self.assertNotIn("SaveImage", {n["class_type"] for n in g.values()}, name)


class Overrides(unittest.TestCase):
    def test_depth_overrides_land_on_the_right_nodes(self):
        g = run_local.depth_graph(video="take3.mp4", prompt="ink cat", frames=97, seed=7, prefix="p/x")
        self.assertEqual(g["12"]["inputs"]["file"], "take3.mp4")
        self.assertEqual(g["4"]["inputs"]["text"], "ink cat")
        self.assertEqual(g["7"]["inputs"]["length"], 97)
        self.assertEqual(g["8"]["inputs"]["seed"], 7)
        self.assertEqual(g["14"]["inputs"]["filename_prefix"], "p/x")
        # nothing else moved
        base = shipped("workflow-depth-control-api.json")
        for nid in set(base) - {"12", "4", "7", "8", "14"}:
            self.assertEqual(g[nid], base[nid], nid)

    def test_cinema_overrides_land_on_the_right_nodes(self):
        g = run_local.cinema_graph(prompt="crane up", frames=97, seed=3, prefix="c/y")
        self.assertEqual(g["4"]["inputs"]["text"], "crane up")
        self.assertEqual(g["10"]["inputs"]["length"], 97)
        self.assertEqual(g["8"]["inputs"]["seed"], 3)
        self.assertEqual(g["14"]["inputs"]["filename_prefix"], "c/y")

    def test_frames_must_be_4n_plus_1(self):
        for bad in (48, 50, 1, 0, 3):
            with self.assertRaises(ValueError, msg=bad):
                run_local.depth_graph(frames=bad)
        for good in (5, 49, 97):
            run_local.check_frames(good)

    def test_base_graph_has_no_control_input(self):
        g = run_local.base_graph()
        self.assertNotIn("12", g)
        self.assertNotIn("13", g)
        self.assertEqual(g["7"]["class_type"], "WanImageToVideo")
        self.assertEqual(g["1"]["inputs"]["unet_name"], "Wan2.2-TI2V-5B-Q5_K_M.gguf")
        linked = {v[0] for n in g.values() for v in n["inputs"].values() if isinstance(v, list)}
        self.assertTrue(linked <= set(g), f"dangling links to {linked - set(g)}")


if __name__ == "__main__":
    unittest.main()
