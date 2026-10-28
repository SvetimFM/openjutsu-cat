import copy
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.validate_workflows import shipped_graphs, validate  # noqa: E402

# Node definitions captured from the ComfyUI v0.42.1 server the entry was built
# on (model-file choices trimmed to the files the workflows use).
with open(os.path.join(ROOT, "tests", "fixtures", "object_info.json"), encoding="utf-8") as f:
    OBJECT_INFO = json.load(f)


class ValidateShipped(unittest.TestCase):
    def test_every_shipped_graph_is_valid(self):
        for name, graph in shipped_graphs().items():
            self.assertEqual(validate(graph, OBJECT_INFO), [], name)


class ValidatorCatchesBreakage(unittest.TestCase):
    def setUp(self):
        self.g = copy.deepcopy(shipped_graphs()["workflow-depth-control-api.json"])

    def assertFlags(self, fragment):
        errs = validate(self.g, OBJECT_INFO)
        self.assertTrue(any(fragment in e for e in errs), f"expected {fragment!r} in {errs}")

    def test_partner_api_node(self):
        # the Open Weights Only category forbids partner nodes / closed APIs;
        # ComfyUI marks those nodes api_node in /object_info
        info = copy.deepcopy(OBJECT_INFO)
        info["KSampler"]["api_node"] = True
        errs = validate(self.g, info)
        self.assertTrue(any("partner/API node" in e for e in errs), errs)

    def test_unknown_node_class(self):
        self.g["1"]["class_type"] = "UnetLoaderGGUF_renamed"
        self.assertFlags("unknown class_type")

    def test_missing_required_input(self):
        del self.g["8"]["inputs"]["steps"]
        self.assertFlags("required input 'steps'")

    def test_link_to_missing_node(self):
        self.g["9"]["inputs"]["samples"] = ["99", 0]
        self.assertFlags("links to missing node 99")

    def test_link_of_wrong_type(self):
        self.g["7"]["inputs"]["control_video"] = ["3", 0]  # VAE into an IMAGE input
        self.assertFlags("expects IMAGE")

    def test_model_file_the_server_does_not_have(self):
        self.g["3"]["inputs"]["vae_name"] = "wan_2.1_vae.safetensors"
        self.assertFlags("not offered by this server")

    def test_bad_save_container(self):
        self.g["14"]["inputs"]["format"] = "avi"
        self.assertFlags("'avi' is not offered")

    def test_uploaded_control_clip_name_is_not_a_fixed_choice(self):
        self.g["12"]["inputs"]["file"] = "my-own-take.mp4"
        self.assertEqual(validate(self.g, OBJECT_INFO), [])


if __name__ == "__main__":
    unittest.main()
