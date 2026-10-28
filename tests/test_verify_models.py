import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import verify_models  # noqa: E402

MODEL = {"path": "models/text_encoders/t5xxl_fp16.safetensors", "bytes": 4096, "sha256": None,
         "url": "https://example.invalid/t5", "role": "cinema"}


def make(path, size):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.truncate(size)


class VerifyModels(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.comfy = os.path.join(self.tmp.name, "ComfyUI")
        self.extra = os.path.join(self.tmp.name, "big-drive")

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_size_in_install_passes(self):
        make(os.path.join(self.comfy, "models", "text_encoders", "t5xxl_fp16.safetensors"), 4096)
        ok, msg = verify_models.check(MODEL, self.comfy, [])
        self.assertTrue(ok, msg)

    def test_missing_file_fails_with_download_url(self):
        ok, msg = verify_models.check(MODEL, self.comfy, [self.extra])
        self.assertFalse(ok)
        self.assertIn("MISSING", msg)
        self.assertIn(MODEL["url"], msg)

    def test_truncated_file_fails(self):
        make(os.path.join(self.comfy, "models", "text_encoders", "t5xxl_fp16.safetensors"), 2048)
        ok, msg = verify_models.check(MODEL, self.comfy, [])
        self.assertFalse(ok)
        self.assertIn("BAD SIZE", msg)

    def test_good_copy_on_extra_drive_passes(self):
        make(os.path.join(self.extra, "text_encoders", "t5xxl_fp16.safetensors"), 4096)
        ok, msg = verify_models.check(MODEL, self.comfy, [self.extra])
        self.assertTrue(ok, msg)

    def test_truncated_install_copy_shadows_good_extra_copy(self):
        # The FAT32 failure: a partial in the install folder wins over the good file.
        make(os.path.join(self.comfy, "models", "text_encoders", "t5xxl_fp16.safetensors"), 2048)
        make(os.path.join(self.extra, "text_encoders", "t5xxl_fp16.safetensors"), 4096)
        ok, msg = verify_models.check(MODEL, self.comfy, [self.extra])
        self.assertFalse(ok)
        self.assertIn("shadows", msg)

    def test_unet_and_diffusion_models_are_one_search_path(self):
        model = dict(MODEL, path="models/unet/x.gguf")
        make(os.path.join(self.comfy, "models", "diffusion_models", "x.gguf"), 4096)
        ok, msg = verify_models.check(model, self.comfy, [])
        self.assertTrue(ok, msg)

    def test_hash_mismatch_fails(self):
        make(os.path.join(self.comfy, "models", "text_encoders", "t5xxl_fp16.safetensors"), 4096)
        ok, msg = verify_models.check(dict(MODEL, sha256="0" * 64), self.comfy, [], do_hash=True)
        self.assertFalse(ok)
        self.assertIn("BAD HASH", msg)


class Manifest(unittest.TestCase):
    def test_manifest_entries_are_complete(self):
        models = verify_models.load_manifest()
        self.assertEqual(len({m["path"] for m in models}), len(models))
        for m in models:
            self.assertRegex(m["path"], r"^models/[a-z_]+/[^/]+$")
            self.assertGreater(m["bytes"], 0)
            self.assertRegex(m["sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(m["url"].startswith("https://huggingface.co/"), m["url"])
            self.assertIn(m["role"], {"depth", "cinema", "benchmark", "optional", "fallback"})

    def test_every_model_a_shipped_graph_loads_is_in_the_manifest(self):
        from tools.validate_workflows import shipped_graphs
        names = {m["path"].rsplit("/", 1)[1]: m["role"] for m in verify_models.load_manifest()}
        for gname, graph in shipped_graphs().items():
            for node in graph.values():
                for key in ("unet_name", "clip_name", "vae_name", "ckpt_name"):
                    if key in node["inputs"]:
                        self.assertIn(node["inputs"][key], names, f"{gname}: {key}")


if __name__ == "__main__":
    unittest.main()
